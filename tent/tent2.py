from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from torch import Tensor
from operator import itemgetter
from utils import map_
import torch.nn.functional as F


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.losses = eval(args.target_losses)
        self.loss_fns = []
        for loss_name, loss_params, type_bounds, bounds_params, fn, _ in self.losses:
            loss_class = getattr(__import__('losses'), loss_name)
            self.loss_fns.append(loss_class(**loss_params, dtype=torch.float32, fn=fn))

        self.loss_weights = map_(itemgetter(5), self.losses)

    def forward(self, x, y, bounds):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, y, bounds, self.model, self.optimizer, self.loss_fns, self.loss_weights,
                                        self.device, self.args)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


#@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


#@torch.jit.script
def softmax_entropy_seg(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    weights = [0.1, 0.9]
    log_p = (probs[:, ...] + 1e-10).log()
    mask = probs[:, ...]
    mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, torch.tensor(weights).to(mask.device)])
    loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
    loss /= mask.sum() + 1e-10
    return loss


#@torch.jit.script
def cross_entropy_seg(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""

    log_p = (probs[:, ...] + 1e-10).log()
    mask = y[:, ...]
    loss = - torch.einsum("bcwh,bcwh->", [mask, log_p])
    loss /= mask.sum() + 1e-10
    return loss


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, y, bounds, model, optimizer, loss_fns, loss_weights, device, args):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    pred_probs = F.softmax(outputs, dim=1)
    # adapt
    loss: Tensor = torch.zeros(1, requires_grad=True).to(device)

    for loss_fn, label, w, bound in zip(loss_fns, [y], loss_weights, bounds):
        if w > 0:
            if eval(args.target_losses)[0][0] == "EntKLProp" or eval(args.target_losses)[0][0] == "EntKLPropWComp":
                loss_1, loss_cons_prior, est_prop = loss_fn(pred_probs, label, bound)
                loss_tmp = loss_1 + loss_cons_prior
                loss = loss + w * loss_tmp
                '''
                if args.n_warmup > epc:
                    loss_tmp = loss_1
                else:
                    loss_tmp = loss_1 + loss_cons_prior
                '''
            else:
                loss_tmp = loss_fn(x, label, bound)
                loss = loss + w * loss_tmp

    # loss = softmax_entropy_seg(outputs).mean(0)
    # loss = cross_entropy_seg(outputs,y).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
