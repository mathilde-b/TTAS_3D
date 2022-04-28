from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from torch import Tensor
import torch.nn.functional as F

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x,y):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, y, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy_seg(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #weights = [0.1,0.9]
    #weights = [0.9,0.1]
    _,C,_,_ = probs.shape
    weights = [1]*C
    #print("probs is nan",torch.einsum("bcwh->",torch.isnan(probs)))
    #print("probs is neg",torch.einsum("bcwh->",probs<0))
    log_p = (probs[:, ...] + 1e-10).log()
    #print("log p is nan",torch.einsum("bcwh->",torch.isnan(log_p)))
    #print("log p not is nan",torch.einsum("bcwh->",~torch.isnan(log_p)))
    mask = probs[:, ...]
    mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, torch.tensor(weights).to(mask.device)])
    #print(torch.einsum("bcwh->", torch.isnan(mask_weighted)))
    loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
    #print("loss",loss)
    loss /= mask.sum() + 1e-10
    #print(loss)
    return loss

def cross_entropy_seg(probs: torch.Tensor,y: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    _,C,_,_ = probs.shape
    weights = [0.1]+[0.9]*(C-1)
    log_p = (probs[:, ...] + 1e-10).log()
    mask = y[:, ...]
    #print(torch.einsum("bcwh->c", [mask[:,1:,...]]))
    loss = - torch.einsum("bcwh,bcwh->", [mask, log_p])
    loss /= mask.sum() + 1e-10
    return loss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, y, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    #print(len(model(x)))
    #outputs,aux, aux2, aux3 = model(x)
    pred_probs = F.softmax(outputs, dim=1)
    # adapt
    #loss = softmax_entropy_seg(pred_probs).mean(0)
    #params, param_names = collect_params(model)
    loss = cross_entropy_seg(outputs, y.type((torch.float32))).mean(0)
    #print("before backward", params[0])
    loss.backward()
    optimizer.step()
    #print("after optimizer step", params[0])
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
