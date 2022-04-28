#!/usr/env/bin python3.6

from typing import List, Tuple,cast
# from functools import reduce
from operator import add
from functools import reduce
import numpy as np
import torch
from torch import einsum
from torch import Tensor
import pandas as pd
import torch.nn.functional as F
from operator import mul
from utils import soft_compactness, soft_length, soft_size, soft_inertia,soft_eccentricity,soft_moment

from utils import simplex, sset, probs2one_hot
import torch.nn.modules.padding
from torch.nn import BCEWithLogitsLoss

class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        #self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        #print(f"> Initialized {self.__class__.__name__} with kwargs:")
        #pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape
        predicted_mask = probs2one_hot(probs).detach()
        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        #size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        #bool_size = (size_pred > 10).type(torch.float32)
        #print("before",value)
        #print(value.shape,bool_size.shape)
        #value = torch.einsum("bco,bco->bco", [value, bool_size])
        #print("after", value)
        lower_b = bounds[:, self.idc, :, 0]
        #print("lower_b",lower_b)
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape
        #print(" estimation from probs: ",value,"gt: ", lower_b, "gt size", soft_size(target[:, self.idc, ...]))
        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        #assert len(upper_z) == len(lower_b) == len(filenames)


        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class NaivePenalty(AbstractConstraints):
    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        return F.relu(z_)**2
    
class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        #self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum(f"bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum(f"bkwh->bk", pc) + einsum(f"bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

class AdaEntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        #log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        #log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        loss_cons_prior = - torch.einsum("bc,bc->", [gt_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b

        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop


class EntKLPropNoTag():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0] 
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        #print(est_prop_mask)
        #bool_est_prop_ispos = (est_prop_mask>0).float()
        bool_est_prop_select = (est_prop_mask>0.02).float()
        bool_fg_prop = torch.einsum("bc->b",est_prop_mask[:,1:])
        bool_fg_prop = (bool_fg_prop>0.02).float()
        bool_est_prop_select[:,0] = bool_fg_prop
        #print(bool_est_prop_ispos)
        # if the network estimates that the structure is absent, put the gt_estimation to zero.
        #gt_prop_pos = torch.einsum("bc,bc->bc", [gt_prop,bool_est_prop_ispos])
        est_prop_select = torch.einsum("bc,bc->bc", [est_prop,bool_est_prop_select])
        #gt_fg_prop = torch.einsum("bc->b",gt_prop_pos[:,1:])
        # correct the background size to get correct sum
        #gt_prop_pos[:,0] = 1 -gt_fg_prop
        #print(gt_prop,bool_est_prop_ispos,gt_prop_pos)
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        #log_gt_prop: Tensor = (gt_prop_pos + 1e-10).log()
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        #print(est_prop,est_prop_select)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop_select, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop_select, log_est_prop])
        # Adding division by batch_size to normalise 
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        #print(bool_est_prop)
        #mask_weighted_pos = torch.einsum("bcwh,bc->bcwh", [mask,bool_est_prop])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop 


class EntKLPropSelect():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        #print(est_prop_mask)
        #bool_est_prop_ispos = (est_prop_mask>0).float()
        bool_gt_prop_select = torch.einsum("bc->b",gt_prop)
        bool_gt_prop_select = (bool_gt_prop_select>0).float()

        est_prop_select = torch.einsum("bc,b->bc", [est_prop,bool_gt_prop_select])
        #log_est_prop: Tensor = (est_prop + 1e-10).log()
        log_est_prop: Tensor = (est_prop_select + 1e-10).log()
        #log_gt_prop: Tensor = (gt_prop_pos + 1e-10).log()
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        #print(est_prop,est_prop_select)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop_select, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop_select, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        #print(bool_est_prop)
        #mask_weighted_pos = torch.einsum("bcwh,bc->bcwh", [mask,bool_est_prop])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop


class FocalEntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop


class EntKLPropWMoment2():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.__momfn__ = getattr(__import__('utils'), kwargs['moment_fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.mom_est: List[float] = kwargs["mom_est"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_moment: float = kwargs["lamb_moment"]
        self.margin: float = kwargs["margin"]
        self.temp: float = kwargs["temp"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, c, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power)
        if not self.curi and len(gt_prop.shape) == 3:
            gt_prop = gt_prop.squeeze(2)
        if len(est_prop.shape) == 3:
            est_prop = est_prop.squeeze(2)

        log_est_prop: Tensor = (est_prop + 1e-10).log()
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        #log_div_prop = (torch.div(est_prop,gt_prop)+1e-10).log()
        log_div_prop = (torch.div(est_prop,gt_prop)).log()
        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        size_gt = soft_size(target[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 10).type(torch.float32)
        bool_gt_size = (size_gt > 1).type(torch.float32)
        #loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        loss_cons_prior = torch.einsum("bc,bc->", [est_prop, log_div_prop])
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        loss_cons_prior3 = kl_loss(est_prop, gt_prop)
        #print(loss_cons_prior,loss_cons_prior2)
        #if loss_cons_prior<0:
        #    print(loss_cons_prior,loss_cons_prior2, loss_cons_prior3)
            #print("est_prop:",est_prop,"log_gt_prop",log_gt_prop,"log_est_prop",log_est_prop)
        loss_cons_prior /= est_prop.shape[0]
        loss_cons_prior *= self.temp

        if self.__momfn__.__name__=="class_dist_centroid":
            loss_moment = self.__momfn__(probs.type(torch.float32))
            loss_moment = einsum("bou->", loss_moment)
        else:
            probs_moment = self.__momfn__(probs[:, self.idc, ...].type(torch.float32))
            if probs_moment.shape[2] != 3:
                # for numerical stability, only keep probs_moment if there is a predicted structure
                probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_size])
                # add the tag
                probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_gt_size])
            if probs_moment.shape[2] == 3:  # centroid 3D and dist2centroid 3d
                if self.curi:
                    est_gt_moment = torch.FloatTensor(self.mom_est).unsqueeze(0).to(loss_cons_prior.device)
                else:
                    est_gt_moment = self.__momfn__(target[:, self.idc, ...].type(torch.float32))
            else:
                if probs_moment.shape[2] == 2: # centroid and dist2centroid
                    binary_est_gt_moment_w = torch.FloatTensor(self.mom_est[0]).expand(b, c).unsqueeze(2)
                    binary_est_gt_moment_h = torch.FloatTensor(self.mom_est[1]).expand(b, c).unsqueeze(2)
                    binary_est_gt_moment = torch.cat((binary_est_gt_moment_w, binary_est_gt_moment_h), 2)
                    binary_est_gt_moment = binary_est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
                    est_gt_moment = binary_est_gt_moment
                else:
                    est_gt_moment = torch.FloatTensor(self.mom_est).unsqueeze(0).unsqueeze(2)
                    est_gt_moment = est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
                est_gt_moment = torch.einsum("bct,bco->bct", [est_gt_moment, bool_gt_size])
            #print(probs_moment.shape, est_gt_moment.shape)
            upper_z = (est_gt_moment*(1+self.margin) - probs_moment).flatten()
            lower_z = (probs_moment- est_gt_moment*(1-self.margin)).flatten()
            upper_z = F.relu(upper_z) ** 2
            lower_z = F.relu(lower_z) ** 2
            loss_moment = upper_z + lower_z
            loss_moment = einsum("f->", loss_moment)
        if probs_moment.shape[2] != 3:
            # Adding division by batch_size to normalise
            loss_moment /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,self.lamb_moment*loss_moment,est_prop


class EntKLPropWMoment():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.__momfn__ = getattr(__import__('utils'), kwargs['moment_fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.mom_est: List[float] = kwargs["mom_est"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_moment: float = kwargs["lamb_moment"]
        self.rel_diff: bool = kwargs["rel_diff"]
        self.margin: float = kwargs["margin"]
        self.temp: float = kwargs["temp"]
        self.linreg: bool = kwargs["linreg"]
        if not self.linreg:
            self.matrix = False
        else:
            self.matrix : bool = kwargs["matrix"]
        if self.linreg and not self.matrix:
            self.reg = eval(open(kwargs["reg"],'r').read())
            self.reg2 = eval(open(kwargs["reg2"],'r').read())
        if self.linreg and self.matrix:
            self.reg = torch.from_numpy(np.load(kwargs["reg"],allow_pickle=True))
            self.reg2 = torch.from_numpy(np.load(kwargs["reg2"],allow_pickle=True))

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, c, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power)
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        size_gt = soft_size(target[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 10).type(torch.float32)
        #bool_size = (size_pred > 2000).type(torch.float32)
        bool_gt_size = (size_gt > 1).type(torch.float32)

        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        loss_cons_prior /= b
        loss_cons_prior *= self.temp

        if self.__momfn__.__name__=="class_dist_centroid":
            loss_moment = self.__momfn__(probs.type(torch.float32))
            loss_moment = einsum("bou->", loss_moment)
        else:
            probs_moment = self.__momfn__(probs[:, self.idc, ...].type(torch.float32))
            # for numerical stability, only keep probs_moment if there is a predicted structure
            #print(probs_moment.shape,bool_size.shape)
            probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_size])
            # add the tag
            probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_gt_size])

            # if true ground truth moment:
            #true_gt_moment = self.__momfn__(target[:, self.idc, ...].type(torch.float32))
            # if estimated ground truth moment:
            if probs_moment.shape[2] == 2: # centroid and dist2centroid
                if self.linreg:
                    # method linreg
                    if self.matrix:
                        size_pred_m = torch.cat((torch.ones(b,1).to(loss_cons_prior.device), size_pred.squeeze(2)), 1)
                        coefs = self.reg.to(loss_cons_prior.device).type(torch.float32)
                        coefs2 = self.reg2.to(loss_cons_prior.device).type(torch.float32)
                        #print(size_pred_m.shape,coefs.shape)
                        est_gt_moment_h = torch.matmul(size_pred_m,coefs).unsqueeze(2)
                        est_gt_moment_w = torch.matmul(size_pred_m,coefs2).unsqueeze(2)
                        #print(gt_moment_h.shape)
                    else:
                        #print(self.reg[0])
                        slopes_w = torch.FloatTensor(self.reg[0]).unsqueeze(0).unsqueeze(2).to(loss_cons_prior.device)[:, self.idc, ...]
                        slopes_h = torch.FloatTensor(self.reg2[0]).unsqueeze(0).unsqueeze(2).to(loss_cons_prior.device)[:, self.idc, ...]
                        ints_w = torch.FloatTensor(self.reg[1]).unsqueeze(0).unsqueeze(2).to(loss_cons_prior.device)[:, self.idc, ...]
                        ints_h = torch.FloatTensor(self.reg2[1]).unsqueeze(0).unsqueeze(2).to(loss_cons_prior.device)[:, self.idc, ...]
                        #print(ints_w,slopes_w)
                        est_gt_moment_w = torch.einsum("bco,oco->bco", [size_pred, slopes_w]) + ints_w
                        est_gt_moment_h = torch.einsum("bco,oco->bco", [size_pred, slopes_h]) + ints_h
                    est_gt_moment = torch.cat((est_gt_moment_w, est_gt_moment_h), 2)
                else:
                    # method binary
                    #binary_est_gt_moment_w = torch.FloatTensor(self.mom_est[0]).unsqueeze(0).unsqueeze(2)
                    binary_est_gt_moment_w = torch.FloatTensor(self.mom_est[0]).expand(b, c).unsqueeze(2)
                    binary_est_gt_moment_h = torch.FloatTensor(self.mom_est[1]).expand(b, c).unsqueeze(2)
                    binary_est_gt_moment = torch.cat((binary_est_gt_moment_w, binary_est_gt_moment_h), 2)
                    binary_est_gt_moment = binary_est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
                    est_gt_moment = binary_est_gt_moment

            else:
                est_gt_moment = torch.FloatTensor(self.mom_est).unsqueeze(0).unsqueeze(2)
                est_gt_moment = est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
            # add tag to estimated gt moment : if there is no structure, put the moment at zero
            #print(est_gt_moment)
            #print(est_gt_moment.shape,bool_gt_size.shape)
            est_gt_moment = torch.einsum("bct,bco->bct", [est_gt_moment, bool_gt_size])
            #est_gt_moment = torch.einsum("bct,bco->bct", [est_gt_moment, bool_size])
            #binary_gt_moment = torch.einsum("bco,bco->bco", [binary_gt_moment, bool_gt_size])
            #print(old_gt_moment.shape, gt_moment.shape, true_gt_moment.shape)
            #print(old_gt_moment[1:2, ...], gt_moment[1:2, ...], true_gt_moment[1:2, ...])
            if self.rel_diff:
                upper_z = ((1+self.margin) - torch.div(probs_moment,(est_gt_moment+1e-6))).flatten()
                lower_z = (torch.div(probs_moment,(est_gt_moment+1e-6))- (1-self.margin)).flatten()
            else:
                upper_z = (est_gt_moment*(1+self.margin) - probs_moment).flatten()
                lower_z = (probs_moment- est_gt_moment*(1-self.margin)).flatten()
            upper_z = F.relu(upper_z) ** 2
            lower_z = F.relu(lower_z) ** 2
            # z_ = z.flatten()
            loss_moment = upper_z + lower_z
            loss_moment = einsum("f->", loss_moment)

        # Adding division by batch_size to normalise
        loss_moment /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se+ self.lamb_consprior*loss_cons_prior, self.lamb_moment*loss_moment,est_prop


class EntKLPropWMomentNu():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_moment: float = kwargs["lamb_moment"]
        self.ind_moment: list = kwargs["ind_moment"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        size_gt = soft_size(target[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 0).type(torch.float32)
        bool_gtsize = (size_gt > 0).type(torch.float32)
        #print("before",value)
        #print(value.shape,bool_size.shape)

        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        loss_cons_prior /= b
        gt_moment = soft_moment(target[:, self.idc, ...].type(torch.float32),self.ind_moment)
        #print("gt_moment",gt_moment,"soft_size",soft_size(target[:, self.idc, ...]))
        probs_moment = soft_moment(probs[:, self.idc, ...].type(torch.float32),self.ind_moment)
        probs_moment = torch.einsum("bco,bco->bco", [probs_moment, bool_size])
        probs_moment = torch.einsum("bco,bco->bco", [probs_moment, bool_gtsize])
        z = gt_moment-probs_moment
        z_ = z.flatten()
        loss_moment = z_**2
        loss_moment = einsum("f->", loss_moment)
        loss_moment /= b

        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se+ self.lamb_consprior*loss_cons_prior, self.lamb_moment*loss_moment,est_prop


class EntKLPropWInertia():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_inertia: float = kwargs["lamb_inertia"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 0).type(torch.float32)
        #print("before",value)
        #print(value.shape,bool_size.shape)

        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        loss_cons_prior /= b
        gt_inertia = soft_inertia_moment(target[:, self.idc, ...].type(torch.float32))
        probs_inertia = soft_inertia_moment(probs[:, self.idc, ...].type(torch.float32))
        probs_inertia = torch.einsum("bco,bco->bco", [probs_inertia, bool_size])
        z = gt_inertia-probs_inertia
        z_ = z.flatten()
        loss_inertia = F.relu(z_)**2
        loss_inertia = einsum("f->", loss_inertia)
        # Adding division by batch_size to normalise
        loss_inertia /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se+self.lamb_consprior*loss_cons_prior, self.lamb_inertia*loss_inertia,est_prop


class EntKLPropWEcc():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_ecc: float = kwargs["lamb_ecc"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 0).type(torch.float32)
        #print("before",value)
        #print(value.shape,bool_size.shape)

        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        loss_cons_prior /= b
        gt_ecc = soft_eccentricity(target[:, self.idc, ...].type(torch.float32))
        probs_ecc = soft_eccentricity(probs[:, self.idc, ...].type(torch.float32))
        #print(gt_ecc,probs_ecc)
        probs_ecc = torch.einsum("bco,bco->bco", [probs_ecc, bool_size])
        z = gt_ecc-probs_ecc
        z_ = z.flatten()
        loss_ecc = F.relu(z_)**2
        loss_ecc = einsum("f->", loss_ecc)
        # Adding division by batch_size to normalise
        loss_ecc /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se+ self.lamb_consprior*loss_cons_prior, self.lamb_ecc*loss_ecc,est_prop


class EntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0] 
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        # Adding division by batch_size to normalise 
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10


        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop 


class AlphaEntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = 2
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)

        #print(est_prop,log_gt_prop,log_est_prop)
        probs_pow = torch.pow(probs + 1e-12, self.power)
        gt_prop_pow = torch.pow(gt_prop, self.power)

        loss_cons_prior =  1/(w*h)*torch.einsum("bcwh->", [probs_pow])  - torch.einsum("bc->", [gt_prop_pow])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        loss_cons_prior /= (self.power-1)
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10


        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop


class EntKLPropWComp():
    """
    Entropy minimization with KL proportion regularisation and Compactness
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_comp: float = kwargs["lamb_comp"]
        self.inv_consloss: float = True #kwargs["inv_consloss"]
    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        selected: Tensor = probs[:, self.idc, ...]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        if not self.inv_consloss:
            loss_cons_prior = - torch.einsum("bc,bc->", [gt_prop, log_est_prop])
        else:
            log_gt_prop: Tensor = (gt_prop + 1e-10).log()
            log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
            loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
            # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10
        loss_comp: Tensor = soft_compactness(selected) / (w * h)  # Normalize by the size of the img
        assert loss_comp.dtype == torch.float32

        loss_comp = loss_comp.mean()
        #loss_f = loss_se+loss_comp
        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation
        return self.lamb_se*loss_se, self.lamb_comp*loss_comp+self.lamb_consprior*loss_cons_prior,est_prop

class EntKLPropWCompLen():
    """
    Entropy minimization with KL proportion regularisation and Compactness
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_comp: float = kwargs["lamb_comp"]
        self.lamb_len: float = kwargs["lamb_len"]
        self.inv_consloss: float = True #kwargs["inv_consloss"]
    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        selected: Tensor = probs[:, self.idc, ...]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        if not self.inv_consloss:
            loss_cons_prior = - torch.einsum("bc,bc->", [gt_prop, log_est_prop])
        else:
            log_gt_prop: Tensor = (gt_prop + 1e-10).log()
            log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
            loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
            # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10
        loss_comp: Tensor = soft_compactness(selected) / (w * h)  # Normalize by the size of the img
        loss_len: Tensor = soft_length(selected) / np.sqrt(w * h)  # Normalize by the length of the img
        assert loss_comp.dtype == torch.float32

        loss_comp = loss_comp.mean()
        loss_len = loss_len.mean()
        #loss_f = loss_se+loss_comp
        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation
        return self.lamb_se*loss_se, self.lamb_len*loss_len+self.lamb_comp*loss_comp+self.lamb_consprior*loss_cons_prior,est_prop


class EntComp():
    """
    Entropy minimization with Compactness
    """
    def __init__(self, **kwargs):
        self.idc: bool = kwargs["idc_c"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_comp: float = kwargs["lamb_comp"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        selected: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        loss_comp: Tensor = soft_compactness(selected) / (w * h)  # Normalize by the size of the img
        assert loss_comp.dtype == torch.float32

        loss_comp = loss_comp.mean()
        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation
        return self.lamb_se*loss_se+self.lamb_comp*loss_comp


class EntCompLength():
    """
    Entropy minimization with Compactness
    """
    def __init__(self, **kwargs):
        self.idc: bool = kwargs["idc_c"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_comp: float = kwargs["lamb_comp"]
        self.lamb_len: float = kwargs["lamb_comp"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        selected: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        loss_comp: Tensor = soft_compactness(selected) / (w * h)  # Normalize by the size of the img
        loss_len: Tensor = soft_length(selected) / np.sqrt(w * h)  # Normalize by the length of the img
        assert loss_comp.dtype == torch.float32

        loss_comp = loss_comp.mean()
        loss_len = loss_len.mean()
        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation
        return self.lamb_se*loss_se+self.lamb_comp*loss_comp+self.lamb_len*loss_len


class ProposalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        predicted_mask = probs2one_hot(probs).detach()
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = predicted_mask[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss


class SelfEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss



class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss


class BCELoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.dtype = kwargs["dtype"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, d_out: Tensor, label: float):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(d_out,Tensor(d_out.data.size()).fill_(label).to(d_out.device))
        return loss


class BCEGDice():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.lamb: List[int] = kwargs["lamb"]
        self.weights: List[float] = kwargs["weights"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss_gde = divided.mean()

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask_weighted = torch.einsum("bcwh,c->bcwh", [tc, Tensor(self.weights).to(tc.device)])
        loss_ce = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_ce /= tc.sum() + 1e-10
        loss = loss_ce + self.lamb*loss_gde

        return loss



class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss
