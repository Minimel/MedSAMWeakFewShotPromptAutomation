"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
from typing import List, cast, Dict, Union
import torch
import torch.nn as nn
from torch import Tensor

from Utils.utils import get_nested_value


class BinaryCrossEntropy_OuterBoundingBox(nn.Module):
    """
    Code modified from 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
    https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Self.idc is used to filter out some classes of the prediction mask
        self.target_str: str = kwargs["target_str"] # ie: 'weak_label', which refers to bounding box
        self.idc: List[int] = kwargs["idc"] #ie: 0 to compute BCE on region outside bounding box
        self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, batch: Dict[str, Tensor], eps=1e-10, **kwargs) -> Tensor:
        target = batch[self.target_str]
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        # We create a mask to only consider the target region (true mask, bounding box, etc.), with the idc classes as positive
        mask: Tensor = torch.zeros(target.shape)
        for i in self.idc:
            mask[target == i] = 1
        mask = cast(Tensor, mask).to(target.device)

        # We compute log_p on all values in mask (for self.idc=0, we know region in mask should be background)
        loss = -torch.sum(torch.mul(mask, log_p))
        loss /= mask.sum() + eps

        return loss, None


class BoxSizePrior(nn.Module):
    """
    Code modified from 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
    https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.target_str: str = kwargs["target_str"] # ie: 'weak_label', which refers to bounding box
        self.idc: List[int] = kwargs["idc"] #ie: 1 for foreground
        self.C: int = len(self.idc)

        # Selecting which penalty to apply (log_barrier or relu)
        penalty = kwargs.get('penalty_type', 'log_barrier')
        self.penalty = penalty_zoo[penalty](**kwargs)
        
    def __call__(self, probs: Tensor, batch: Dict[str, Tensor],  **kwargs) -> Tensor:
        bounds = batch['bounds']
        assert self.target_str == 'weak_label'
        box_sizes = torch.sum(batch[self.target_str], dim=(-1, -2)).squeeze(1)
        
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1

        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        # Adding threshold on probabilities (to only consider big enough probabilities)
        _probs = probs[:, self.idc, ...].type(torch.float64)
        value = torch.sum(_probs, dim=(-1, -2))[..., None]
        lower_b = bounds[:, [i - 1 for i in self.idc], :, 0]
        upper_b = bounds[:, [i - 1 for i in self.idc], :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float64)).flatten()
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float64)).flatten()

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)

        _loss: Tensor = upper_penalty + lower_penalty
        loss: Tensor = torch.mean(_loss / box_sizes) 

        return loss, None
        
    def scheduler_step(self):
        if hasattr(self.penalty, 'step'):
            return self.penalty.step()
        else:
            pass


class TightBoxPrior(nn.Module):
    """
    Code modified from 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
    https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.idc: List[int] = kwargs["idc"] #ie: 1 for foreground
        
        # Selecting which penalty to apply (log_barrier or relu)
        penalty = kwargs.get('penalty_type', 'log_barrier')
        self.penalty = penalty_zoo[penalty](**kwargs)

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def scheduler_step(self):
        if hasattr(self.penalty, 'step'):
            return self.penalty.step()
        else:
            pass

    def compute_shift(self, probs, bounds, masks) -> Tensor:
        # We compute the sum of probs in each band
        # Adding threshold on probabilities (to only consider big enough probabilities)
        _probs =  probs[None, :, :].repeat(masks.shape[0], 1, 1)  # (#bands, H, W)
        sizes: Tensor = torch.sum(torch.mul(masks, _probs), dim=(-2, -1))

        assert sizes.shape == bounds.shape == (masks.shape[0],), (sizes.shape, bounds.shape, masks.shape)
        
        # We compute the difference between the band width and sum of probs, for each band
        shifted: Tensor = bounds - sizes
        return shifted
    
    def remove_padding(self, bounds, masks):
        # Remove the padding 0s in bounds and masks (added for collating function in dataloader)
        if len((bounds == 0).nonzero(as_tuple=True)[0]) == 0:
            # There is no padding
            return bounds, masks
        else:
            max_index = min(((bounds == 0).nonzero(as_tuple=True)[0])) # index of first 0 in bounds
            bounds = bounds[:max_index]
            masks = masks[:max_index]
        return bounds, masks
        
    def __call__(self, probs: Tensor, batch: Tensor, **kwargs) -> Tensor:
        # We extract the values in batch["box_priors"] for each image of the batch
        box_prior = [[[(m, b)] for (m, b) in zip(M, B)] for M, B in batch["box_priors"]][0]
        
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1

        B: int = probs.shape[0]

        _loss = 0
        for b in range(B):
            for k in self.idc:
                masks, bounds = box_prior[b][k - 1] # k-1 because box_prior contains mask and bounds for foreground only
                _probs = probs[b, k]

                # Remove the padding 0s in bounds and masks (added for collating)
                bounds, masks = self.remove_padding(bounds, masks)

                shifted = self.compute_shift(_probs, bounds, masks)

                # We apply penalty to each 
                error = self.penalty(shifted)
                _loss += torch.sum(error) # sum across all segments

        loss: Tensor = _loss / B # Dividing by batch size

        return loss, None


class LogBarrierPenalty(nn.Module):
    def __init__(self, **kwargs): #t: float = 5, epoch_multiplier: Union[int, float] = 1.1):
        super().__init__()
        t: float = kwargs.get('t', 5.0)
        multiplier: Union[int, float] = get_nested_value(kwargs, 'scheduler', 'multiplier', default=1.1)
        update_frequency: Union[int, float] = get_nested_value(kwargs, 'scheduler', 'update_frequency', default=1)
        self.register_buffer('t', torch.as_tensor(t))
        self.register_buffer('multiplier', torch.as_tensor(multiplier))
        self.register_buffer('update_frequency', torch.as_tensor(update_frequency))
        self.register_buffer('ceil', -1 / self.t ** 2)
        self.register_buffer('b', -torch.log(1 / (self.t ** 2)) / self.t + 1 / self.t)
        
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(
            cast(torch.Tensor, z <= self.ceil),
            - torch.log(-z) / self.t,
            self.t * z + self.b,
        )
        
    def step(self):
        self.t *= self.multiplier
        self.ceil[...] = -1 / self.t ** 2
        self.b[...] = -torch.log(1 / (self.t**2)) / self.t + 1 / self.t
        return self.t


penalty_zoo = {
    'log_barrier': LogBarrierPenalty
               }