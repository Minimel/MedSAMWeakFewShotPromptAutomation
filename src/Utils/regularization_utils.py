"""
Code for bounding box tightness prior utils 
From official repository of 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
Modified from https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/utils.py
"""
from collections import namedtuple
from typing import Iterable, List, Set, Tuple, cast

import torch
import numpy as np
from torch import Tensor
from skimage import measure


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    """
    From https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/utils.py#L124
    """
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


BoxCoords = namedtuple("BoxCoords", ["x", "y", "w", "h"])


def binary2boxcoords(seg: Tensor) -> List[BoxCoords]:
    """
        Converts (0-1) bounding box mask to box prompt coordinates (x, y, box_height, box_width)
    """
    assert sset(seg, [0, 1])
    _, __ = seg.shape  # dirty way to ensure the 2d shape

    blobs: np.ndarray
    n_blob: int
    blobs, n_blob = measure.label(seg.cpu().numpy(), background=0, return_num=True)

    assert set(np.unique(blobs)) <= set(range(0, n_blob + 1)), np.unique(blobs)

    class_coords: List[BoxCoords] = []
    for b in range(1, n_blob + 1):
        blob_mask: np.ndarray = blobs == b

        assert blob_mask.dtype == np.bool_, blob_mask.dtype
        # assert set(np.unique(blob_mask)) == set([0, 1])

        coords = np.argwhere(blob_mask)

        x1, y1 = coords.min(axis=0)
        x2, y2 = coords.max(axis=0)

        class_coords.append(BoxCoords(x1, y1, x2 - x1, y2 - y1))

    assert len(class_coords) == n_blob

    return class_coords


def boxcoords2masks_bounds(boxes: List[BoxCoords], shape: Tuple[int, int], d: int) -> Tuple[Tensor, Tensor]:
    '''
        Divides the box mask into individual bands of width d.
        Returns 2 tensors: one of shape [#bands, H, W], containing the value 1 for every vertical and horizontal band, and another containing the associated width of every band
        #bands = height of bounding box / d + width of bounding_box / d
        For nested list, can just iterate over this function
    '''

    masks_list: List[Tensor] = []
    bounds_list: List[float] = []

    box: BoxCoords
    for box in boxes:
        for i in range(box.w // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + i * d:box.x + (i + 1) * d, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.w % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + box.w - (box.w % d):box.x + box.w + 1, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.w % d + 1)   # +1 because the width does not include the first pixel

        for j in range(box.h // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + j * d:box.y + (j + 1) * d] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.h % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + box.h - (box.h % d):box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.h % d + 1)   # +1 because the width does not include the first pixel

    bounds = torch.tensor(bounds_list, dtype=torch.float32) if bounds_list else torch.zeros((0,), dtype=torch.float32)
    masks = torch.stack(masks_list) if masks_list else torch.zeros((0, *shape), dtype=torch.float32)

    # We fill up the tensor to make it same size as the other tensors
    # Create a zero tensor of size [256 - # bands in mask, 512, 512]
    max_n_channels = masks.shape[1] // bounds_list[0] + masks.shape[2] // bounds_list[0]
    zero_padding = torch.zeros(max_n_channels - masks.shape[0], masks.shape[1], masks.shape[2])

    # Concatenate the original tensor with the zero tensor
    masks = torch.cat((masks, zero_padding), 0)
    bounds = torch.cat((bounds, torch.zeros(max_n_channels - bounds.shape[0])))

    assert masks.dtype == torch.float32

    return masks, bounds