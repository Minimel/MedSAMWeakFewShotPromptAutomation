"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import os
from typing import Callable, Dict, List, Tuple, Union
import h5py
import numpy as np
from flatten_dict import flatten
from monai.transforms import (Compose, LoadImaged)

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms
from torchvision.transforms import InterpolationMode

from Utils.utils import get_nested_value
from Utils.dataset_utils import get_bounding_box, create_bbox_mask
from Regularization.BoxPrior import BoxPriorBounds, box_prior_zoo


class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, **kwargs):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.data_shape = kwargs.get('data_shape')
        self.model_image_size = kwargs.get('model_image_size', 1024)  # Default is 1024 for SAM
        self.prompt_type = get_nested_value(kwargs, 'prompt', 'prompt_type', default={}) 
        self.prompt_args = get_nested_value(kwargs, 'prompt', 'args', default=1)
        self.class_to_segment = kwargs.get('class_to_segment', 1)
        self.box_prior_args = kwargs.get('box_priors_args')
        self.use_precomputed_sam_embeddings = kwargs.get('use_precomputed_sam_embeddings', False)
        self.sam_checkpoint: str = kwargs.get('sam_checkpoint') # To load precomputed sam embeddings
        self.bounds_args_list: List[Callable] = kwargs.get('bounds_args_list', [])
        self.sam_positional_encoding: str = kwargs.get('sam_positional_encoding')

        self.transforms = Compose([LoadImaged(keys=['img', 'label'])])
        
        if self.box_prior_args is not None:
            self.box_priors_gen = BoxPriorBounds(**self.box_prior_args)
        if self.bounds_args_list != []:
            bounds_class = self.bounds_args_list[0]['bounds_name']
            self.bounds_generators: List[Callable] = [box_prior_zoo[bounds_class](**{k[-1]: v for k, v in flatten(self.bounds_args_list[0]).items()})]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[str,
                                                         Tensor,
                                                         List[Tensor],
                                                         List[Tuple[slice, ...]],
                                                         List[Tuple[Tensor, Tensor]]]]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # We create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'img': image_path, 'label': mask_path})

        # We squeeze extra dimensions
        image = data_dict['img'].squeeze()
        ground_truth_mask = data_dict['label'].squeeze()
        
        # We only look at the class to segment
        ground_truth_mask = torch.isin(ground_truth_mask, torch.Tensor([self.class_to_segment])).int()

        # We add the bounding box prompt
        box_prompts, _box_prompts = None, None
        if 'box' in self.prompt_type:
            _box_prompts = get_bounding_box(ground_truth_mask, perturbation_bounds=[self.prompt_args['perturbation_bound'][0], self.prompt_args['perturbation_bound'][1]])
            box_prompts = [[_box_prompts]]
        
        # We determine the new image size if needed
        if self.model_image_size != image.shape[-1]:
            new_image = torchvision.transforms.Resize((self.model_image_size, self.model_image_size), interpolation=InterpolationMode.BICUBIC)(image[None, :, :].repeat(3, 1, 1))
        else:
            new_image = image[None, :, :].repeat(3, 1, 1)
        inputs = {
            'data':  (new_image / 255).type(torch.float32),
            'original_sizes': torch.tensor([image.shape[-2], image.shape[-1]]).type(torch.int64),
            'label': ground_truth_mask[None, :, :].long()
        } 

        # We adapt the prompt to the new image size
        if 'box' in self.prompt_type:
            # Calculate scale factors
            ScaleX = inputs['data'].shape[-1] / image.shape[-1]
            ScaleY = inputs['data'].shape[-2] /image.shape[-2]
            # Adapt the coordinates to the new image size
            [x_min, y_min, x_max, y_max] = box_prompts[0][0]
            x_min_new = int(x_min * ScaleX)
            y_min_new = int(y_min * ScaleY)
            x_max_new = int(x_max * ScaleX)
            y_max_new = int(y_max * ScaleY)
            inputs['input_boxes'] = torch.tensor([[x_min_new, y_min_new, x_max_new, y_max_new]]).type(torch.float64)  #torch.tensor(box_prompts[0]).type(torch.float64)

        # We also add the original prompts
        if 'box' in self.prompt_type:
            inputs['original_input_boxes'] = torch.tensor(_box_prompts).type(torch.float64)
            weak_labels = torch.tensor(np.stack([create_bbox_mask(*inputs['original_input_boxes'].to(int).tolist(), inputs['original_sizes'].tolist())]))
            inputs['weak_label'] = weak_labels
        
        filename = os.path.basename(self.image_paths[idx]).split('.')[0]
        inputs['filename'] = filename
        inputs['patient_name'] = '_'.join(filename.split('_')[:-1])
        inputs['slice_number'] = filename.split('_')[-1]
        inputs['idx'] = idx
        
        # We get the image embeddings from the saved folder
        base_folder = os.path.dirname(os.path.dirname(self.image_paths[idx]))
        context = os.path.basename(os.path.dirname(self.image_paths[idx]))
        if self.use_precomputed_sam_embeddings:
            embed_path = os.path.join(base_folder, 'image_embeddings', self.sam_checkpoint.split('/')[-1].replace('.', '-'), context, filename)
            with h5py.File(embed_path + '.h5', 'r') as f:
                inputs['image_embeddings'] = f[filename][()]  # The [()] syntax reads the entire dataset

        # We also get the positional embeddings from the saved folder
        if self.sam_positional_encoding == 'saved':
            # We load the saved positional encoding, for each image
            base_folder = os.path.dirname(os.path.dirname(self.image_paths[idx]))
            context = os.path.basename(os.path.dirname(self.image_paths[idx]))
            pos_embed_path = os.path.join(base_folder, 'image_positional_embeddings'+self.processor_str, self.sam_checkpoint.split('/')[-1].replace('.', '-'), context, filename)
            with h5py.File(pos_embed_path + '.h5', 'r') as f:
                inputs['image_positional_embeddings'] = f[filename][()]  # The [()] syntax reads the entire dataset
        elif self.sam_positional_encoding == 'fixed':
            # We take the unique folder in the 'fixed directory'. The file has the name of the first image in train_images
            pos_embed_folder = os.path.join(base_folder, 'image_positional_embeddings', self.sam_checkpoint.split('/')[-1].replace('.', '-'), 'fixed')
            assert len(os.listdir(pos_embed_folder)) == 1, 'There should be only one file in the fixed folder'
            fixed_filename = os.listdir(pos_embed_folder)[0].replace('.h5', '')
            pos_embed_path = os.path.join(pos_embed_folder, fixed_filename)
            with h5py.File(pos_embed_path+ '.h5', 'r') as f:
                inputs['fixed_image_positional_embeddings'] = f[fixed_filename][()]  # The [()] syntax reads the entire dataset
        
        # For the BB tightness prior, we separate the bounding box into individual lines and columns and save into separate tensors.
        # (Although there is surely a more optimal way of doing this) 
        if self.box_prior_args is not None:
            box_priors: List[Tuple[Tensor, Tensor]] = self.box_priors_gen(weak_labels)
            if box_priors[0][0].shape[0] != (box_priors[0][0].shape[1] // box_priors[0][1][0] + box_priors[0][0].shape[2] // box_priors[0][1][0]).item():
                print('Here')
            inputs['box_priors'] = box_priors   
        if self.bounds_args_list != []:
            if len(self.bounds_generators) > 0:
                bounds: List[Tensor] = self.bounds_generators[0](image[None, :, :], ground_truth_mask[None, :, :], weak_labels, filename)
                inputs['bounds'] = bounds
  
        return inputs
