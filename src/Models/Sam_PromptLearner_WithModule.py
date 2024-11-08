"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import sys
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Resize, InterpolationMode
import pytorch_lightning as pl

sys.path.append(".") 
from segment_anything import sam_model_registry

from Models.Base import BaseModel
from Models.promptlearning_modules import promptmodule_zoo


class SamPromptLearner_WithModule(BaseModel):
    def __init__(self, 
                 per_device_batch_size: int = 1,
                 num_devices: int = 1,
                 lr: float = 1e-6, #1e-3,
                 weight_decay: float = 5e-4, #1e-4,
                 sam_config: dict = {},
                 module_config: dict = {},
                 sched_config: dict = {},
                 loss_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed = 42,
                 **kwargs
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__(per_device_batch_size, num_devices, 
                         lr, weight_decay, sam_config, module_config, sched_config,
                         loss_config, val_plot_slice_interval, seed)
        
        model_args = argparse.Namespace(**sam_config)        
        self.sam = sam_model_registry[sam_config["model_name"]](model_args.sam_checkpoint)
        
        # SAM only works for 2 classes, so activation will be sigmoid
        self.activation_fct = nn.Sigmoid()
        
        # We freeze SAM and remove the prompt encoder
        for param in self.sam.parameters():
            param.requires_grad = False

        self.sam_positional_encoding = kwargs.get('sam_positional_encoding')

        # We create a prompt embedding module
        # Input: [BS, C=256, H=64, W=64]
        # Output: [BS, #pts=1, 2, 256] and [BS, 256, 64, 64] (sparse and dense embeddings)
        module_name = module_config['type']
        self.prompt_embedding_module = promptmodule_zoo[module_name](**module_config['args'])


    def forward(self, batched_input):    
        try:
            image_embeddings = batched_input['image_embeddings']
        except KeyError:
            with torch.no_grad():        
                image_embeddings = self.sam.image_encoder(batched_input['data'])
            
        # We get the prompt embeddings
        sparse_embeddings, dense_embeddings, mask_prompt = self.prompt_embedding_module(image_embeddings)

        if self.sam_positional_encoding == 'saved':
            image_positional_embeddings = batched_input['image_positional_embeddings']
        elif self.sam_positional_encoding == 'fixed': # or 'image_positional_embeddings' not in batched_input.keys():
            image_positional_embeddings = batched_input['fixed_image_positional_embeddings']

        low_res_pred_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings = image_embeddings, # (B, 256, 64, 64)
            image_pe = image_positional_embeddings[0:1],  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings.squeeze(1), # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False)

        pred_masks = F.interpolate(low_res_pred_masks,(batched_input['label'].shape[-2], batched_input['label'].shape[-1]), mode="bilinear", align_corners=False)
 
        return pred_masks

    def _training_step(self, batch, batch_idx):
        """
        Batch should have keys:
            'data': (BS, 1, 3, H_target, W_target)
            'seg_mask': (BS, n_channels, 1, H_target, W_target)
            'boxes': (BS, n_channels, H_target, W_target)
            'point_coords': (BS, n_channels, 1, 2)
            'point_labels': (BS, n_channels, 1)
        """      
        torch.use_deterministic_algorithms(True, warn_only=True)  # Because cumsum_cuda_kernel does not have a deterministic implementation...
        
        x, y, img_idx = batch['data'], batch['label'], batch['idx'] 
        logits = self.forward(batch)
            
        out_probs = self.activation_fct(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]
        losses = []
        for name, func, w in zip(self.loss_names['probs'], self.loss_fct['probs'], self.loss_weights['probs']):
            _loss, kwargs = func(probs, batch, **self.loss_kwargs)
            losses.append(w * _loss)  
        loss = sum(losses)
            
        return loss
    
    def _validation_step(self, batch, batch_idx):

        x, y, img_idx = batch['data'], batch['label'], batch['idx'] 
        logits = self.forward(batch)

        out_probs = self.activation_fct(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]
        losses = []
        for name, func, w in zip(self.loss_names['probs'], self.loss_fct['probs'], self.loss_weights['probs']):
            _loss, kwargs = func(probs, batch, **self.loss_kwargs)
            losses.append(w * _loss)
        loss = sum(losses)

        return loss
    
    def _test_step(self, batch, batch_idx):
        torch.use_deterministic_algorithms(True)
        x, y, img_idx = batch['data'], batch['label'], batch['idx'].item()
        logits = self.forward(batch)
        
        out_probs = self.activation_fct(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]
        pred_masks = (out_probs > 0.5).float()
        metrics = self._compute_seg_metrics(pred_masks, y)

        for cur_metric in metrics.keys():
            self.log('test/{}'.format(cur_metric), metrics[cur_metric])

    def configure_optimizers(self):
        params = list(self.prompt_embedding_module.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if 'MultiStepLR' in self.sched_config:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.sched_config["MultiStepLR"]["milestones"], 
                                                 gamma=self.sched_config["MultiStepLR"]["gamma"])
        elif 'CosineAnnealingLR' in self.sched_config:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.sched_config['CosineAnnealingLR']["max_epoch"])

        scheduler = {
            "scheduler": scheduler,
            "interval": self.sched_config["update_interval"],
            "frequency": self.sched_config["update_freq"],
        }

        return [optimizer], [scheduler]

