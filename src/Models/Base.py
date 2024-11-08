"""
Author: Mélanie Gaillochet
"""
import re
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from Utils.metric_utils import (dice_metric, iou_metric, hausdorff95_metric)
from Utils.utils import to_onehot
from Utils.load_utils import save_to_logger

class BaseModel(pl.LightningModule):
    def __init__(
            self, 
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
    ):
        super().__init__()
        torch.use_deterministic_algorithms(True)  # For reproducibility
        self.save_hyperparameters()
        self.per_device_batch_size = per_device_batch_size
        self.num_devices = num_devices
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.sched_config = sched_config
        self.loss_config = loss_config

        self.in_channels = sam_config["in_channels"]
        self.out_channels = sam_config["out_channels"]

        self.activation_fct = nn.Sigmoid() if self.out_channels == 2 else nn.Softmax(dim=1)
        
        # We define the model losses from the provided loss list
        self.all_loss_names = defaultdict(list)
        self.all_loss_fct = defaultdict(list)
        self.all_loss_weights = defaultdict(list)
        self.all_loss_start_epoch = defaultdict(list)
        for i, loss_name in enumerate(loss_config.keys()):
            print(f">> {i}th list of losses: {loss_name} - {loss_config[loss_name]}")
            loss_params = loss_config[loss_name]["kwargs"]
            pred_name = loss_params.get('pred_name', 'probs')
            fn = loss_config[loss_name]["other_kwargs"]["fn"]
            loss_name = re.sub(r'\d+$', '', loss_name) # We remove integers at the end of the string (added if several losses with the same type). Note that the losses should not apply to the same pred_name
            loss_class = getattr(__import__('losses'), loss_name) #
            self.all_loss_names[pred_name].append(loss_name)
            self.all_loss_fct[pred_name].append(loss_class(**loss_params, fn=fn))
            self.all_loss_weights[pred_name].append(loss_config[loss_name]["weight"])
            self.all_loss_start_epoch[pred_name].append(loss_config[loss_name]["start_epoch"])
                
        # Initialize empty lists for active losses and weights
        self.loss_names = defaultdict(list)
        self.loss_fct = defaultdict(list)
        self.loss_weights = defaultdict(list)
        self.loss_kwargs = {'out_channels': self.out_channels}
                 
    def forward(self, x):
        raise NotImplementedError

    def _training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def _validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def _test_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.train():
                return  self._training_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._training_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.validate():
                return  self._validation_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._validation_step(batch, batch_idx)
        
    def test_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.test():
                return  self._test_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._test_step(batch, batch_idx)

    def on_train_epoch_start(self):
        """
        PyTorch Lightning does not natively support learning rate warmup. 
        Therefore, you need to manually adjust the learning rate during the first few epochs (warmup phase).
        """
        # We adjust the learning rate
        cur_epoch = self.current_epoch
        if 'GradualWarmup' in self.sched_config:
            if cur_epoch < self.sched_config['GradualWarmup']["warmup_steps"]:
                lr_scale = min(1., float(cur_epoch + 1) / self.sched_config['GradualWarmup']["warmup_steps"])
                for pg in self.optimizers().param_groups:
                    pg['lr'] = lr_scale * self.learning_rate

        # We adjust the loss list
        self.loss_fct = defaultdict(list)
        self.loss_weights = defaultdict(list)
                
        # Update active losses and weights based on the current epoch
        for pred_name in self.all_loss_names.keys():
            for loss_name, loss_fn, start_epoch, weight in zip(self.all_loss_names[pred_name], self.all_loss_fct[pred_name], 
                                                    self.all_loss_start_epoch[pred_name], self.all_loss_weights[pred_name]):
                if cur_epoch >= start_epoch:
                    # We activate losses based on the current epoch
                    self.loss_names[pred_name].append(loss_name)
                    self.loss_fct[pred_name].append(loss_fn)
                    self.loss_weights[pred_name].append(weight)

    def on_train_epoch_end(self):
        # Update active losses and weights based on the current epoch
        for pred_name in self.all_loss_names.keys():
            for loss_name, loss_fn, in zip(self.all_loss_names[pred_name], self.all_loss_fct[pred_name]):
                # We update the parameters of the loss functions
                if hasattr(loss_fn, 'scheduler_step'):
                    if hasattr(loss_fn.penalty, 'update_frequency') and self.current_epoch % loss_fn.penalty.update_frequency == 0:
                        updated_param = loss_fn.scheduler_step()
                        save_to_logger(self.logger, 'metric', updated_param, loss_name + '_updated_param')

    def configure_optimizers(self):
        raise NotImplementedError
    
    def _compute_seg_metrics(self, pred, y):
        """
        Computing typical segmentation metrics: dice, iou and hausdorff95, both total and per class
        args:
            pred: (B, C, H, W) or (B, H, W) tensor
            y: (B, C, H, W) or (B, H, W) tensor
        """
        onehot_pred = to_onehot(pred.squeeze(1), self.out_channels)
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        _dice = dice_metric(onehot_pred, onehot_target)
        dice = torch.mean(_dice)
        per_class_dice = torch.mean(_dice, dim=0)

        _iou = iou_metric(onehot_pred, onehot_target)
        iou = torch.mean(_iou)
        per_class_iou = torch.mean(_iou, dim=0)

        hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target))
        per_class_hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target), dim=0)

        metrics = {'dice': 100*dice,
                   'iou': 100*iou,
                   'hausdorff95': hausdorff95,
                   'per_class_dice': 100*per_class_dice,
                   'per_class_iou': 100*per_class_iou,
                   'per_class_hausdorff95': per_class_hausdorff95}
        
        return metrics

    

