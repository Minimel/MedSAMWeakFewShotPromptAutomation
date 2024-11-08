"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import os
import glob

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from monai.data import DataLoader

from transformers import SamProcessor

from Data.dataset import SAMDataset


class SAMDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str,
                 dataset_name: str,
                 batch_size: int = 4,
                 val_batch_size: int = 0,
                 num_workers: int = 2,
                 train_indices: list = [], # Indices of sample to use for training. If empty list, use all indices in train set
                 dataset_kwargs: dict = {},   # kwargs for SAM dataset (ie: {'prompt_args': {'prompt_type': 1, 'prompt_num': 1}, 'model_input_shape': 256, 'sam_preprocessor': 'facebook/sam-vit-base'})
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size == 0 else val_batch_size
        self.num_workers = num_workers
        self.train_indices = train_indices
        self.dataset_kwargs = dataset_kwargs
        
        print('dataset_kwargs: {}'.format(self.dataset_kwargs))
        
        # Following preprocessing from Data/sam_preprocessing.py
        datasets = ['train', 'val', 'test']
        data_types = ['2d_images', '2d_masks']
        
        # Initialize dictionary for storing image and label paths
        self.data_paths = {}
        # Create directories and print the number of images and masks in each
        for dataset in datasets:
            for data_type in data_types:
                # Construct the directory path
                dir_path = os.path.join(data_dir, dataset_name, 
                                        f'{dataset}_{data_type}')

                # Find images and labels in the directory
                nii_files = glob.glob(os.path.join(dir_path, "*.nii.gz"))
                png_files = glob.glob(os.path.join(dir_path, "*.png"))

                # Combine the lists
                files = sorted(nii_files + png_files)
                
                # Store the image and label paths in the dictionary
                self.data_paths[f'{dataset}_{data_type.split("_")[1]}'] = files

        if len(train_indices) > 0:
            self.data_paths['train_images'] = [self.data_paths['train_images'][i] for i in range(len(self.data_paths['train_images'])) if i in self.train_indices]
            self.data_paths['train_masks'] = [self.data_paths['train_masks'][i] for i in range(len(self.data_paths['train_masks'])) if i in self.train_indices]
        print('Number of training images', len(self.data_paths['train_images']))
        print('Number of validation images', len(self.data_paths['val_images']))
        print('Number of test images', len(self.data_paths['test_images']))
        
        # create an instance of the processor for image preprocessing
        if 'sam_processor' in self.dataset_kwargs:
            self.processor = SamProcessor.from_pretrained(self.dataset_kwargs['sam_processor'])
        else:
            self.processor = None

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None: 
            # We create 2 datasets, one with augmentations (ds_train) and one without (ds_val)
            self.ds_train = SAMDataset(image_paths=self.data_paths['train_images'], mask_paths=self.data_paths['train_masks'], 
                                       processor=self.processor, **self.dataset_kwargs)
            self.ds_val = SAMDataset(image_paths=self.data_paths['val_images'], mask_paths=self.data_paths['val_masks'], 
                                     processor=self.processor, **self.dataset_kwargs)
         
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            for key in ['box_priors_args', 'bounds_args_list', 'use_precomputed_sam_embeddings']:
                self.dataset_kwargs.pop(key, None)
            self.ds_test = SAMDataset(image_paths=self.data_paths['test_images'], mask_paths=self.data_paths['test_masks'], 
                                      processor=self.processor, **self.dataset_kwargs)

    def train_dataloader(self):
        self.train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        return self.train_loader
                
    def val_dataloader(self):
        self.val_loader = DataLoader(self.ds_val, batch_size=self.val_batch_size, shuffle=False,
                                     num_workers=self.num_workers)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.ds_test, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers)
        return self.test_loader
