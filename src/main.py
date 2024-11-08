"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
from comet_ml import Experiment
import sys
import os
from datetime import datetime
import json
import argparse
from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from segment_anything import sam_model_registry

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Data.datamodule import SAMDataModule
from Models.Sam_PromptLearner_WithModule import SamPromptLearner_WithModule
from Utils.load_utils import get_dict_from_config, NpEncoder
from Utils.utils import update_config_from_args

print("PyTorch version:", torch.__version__)
print("PyTorch Lightning version:", pl.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
    

def train_model(sam_config, module_config, train_config, data_config, data_dir, logger_config,
                checkpoint_path, gpu_devices=1, seed=42):

    seed_everything(seed, workers=True)

    if logger_config['name'] == 'comet':
        # We set comet_ml logger
        logger = CometLogger(
        api_key=logger_config['api_key'],
        workspace=logger_config['workspace'],
        project_name=logger_config['project_name'], 
        experiment_name=logger_config['experiment_name'],
        )
    else:
        logger = True  # Default logger (TensorBoard)
                
    # We create model
    kwargs = {'sam_positional_encoding': train_config['sam_positional_encoding']}
    full_model = SamPromptLearner_WithModule(per_device_batch_size = train_config["batch_size"],
                            lr = train_config["lr"],
                            weight_decay = train_config["weight_decay"],
                            sam_config = sam_config,
                            module_config = module_config,
                            sched_config = train_config["sched"],
                            loss_config=train_config["loss"],
                            seed = seed,
                            **kwargs)

    kwargs = {'prompt': train_config['prompt'],
              'data_shape': data_config['data_shape'],
              'class_to_segment': data_config['class_to_segment'],
              'box_priors_args': train_config["loss"].get('TightBoxPrior', {}).get('kwargs', None),
              'bounds_args_list': [{**_config["other_kwargs"], 'C': full_model.out_channels} for _, _config in train_config["loss"].items() if _config["other_kwargs"]["bounds_name"] is not None],
              'use_precomputed_sam_embeddings': data_config['use_precomputed_sam_embeddings'],
              'sam_positional_encoding': train_config['sam_positional_encoding'],
              'model_image_size': sam_config.get('image_size', None),
              'sam_checkpoint': sam_config.get('sam_checkpoint', None)
            }
    
    data_module = SAMDataModule(data_dir = data_dir,
                                dataset_name = data_config["dataset_name"],
                                batch_size = train_config["batch_size"],
                                val_batch_size = train_config["batch_size"],
                                num_workers = train_config["num_workers"],
                                train_indices=train_config["train_indices"],
                                dataset_kwargs=kwargs)
        
    # We can get rid of the prompt encoder and image encoder if embedding shave been saved
    del full_model.sam.prompt_encoder
    if data_config['use_precomputed_sam_embeddings']:
        del full_model.sam.image_encoder
    
    # Get Total number of parameters and total number of trainable parameters
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"\n Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}\n")

    # We set-up the trainer
    num_indices = 'all' if train_config["train_indices"]==[] else len(train_config["train_indices"])
    os.makedirs(os.path.join(checkpoint_path, '{}labeled'.format(num_indices)), exist_ok=True) # We create the checkpoint folder
    trainer = pl.Trainer(
            deterministic=True,
            max_epochs=train_config["num_epochs"],
            precision=16,
            devices=gpu_devices,
            accelerator='gpu',
            sync_batchnorm=True,
            log_every_n_steps=1,
            callbacks=[ModelCheckpoint(dirpath=os.path.join(checkpoint_path, '{}labeled'.format(num_indices)),
                                       filename='{epoch}', save_top_k=-1, every_n_epochs=20  # Save every 20 epochs
                    )],
            logger=logger,
            num_sanity_val_steps=0)

    # We train our model
    print("\n #### Training model ####")
    trainer.fit(full_model, data_module)

    # We evaluate our trained model on the test set
    print("\n #### Evaluating on test set ####")
    # We put back the original inage encoder
    if data_config['use_precomputed_sam_embeddings']:
        sam_args = argparse.Namespace(**sam_config)
        temp_sam = sam_model_registry[sam_config["model_name"]](sam_args.sam_checkpoint)
        full_model.sam.image_encoder = temp_sam.image_encoder
        full_model.sam.prompt_encoder = temp_sam.prompt_encoder
        del temp_sam
        print('Loaded original image encoder')
    data_module.setup(stage = 'test')    
    metric_dict = trainer.test(full_model, datamodule=data_module)
    
    # We save the metrics
    results_path = os.path.join(checkpoint_path, 'metrics.json')
    with open(results_path, 'w') as file:
        json.dump(metric_dict, file, indent=4, cls=NpEncoder)
        
    return trainer, full_model


if __name__ == "__main__":
    parser = ArgumentParser()   
    # These are the paths to the data and output folder
    parser.add_argument('--data_dir', default='/home/AR32500/net/data', type=str, help='Directory for data')
    parser.add_argument('--output_dir', default='output', type=str, help='Directory for output run')

    # These are config files located in src/Config
    parser.add_argument('--data_config',  type=str, 
                        default='data_config/data_config_ACDC_256.yaml'
                        #default='data_config/data_config_CAMUS_512.yaml'
                        #default='data_config/data_config_HC_640.yaml'
                        )
    parser.add_argument('--sam_config', type=str, default='model_config/medsam_config.yaml')
    parser.add_argument('--module_config', type=str, default='model_config/promptmodule_config.yaml')
    parser.add_argument('--train_config', type=str, 
                        default='train_config/train_config_100_0001.yaml'
                        )
    parser.add_argument('--logger_config', type=str, default='logger_config.yaml')
    parser.add_argument('--loss_config', type=str, nargs='+',
                        help='type of loss to appply (ie: CE, entropy_minimization, bounding_box_prior)',
                        default=[
                            'loss_config/outerBCE_W1.yaml',
                            'loss_config/tightbox_W00001.yaml', 
                            'loss_config/boxsize_w001.yaml'
                            ])
    parser.add_argument('--prompt_config', type=str, default='prompt_config/box_tight.yaml',)

    parser.add_argument('--num_gpu', default=1, help='number of GPU devices to use')
    parser.add_argument('--gpu_idx', default=[0], type=int, nargs='+', help='otherwise, gpu index, if we want to use a specific gpu')

    parser.add_argument('--logger__project_name', type=str, help='name of project in comet',
                        default='')

    # Training hyper-parameters that we should change according to the dataset and experiment
    parser.add_argument('--data__class_to_segment', type=int, nargs='+', help='class values to segment',
                        default=(1))
    parser.add_argument('--train__train_indices', type=int, nargs='+', help='indices of training data for Segmentation task',
                        default=[]
                        )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logger__experiment_name', type=str, help='name of experiment for checkpoint',
                        default='')
    
    # Hyperparameters fixed for our final experiments, but which can be changed.
    parser.add_argument('--data__use_precomputed_sam_embeddings', help='whether to use precomputed embeddings',
                        action="store_true", default=True)
    parser.add_argument('--train__sam_positional_encoding', type=str, help='whether to use positional encoding layer',
                        default='fixed')
    args = parser.parse_args()
    
    # We set the gpu devices (either a specific gpu or a given number of available gpus)
    if args.gpu_idx is not None:
        gpu_devices = args.gpu_idx
    else:
        gpu_devices = args.num_gpu
    print('gpu_devices {}'.format(gpu_devices))

    # We extract the configs from the file names
    train_config = get_dict_from_config(args.train_config)
    data_config = get_dict_from_config(args.data_config)
    sam_config = get_dict_from_config(args.sam_config)
    module_config = get_dict_from_config(args.module_config)
    logger_config = get_dict_from_config(args.logger_config)
    prompt_config = get_dict_from_config(args.prompt_config)
        
    train_config["loss"] = {}
    train_config = {**train_config, **{'prompt': prompt_config}}
    
    # We add the loss configs to the train config. If two losses have the same type, we will add a subscript
    for _file_config in args.loss_config:
        cur_config = get_dict_from_config(_file_config)
        loss_name = cur_config["type"]
        train_config["loss"][loss_name] = cur_config
    
    # We update the model and logger config files with the command-line arguments
    data_config = update_config_from_args(data_config, args, 'data')
    logger_config = update_config_from_args(logger_config, args, 'logger')
    train_config = update_config_from_args(train_config, args, 'train')
    print('train_config {}'.format(train_config))

    # We create a checkpoint path
    start_time = datetime.today()
    log_id = '{}_{}h{}min'.format(start_time.date(), start_time.hour, start_time.minute)
    checkpoint_path = os.path.join(args.output_dir, logger_config['experiment_name'], log_id, 'seed{}'.format(args.seed))
    print('checkpoint_path: {}'.format(checkpoint_path))
    
    trainer, full_model = train_model(sam_config, module_config, train_config, data_config, args.data_dir, logger_config,
                    checkpoint_path, gpu_devices, args.seed)
