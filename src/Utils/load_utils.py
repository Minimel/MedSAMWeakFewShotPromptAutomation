"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import sys
import json
import yaml
import os
import numpy as np

import pytorch_lightning as pl

sys.path.append(".") 
from src.Configs.config import config_folder


def _read_yaml_file(file_path):
    """
    We are reading the yaml file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config yaml file provided
    with open(file_path, 'r') as file:
        output_dict = yaml.safe_load(file)
    return output_dict


def get_dict_from_config(config_filename):
    """
    Get the config file (yaml) as a dictionary
    :param config_filename: name of config file (located in config folder)
    :return: dictionary
    """
    config_filepath = os.path.join(config_folder, config_filename)
    if config_filepath.endswith('.yaml'):
        config_dict = _read_yaml_file(config_filepath)

    return config_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_to_logger(logger, type, data, name, epoch=None):
    """
    We save data to the given logger

    Args:
        logger (tensorboard logger, comet ml logger, etc.): logger of pytorch lightning trainer
        type (str): 'metric' (to save scalar), 'list'
        data (any): what we want to save
        name (str): name to use when saving data
        epoch (int): if we want to assign the data to a given epoch
    """
    if type == 'metric':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_metric(name, data)
        else:
            # Saving on TensorBoardLogger
            logger.experiment.add_scalar(name, data, epoch)
            
    elif type == 'list':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_other(name, data)
        else:
            # Saving on TensorBoardLogger as scalar, with epoch as indice in list
            for i in range(len(data)):
                logger.experiment.add_scalar(name, data[i], i)
                
    elif type == 'hyperparameter':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_parameter(name, data)
        else:
            # Saving on TensorBoardLogger
            logger.log_hyperparams({name: data})
