"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
from flatten_dict import flatten
from flatten_dict import unflatten
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch


def get_nested_value(dictionary, *keys, default=None):
    """
    We retrieve a value nested within a dictionary using the **kwargs approach
    """
    for key in keys:
        dictionary = dictionary.get(key, {})
    return dictionary if dictionary else default


def to_onehot(input, n_classes): 
    """
    We do a one hot encoding of each label in 3D.
    (ie: instead of having a dimension of size 1 with values 0-k,
    we have 3 axes, all with values 0 or 1)
    :param input: tensor
    :param n_classes:
    :return:
    """
    assert torch.is_tensor(input)

    # We get (bs, l, h, w, n_channels), where n_channels is now > 1
    one_hot = torch.nn.functional.one_hot(input.to(torch.int64), n_classes)

    # We permute axes to put # channels as 2nd dim
    if len(one_hot.shape) == 5:
        # (BS, H, W, L, n_channels) --> (BS, n_channels, H, W, L)
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
    elif len(one_hot.shape) == 4:
        # (BS, H, W, n_channels) --> (BS, n_channels, H, W)
        one_hot = one_hot.permute(0, 3, 1, 2)
    elif len(one_hot.shape) == 3:
        # (H, W, n_channels) --> (n_channels, H, W)
        one_hot = one_hot.permute(2, 1, 0)
    return one_hot
  

def update_config_from_args(config, args, prefix):
    """
    We update the given config with the values given by the args
    Args:
        config (list): config that we would like to update
        args (parser arguments): input arguments whose values starting with given prefix we would like to use
                                Must be in the form <prefix>/<config_var_name-separated-by-/-if-leveled>/ (ie: ssl/sched/step_size) 
                                ! Must also not contain 'config' in the name !
        prefix (str): all parser arguments starting with the prefix + '/' will be updated.

    Returns:
        config: updated_config
    """
    # We extract the names of variables to update
    var_to_update_list = [name for name in vars(args) if prefix +'__' in name]# and 'config' not in name)]
    
    updated_config = flatten(config)  # We convert dictionary to list of tuples (tuples incorporating level information)
    for name in var_to_update_list:
        new_val = getattr(args, name)
        if new_val is not None:   # if the values given is not null, we will update the dictionary
            variable = name.replace(prefix + '__', '', 1)  # We remove the prefix
            level_tuple = tuple(variable.split('__'))   # We create a tuple with all sublevels of config
            #if level_tuple in updated_config.keys():  # We update the config
            #    updated_config.pop(level_tuple, None) # We remove the old key/value pair to avoid errors if there are multiple levels
            updated_config[level_tuple] = new_val
    updated_config = unflatten(updated_config)  # We convert back to a dictionary
    
    return updated_config
