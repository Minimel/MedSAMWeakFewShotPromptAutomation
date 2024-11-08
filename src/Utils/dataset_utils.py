"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import numpy as np


def get_bounding_box(ground_truth_map, perturbation_bounds=[5, 20]):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(*perturbation_bounds))
        x_max = min(W, x_max + np.random.randint(*perturbation_bounds))
        y_min = max(0, y_min - np.random.randint(*perturbation_bounds))
        y_max = min(H, y_max + np.random.randint(*perturbation_bounds))
        
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256] # if there is no mask in the array, set bbox to image size


def create_bbox_mask(x_min, y_min, x_max, y_max, mask_size):
    """
    Creates a binary mask with a bounding box.

    Args:
    x_min, y_min, x_max, y_max (int): Coordinates of the bounding box.
    mask_size (tuple): Size of the output mask (height, width).

    Returns:
    numpy.ndarray: A binary mask with the bounding box.
    """
    # Create an empty mask with the given size
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Set the pixels within the bounding box to 1
    mask[y_min:y_max+1, x_min:x_max+1] = 1

    return mask
