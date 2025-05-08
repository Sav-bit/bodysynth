"""_summary_

This is some utility functions for the training, debugging and go on

"""

import numpy as np
import torch
import nibabel as nib
import os


def save_representation(image: torch.Tensor, title: str, path: str = None, image_index : int = 0, affine_matrix = None) -> None:
    """
    Save a tensor representation to a file.
    Args:
        image (torch.Tensor): The tensor to save.
        title (str): The title of the image.
        path (str, optional): The path to save the image. Defaults to None.

    Note: The image is 5D tensor (batch_size, channels, depth, height, width).
    But sometimes is 4D (channels, depth, height, width) or 3D (depth, height, width).

    we need to handle all the cases.
    """

    # If it's a 5D tensor, we need to squeeze the batch size
    if image.ndim == 5:
        # If the batch size is 1, we can squeeze it
        # If the batch size is > 1, we need to select the user defined image
        image = image.squeeze(0) if image.shape[0] == 1 else image[image_index]

    # If it's a 4D tensor, we need check the first dimension (the channels):
    # If it's a 1, we need to squeeze it
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)

    # If it have multiple channels we need to argmax the channels
    if image.ndim == 4 and image.shape[0] > 1:
        image = torch.argmax(image, dim=0)

    header = nib.Nifti1Header()
    header.set_data_dtype(image.cpu().numpy().dtype)
    affine = affine_matrix if affine_matrix is not None else np.eye(4)
    toSave = nib.Nifti1Image(image.cpu().numpy(), affine=affine, header=header)

    filename = f"{title}.nii.gz"

    # If path is present, we check if the directory exists
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = os.path.join(path, filename)
    else:
        path = filename

    nib.save(toSave, path)
    print(f"Representation saved to {path}")
