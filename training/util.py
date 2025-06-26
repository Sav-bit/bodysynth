"""_summary_

This is some utility functions for the training, debugging and go on

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
import os


def save_representation(
    image: torch.Tensor,
    title: str,
    path: str = None,
    image_index: int = 0,
    affine_matrix=None,
) -> None:
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


def plot_loss(
    train_losses: dict[int, float],
    validation_losses: dict[int, float] | None = None,
    save_plot: bool = False,
    run_name: str = None,
) -> None:
    """
    Plot both training and (optionally) validation losses over time.
    The losses are dicts where each key is an epoch, and each value is the loss.

    Args:
        train_losses (dict[int, float]): Dictionary of training losses, {epoch: loss}.
        validation_losses (dict[int, float], optional): Dictionary of validation losses, {epoch: loss}.
        save_plot (bool): If True, save the plot to "checkpoints/loss.png"; otherwise show it.
    """


    # Sort epochs so they are plotted in the correct order
    train_epochs = sorted(train_losses.keys())
    train_vals = [train_losses[e] for e in train_epochs]

    plt.figure()
    plt.plot(train_epochs, train_vals, label="Training Loss")

    if validation_losses is not None and len(validation_losses) > 0:
        val_epochs = sorted(validation_losses.keys())
        val_vals = [validation_losses[e] for e in val_epochs]
        plt.plot(val_epochs, val_vals, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.grid(True)
    plt.legend()

    if save_plot:
        os.makedirs("checkpoints", exist_ok=True)
        plt.savefig(os.path.join("checkpoints", f"loss_{run_name}.png"))
    else:
        plt.show()

    plt.close()
    
def build_CE_weights(
    class_frequencies: dict[int, float],
    num_classes: int,
    mode = "median",
) -> list[float]:
    """
    Build class weights for CrossEntropyLoss based on class frequencies
    Args:
        class_frequencies (dict[int, float]): Dictionary with class frequencies.
        num_classes (int): Total number of classes.
        mode (str): The mode to use for calculating weights. Default is "median".
                    Other options are sqrt and inverse.
        
    Returns:
        list[float] of class weights.
    """
    # weights = np.zeros(num_classes, dtype=np.float32)
    weights = np.ones(num_classes, dtype=np.float32)
    for class_id, frequency in class_frequencies.items():
        if class_id < num_classes and frequency > 0:
            weights[class_id] = frequency
            
    if mode == "median":
        weights = np.median(weights) / weights # median / frequency
    elif mode == "sqrt":
        weights = 1.0 / np.sqrt(weights)
    else: # Inverse
        weights = 1.0 / weights
        
    # Normalize weights to sum to 1
    # weights = weights / np.sum(weights)

    return weights

