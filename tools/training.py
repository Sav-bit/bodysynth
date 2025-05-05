"""
This is the script for training the UNet3D model.
"""

import argparse
import torch
import nibabel as nib
import brainsynth
from tools.data_generator import DataGenerator
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D


def get_device() -> torch.device:
    """
    Returns the device to be used for training as a torch.device.
    MPS is disabled because it throws error for the convolution 😡
    """
    # if torch.backends.mps.is_available():
    #     print("MPS is available")
    #     return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_data_generator(
    seg_path: str, batch_size: int, device: torch.device, out_size=[128, 128, 128]
) -> brainsynth.Synthesizer:
    """
    Returns a data generator for the given segmentation path.
    the out_size is included only because my pc is not able to work with the full image.
    on the hpc it should not be set
    """

    data_gen = DataGenerator(
        seg_dir=seg_path,
        batch_size=batch_size,
        device=device,
        out_size=out_size,
    )

    return data_gen

def get_model(data_gen: DataGenerator) -> AbstractUNet:
    """
    Returns the UNet3D model.
    For readability, the network architecture is hardcoded here.
    """
    model = UNet3D(
        in_channels=1,
        out_channels=data_gen.get_num_classes(),
        f_maps=[32, 64, 128, 256],
        layer_order="gcr",
        num_groups=8,
        is_segmentation=True,
        conv_padding=1,
        is3d=True,
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet3D with a segmentation path"
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        required=True,
        help="Path to the segmentation file (e.g., Ernie segmentation)",
    )
    args = parser.parse_args()
    

    seg_path = args.seg_path
   
    #-----------------------------
    # End of the arguments
    #-----------------------------
    

    # Check the PyTorch version
    print("PyTorch version:", torch.__version__)

    # Get the device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set Epochs and Batch size
    num_epochs = 500
    batch_size = 1
    
    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=device,
    )
    
    # Get the model
    model = get_model(data_gen=data_gen).to(device=device)
    
    
