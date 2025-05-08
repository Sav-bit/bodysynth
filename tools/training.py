"""
This is the script for training the UNet3D model.
"""

import argparse
from itertools import islice
import torch
from tools.data_generator import DataGenerator
from unet3d.buildingblocks import DoubleConv
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """
    Returns the device to be used for training as a torch.device.
    MPS is disabled because it throws error for the convolution on my stupid pc 😡
    """
    # if torch.backends.mps.is_available():
    #     print("MPS is available")
    #     return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_data_generator(
    seg_path: str, batch_size: int, device: torch.device, num_workers: int 
) -> DataLoader:
    """
    Returns a data generator for the given segmentation path.
    the out_size is included only because my pc is not able to work with the full image.
    on the hpc it should not be set
    """

    data_gen = DataGenerator(
        seg_dir=seg_path,
        device=device,
        patch_size=[128, 128, 128],
        padding=22,
    )
    
    loader = DataLoader(
        data_gen,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def get_model(data_gen: DataGenerator) -> AbstractUNet:
    """
    Returns the UNet3D model.
    For readability, the network architecture is hardcoded here.
    """
    model = UNet3D(
        in_channels=1,
        out_channels=data_gen.get_num_classes(),
        f_maps=(32, 64, 128, 256),
        basic_module=DoubleConv, 
        layer_order='cgr',
        num_groups=8,
        final_sigmoid=False,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample='deconv',
        num_levels=5,
        dropout_prob=0.0,
        is_segmentation=True,
        is3d=True
    )

    return model


def get_loss():
    """
    Returns the loss criterion.
    For readability, the loss is hardcoded here.
    """
    # Define your loss configuration
    loss_config = {
        "loss": {
            "name": "DiceLoss",
            "normalization": "sigmoid",
            # additional parameters can go here if needed...
        }
    }

    # Create the loss criterion
    return get_loss_criterion(loss_config)


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

    # -----------------------------
    # End of the arguments
    # -----------------------------

    # Check the PyTorch version
    print("PyTorch version:", torch.__version__)

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Set static parameters
    num_epochs = 5
    batch_size = 1 #How many images to load at once
    num_batches_per_epoch = 1 # How many batches to load per epoch
    patch_size = [128, 128, 128]

    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=device,
        num_workers=2,
    )

    # Get the model
    model = get_model(data_gen=data_gen).to(device=device)

    # Get the loss criterion
    criterion = get_loss()

    # Get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for images, segs in islice(data_gen, num_batches_per_epoch):
            
            #Check the images and segs device
            print(f"Images device: {images.device}")
            print(f"Segs device: {segs.device}")

            optimizer.zero_grad()

            # Forward pass
            prediction = model(images)

            # Compute the loss
            loss = criterion(prediction, segs)
            loss.backward()

            optimizer.step()
            print(f"Loss: {loss.item()}")
