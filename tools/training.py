"""
This is the script for training the UNet3D model.
"""

import argparse
import torch
import nibabel as nib
import brainsynth
from tools.data_generator import DataGenerator
from unet3d.buildingblocks import DoubleConv
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
    seg_path: str, batch_size: int, device: torch.device
) -> DataGenerator:
    """
    Returns a data generator for the given segmentation path.
    the out_size is included only because my pc is not able to work with the full image.
    on the hpc it should not be set
    """

    data_gen = DataGenerator(
        seg_dir=seg_path,
        batch_size=batch_size,
        device=device,
        # out_size=[128, 128, 128],  # Set to None for full image
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
        f_maps=(32, 64, 128, 256, 320),
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
    num_epochs = 2
    batch_size = 2
    patch_size = [128, 128, 128]

    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=device,
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

        for image, seg in data_gen:
            # Move data to the device
            # Is this needed?
            image = image.to(device)
            seg = seg.to(device)

            image_patch, seg_patch = data_gen.get_random_patch(image, seg)

            optimizer.zero_grad()

            # Forward pass
            prediction = model(image_patch)

            # Compute the loss
            loss = criterion(prediction, seg_patch)
            loss.backward()

            optimizer.step()
            print(f"Loss: {loss.item()}")
