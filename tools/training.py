"""
This is the script for training the UNet3D model.
"""

import argparse
from itertools import islice
import torch
from tools.data_generator import DataGenerator
from unet3d import utils
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
    seg_path: str,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    patch_size: list = [128, 128, 128],
) -> DataLoader:
    """
    Returns a data loader for the given segmentation path.

    Args:
        seg_path (str): Path to the segmentation file (e.g Ernie segmentation).
        batch_size (int): Batch size for the data loader.
        device (torch.device): Device to be used for training.
        num_workers (int): Number of workers for the data loader.
    """

    data_gen = DataGenerator(
        seg_dir=seg_path,
        device=device,
        patch_size=patch_size,
        padding=22,
    )

    loader = DataLoader(
        data_gen,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    return loader


def get_model(data_gen: DataLoader) -> AbstractUNet:
    """
    Returns the UNet3D model.
    For readability, the network architecture is hardcoded here.
    """
    model = UNet3D(
        in_channels=1,
        out_channels=data_gen.dataset.get_num_classes(),
        f_maps=(32, 64, 128, 256, 512),
        layer_order="cgr",
        num_groups=8,
        final_sigmoid=False,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="deconv",
        num_levels=5,
        dropout_prob=0.0,
        is_segmentation=True,
        is3d=True,
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
    num_epochs = 50  # How many epochs to train
    batch_size = 1  # How many images to load at once
    num_batches_per_epoch = 1  # How many batches to load per epoch
    patch_size = [128, 128, 128]

    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=device,
        num_workers=0,
        patch_size=patch_size,
    )

    # Get the model
    model = get_model(data_gen=data_gen).to(device=device)

    # Get the loss criterion
    criterion = get_loss()

    # Get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        # The data generator is infinite, so we need to limit the number of batches
        for images, segs in islice(data_gen, num_batches_per_epoch):

            optimizer.zero_grad()

            # Forward pass
            prediction = model(images)

            # Compute the loss
            loss = criterion(prediction, segs)
            loss.backward()

            optimizer.step()
            curr_loss = loss.item()
            losses.append(curr_loss)
            print(f"Batch loss: {curr_loss:.4f}")

        if epoch % 50 == 0:
            checkpoint_dir = "./checkpoints"
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                # any other info
            }
            is_best = False
            utils.save_checkpoint(state, False, checkpoint_dir)
