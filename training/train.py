"""
This is the script for training the UNet3D model.
"""

import argparse
from itertools import islice
import torch
from training.data_generator import DataGenerator
from unet3d import utils
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


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
        padding=10,
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
    dice_loss_config = {
        "loss": {
            "name": "DiceLoss",
            "normalization": "sigmoid",
        }
    }

    dice_loss = get_loss_criterion(dice_loss_config)

    cross_entropy_loss = get_loss_criterion(
        {
            "loss": {
                "name": "CrossEntropyLoss",
            }
        }
    )

    # Create the loss criterion
    return dice_loss + cross_entropy_loss


def save_checkpoint_state(model, optimizer, losses, epoch, is_final=False):
    checkpoint_dir = "./checkpoints"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses,
        "is_final": is_final,
    }
    is_best = False
    utils.save_checkpoint(state, is_best, checkpoint_dir)


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

    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint",
    )

    args = parser.parse_args()

    seg_path = args.seg_path
    continue_training = args.continue_training

    # -----------------------------
    # End of the arguments
    # -----------------------------

    # Check the PyTorch version
    print("PyTorch version:", torch.__version__)

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Set static parameters
    num_epochs = 250  # How many epochs to train
    batch_size = 1  # How many images to load at once
    num_batches_per_epoch = 1  # How many batches to load per epoch
    patch_size = [180, 180, 180]

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
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    losses = []
    last_epoch = 0

    if continue_training:
        retrieved_state = utils.load_checkpoint(
            "./checkpoints/last_checkpoint.pytorch",
            model,
            optimizer=optimizer,
        )

        last_epoch = retrieved_state["epoch"]

        if retrieved_state["is_final"]:
            print(
                "The last checkpoint is the final checkpoint. No need to continue training."
            )
            exit(0)

        print(f"Continuing training from epoch {last_epoch + 1}")
        losses = retrieved_state["loss"]

    scheduler = StepLR(optimizer, step_size=30, gamma=0.2, last_epoch=last_epoch - 1)

    # early-stopping callback for validation loss
    early_stop_patience = 60  # epochs
    best_val, epochs_no_improve = float("inf"), 0
    min_lr = 1e-6

    # Training loop
    for epoch in range(last_epoch, num_epochs):

        model.train()

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

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {curr_loss:.4f}")

            # ----- early-stopping -----
            if curr_loss + 5e-3 < best_val:  # “improved by ≥ 5 × 10⁻³”
                best_val = curr_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if (
                epochs_no_improve >= early_stop_patience
                and optimizer.param_groups[0]["lr"] < 1e-6
            ):
                print(f"Stopped at epoch {epoch}")
                break

        if epoch % 50 == 0:
            save_checkpoint_state(model, optimizer, losses, epoch)

    # Save the final model
    save_checkpoint_state(model, optimizer, losses, num_epochs, is_final=True)
    print("Training complete. Model saved.")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final epoch: {num_epochs}")
    print(f"Final losses: {losses}")
