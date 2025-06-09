"""
This is the script for training the UNet3D model.
"""

import argparse
from itertools import islice
import torch
from training.data_generator import DataGenerator
from training.util import plot_loss
from training.validation_dataset import ValidationDataset
from unet3d import utils
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import copy


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


def get_validation_data_loader(
    segmentation_path: str,
    image_path: str,
    batch_size: int,
    num_workers: int,
    patch_size: list = [128, 128, 128],
):
    dataset = ValidationDataset(
        img=image_path,
        seg=segmentation_path,
        patch_size=patch_size,
        device="cpu",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )


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


def get_losses():
    """
    Returns the loss criterion.
    For readability, the loss is hardcoded here.
    """
    # Define your loss configuration
    dice_loss_config = {
        "loss": {
            "name": "DiceLoss",
            "normalization": "softmax",
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
    return dice_loss, cross_entropy_loss


def merge_losses(dice_loss, cross_entropy_loss):
    """
    Merges the two loss functions into one.
    """

    def merged_loss(prediction, segs_onehot):
        # segs_onehot: LongTensor or FloatTensor, shape [N, C, D, H, W]
        # 1) Dice wants [N,C,…] float probabilities / one-hot
        dice_term = dice_loss(prediction, segs_onehot)

        # 2) CE wants [N, D, H, W] LongTensor of class indices
        labels = segs_onehot.argmax(dim=1)  # → [N, D, H, W]
        ce_term = cross_entropy_loss(prediction, labels.long())

        return dice_term + ce_term

    return merged_loss


def save_checkpoint_state(
    model,
    optimizer,
    train_losses,
    epoch,
    learning_rate,
    is_final=False,
    val_lossess=None,
    run_name=None,
    is_best=False,
):
    checkpoint_dir = "./checkpoints"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_losses,
        "is_final": is_final,
        "val_loss": val_lossess,
        "learning_rate": learning_rate,
    }
    utils.save_checkpoint(state, is_best, checkpoint_dir, title=run_name)
    plot_loss(train_losses, val_lossess, save_plot=True, run_name=run_name)


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

    parser.add_argument(
        "--validation_path",
        type=str,
        default=None,
        help="Path to the validation data (optional)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run for logging purposes",
    )

    args = parser.parse_args()

    seg_path = args.seg_path
    continue_training = args.continue_training
    val_path = args.validation_path
    run_name = args.run_name

    # -----------------------------
    # End of the arguments
    # -----------------------------

    # Check the PyTorch version
    print("PyTorch version:", torch.__version__)

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    # Set static parameters
    num_epochs = 5000  # How many epochs to train
    batch_size = 1  # How many images to load at once
    num_batches_per_epoch = 10  # How many batches to load per epoch
    patch_size = [150, 150, 150]
    VALIDATION_INTERVAL = 10  # How often to validate the model
    LEARNING_RATE = 3e-4  # Learning rate for the optimizer

    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=device,
        num_workers=0,
        patch_size=patch_size,
    )

    # If validation path is provided, get the validation data loader
    if val_path:
        val_loader = get_validation_data_loader(
            segmentation_path=seg_path,
            image_path=val_path,
            batch_size=batch_size,
            num_workers=0,
            patch_size=patch_size,
        )
        print(f"Validation data loader created with {len(val_loader)} batches.")
    else:
        val_loader = None
        print("No validation data loader created.")

    # Get the model
    model = get_model(data_gen=data_gen).to(device=device)

    print(
        f"[DEBUG...] The number of classes in the model: {data_gen.dataset.get_num_classes()}"
    )

    # Get the loss criterion
    dice_loss, cross_entropy_loss = get_losses()

    # Merge the two loss functions
    criterion = merge_losses(dice_loss, cross_entropy_loss)

    # Get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # The lossess here are a dict where keys are the epoch numbers and values are the losses
    train_losses = {}
    validation_losses = {}
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
        train_losses = retrieved_state["loss"]
        if "val_loss" in retrieved_state:
            validation_losses = retrieved_state["val_loss"]

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.2, last_epoch=last_epoch - 1)

    # Early-stop settings
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 60  # epochs to wait for an improvement
    min_delta = 5e-3  # minimum drop in loss to count as “improvement”
    patience_counter = 0

    # Training loop
    for epoch in range(last_epoch, num_epochs):

        model.train()

        batch_losses = []

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
            batch_losses.append(curr_loss)

        # Compute the average loss for the epoch
        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses[epoch + 1] = epoch_loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # scheduler.step()

        if val_loader and (epoch + 1) % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for val_images, val_segs in val_loader:
                    val_images = val_images.to(device)
                    val_segs = val_segs.to(device)
                    val_prediction = model(val_images)
                    val_loss = criterion(val_prediction, val_segs)
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"Validation Loss at epoch {epoch + 1}: {avg_val_loss:.4f}")
                validation_losses[epoch + 1] = avg_val_loss

            # ———————————— FREE UP GPU MEMORY BEFORE GOING BACK TO TRAIN ————————————
            # Delete the last‐used validation tensors so they drop out of scope:
            del val_images, val_segs, val_prediction, val_loss
            # This will (mostly) clear PyTorch’s cached blocks:
            torch.cuda.empty_cache()

            # Return the model to training mode and clear any leftover gradients:
            model.train()
            optimizer.zero_grad()

            # —— EARLY-STOPPING CHECK ——
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print("Validation loss improved. Saving model state.")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} :Best val loss: {best_val_loss:.4f}"
                )
                break

        if (epoch + 1) % 50 == 0:
            save_checkpoint_state(
                model=model,
                optimizer=optimizer,
                train_losses=train_losses,
                epoch=epoch,
                val_lossess=validation_losses,
                run_name=run_name,
                learning_rate=LEARNING_RATE,
                is_best=False,
            )

    model.load_state_dict(best_model_wts)
    # Save the final model
    save_checkpoint_state(
        model=model,
        optimizer=optimizer,
        train_losses=train_losses,
        epoch=num_epochs,
        val_lossess=validation_losses,
        run_name=run_name,
        is_final=True,
        is_best=True,
    )
    print("Training complete. Model saved.")
