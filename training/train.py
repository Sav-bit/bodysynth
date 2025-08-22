"""
This is the script for training the UNet3D model.
"""

import argparse
from itertools import islice
import torch
from training.data_generator import DataGenerator
from training.validation_dataset import ValidationDataset
from unet3d import utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import copy
import wandb
from training.train_util import (
    get_device,
    get_model,
    get_losses,
    merge_losses,
    save_checkpoint_state,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        pin_memory=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet3D with a segmentation path and brainSynth image generator."
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

    parser.add_argument("--patch_size", type=int, nargs=3, default=[170, 170, 170])

    parser.add_argument("--small_description", type=str, default='', help="A small description of the run")

    args = parser.parse_args()

    seg_path = args.seg_path
    continue_training = args.continue_training
    val_path = args.validation_path
    run_name = args.run_name
    description = args.small_description

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
    num_batches_per_epoch = 50  # How many batches to load per epoch
    patch_size = args.patch_size

    VALIDATION_INTERVAL = 2  # How often to validate the model
    LEARNING_RATE = 1e-4  # Learning rate for the optimizer

    run = wandb.init(
        project="bodysynth",
        name=run_name if run_name else "UNet3D Training w BodySynth",
        notes=description,
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_batches_per_epoch": num_batches_per_epoch,
            "patch_size": patch_size,
            "validation_interval": VALIDATION_INTERVAL,
            # "learning_rate": LEARNING_RATE,
        },
    )

    # Get the data generator
    data_gen = get_data_generator(
        seg_path=seg_path,
        batch_size=batch_size,
        device=torch.device("cpu"),
        num_workers=8,
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

    wandb.watch(model, log="all")

    # Get the loss criterion
    dice_loss, cross_entropy_loss = get_losses(data_gen.dataset)

    # Merge the two loss functions
    criterion = merge_losses(dice_loss, cross_entropy_loss, model)

    # Get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # The lossess here are a dict where keys are the epoch numbers and values are the losses
    train_losses = {}
    validation_losses = {}
    last_epoch = 0

    if continue_training:
        # TODO fix this, there is no last_checkpoint.pytorch file anymore

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

    # Early-stop settings
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    SCHED_PATIENCE = (
        5  # epochs to wait for an improvement before reducing the learning rate
    )
    EARLYSTOP_PATIENCE = 10 * SCHED_PATIENCE  # epochs to wait for an improvement
    min_delta = 1e-3  # minimum drop in loss to count as “improvement”
    patience_counter = 0

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.3,  # LR ← LR × 0.3
    #     patience=SCHED_PATIENCE,
    #     threshold=min_delta,  # same definition of “improve” as early-stop
    #     threshold_mode="rel",
    #     cooldown=1,
    #     min_lr=1e-5,
    #     verbose=True,
    # )
    
    global_update = 0
    # ACCUM = 8

    # Training loop
    for epoch in range(last_epoch, num_epochs):

        model.train()
        optimizer.zero_grad()

        batch_losses = []

        # The data generator is infinite, so we need to limit the number of batches
        for batch_idx, (images, segs) in enumerate(islice(data_gen, num_batches_per_epoch), 1):

            # Normalize the images performing zscore normalization
            
            images = images.to(device, non_blocking=True)
            segs = segs.to(device, non_blocking=True)

            images = (images - images.mean()) / (images.std() + 1e-6)
            
            # Forward pass
            prediction = model(images)

            # Compute the loss
            loss = criterion(prediction, segs, global_update)
            loss.backward()
            curr_loss = loss.item()
            batch_losses.append(curr_loss)

            # if batch_idx % ACCUM == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_update += 1

        
        # flush leftover microbatches if epoch isn't a multiple of ACCUM
        # if (batch_idx % ACCUM) != 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     global_update += 1

        # Compute the average loss for the epoch
        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses[epoch + 1] = epoch_loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": epoch_loss})

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
                wandb.log({"epoch": epoch + 1, "validation/loss": avg_val_loss})
                validation_losses[epoch + 1] = avg_val_loss
                # scheduler.step(avg_val_loss)                     
                wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1})

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

            if patience_counter >= EARLYSTOP_PATIENCE:
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
        learning_rate=LEARNING_RATE,
        is_final=True,
        is_best=True,
    )
    print("Training complete. Model saved.")
