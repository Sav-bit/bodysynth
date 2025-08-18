import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data_generator import DataGenerator
from training.util import build_CE_weights, build_CE_weights_test, plot_loss
from unet3d import utils
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
import wandb
from monai.losses import DiceLoss, FocalLoss


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


def get_losses(data_gen: DataGenerator = None) -> tuple:
    """
    Returns the loss criterion (CrossEntropyLoss and DiceLoss) for training.
    For readability, the loss is hardcoded here.

    Args:
        data_gen (DataGenerator, optional): Data generator to compute class frequencies
        for CrossEntropyLoss. If None, no weights are applied.
    Returns:
        tuple: A tuple containing the DiceLoss and CrossEntropyLoss criteria.
    """

    ce_weights = None

    if data_gen is not None:
        freq = data_gen.get_class_frequencies()
        ce_weights = build_CE_weights_test(
            freq=freq,
            num_classes=data_gen.get_num_classes(),
        )
        ce_weights = torch.tensor(ce_weights, dtype=torch.float32, device=data_gen.device)
        print(f"CrossEntropyLoss weights: {ce_weights}")

    # dice_loss_config = {
    #     "loss": {
    #         "name": "DiceLoss",
    #         "normalization": "softmax",
    #         "weight": dice_weights if data_gen else None,
    #     }
    # }

    # dice_loss = get_loss_criterion(dice_loss_config)

    # cross_entropy_loss = get_loss_criterion(
    #     {
    #         "loss": {
    #             "name": "CrossEntropyLoss",
    #             "weight": ce_weights if data_gen else None,
    #         }
    #     }
    # )

    dice_loss = DiceLoss(
        softmax=True,
        to_onehot_y=False,  # your target is already one-hot
        include_background=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )

    focal_ce = FocalLoss(
        to_onehot_y=True,  # your target is already one-hot
        include_background=True,  # background still participates in CE
        gamma=2.0,
        weight=ce_weights,  # your clamped mean=1 weights
        reduction="mean",
    )

    return dice_loss, focal_ce


def merge_losses(dice_loss, cross_entropy_loss, model: AbstractUNet = None, alpha_start=0.8, alpha_end=0.3, decay_steps=2000):
    """
    Merges the two loss functions into one.
    """

    def merged_loss(prediction, segs_onehot, step=None):
        # segs_onehot: [N,C,D,H,W], float32 (may already be one-hot)
        # Sanitize to strictly one-hot:
        segs_bin = (segs_onehot > 0).float()
        labels   = segs_bin.argmax(dim=1, keepdim=True).long()              # [N,1,D,H,W]
        segs_1hot = torch.zeros_like(segs_bin).scatter_(1, labels, 1.0)     # [N,C,D,H,W]

        # Compute losses
        dice_term = dice_loss(prediction, segs_1hot)            # excludes background
        ce_term   = cross_entropy_loss(prediction, labels)                # indices + to_onehot_y=True

        # CE weight schedule (optimizer-step based)
        s = 0 if step is None else step
        t = min(s / decay_steps, 1.0)
        alpha = alpha_start + (alpha_end - alpha_start) * t

        # Optional logging
        if model is not None:
            key = "train" if model.training else "validation"
            wandb.log({f"{key}/dice_loss": float(dice_term.item()),
                       f"{key}/focal_ce":   float(ce_term.item()),
                       f"{key}/alpha_ce":   float(alpha)})

        return dice_term + (alpha * ce_term)

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
    # plot_loss(train_losses, val_lossess, save_plot=True, run_name=run_name)
