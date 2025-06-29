"""
Sliding‑window inference script for the 3‑D UNet you trained.

Example
-------
python sliding_window_inference.py \
        --image_path path/to/volume.nii.gz \
        --checkpoint ./checkpoints/last_checkpoint.pytorch \
        --out_path pred_seg.nii.gz \
        --patch_size 180 180 180 \
        --stride 90 90 90 \
        --sw_batch_size 4

If you prefer MONAI’s helper, replace the manual `sliding_window_predict` call
with `monai.inferers.sliding_window_inference(...)`.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import nibabel as nib

from unet3d.model import UNet3D
from unet3d import utils
from monai.inferers import sliding_window_inference
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import ornt_transform


def get_orientation(nii: nib.Nifti1Image) -> tuple[str, str, str]:
    """Gets the orientation of a nifti image."""
    orientation = nib.aff2axcodes(nii.affine)
    return orientation


def reorient(
    nii: nib.Nifti1Image,
    orientation: str | tuple[str, str, str] = "RAS",
) -> nib.Nifti1Image:
    """Reorients a nifti image to specified orientation. Orientation string or tuple
    must consist of "R" or "L", "A" or "P", and "I" or "S" in any order."""
    orig_ornt = nib.io_orientation(nii.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii


def get_device() -> torch.device:
    """Return GPU device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> UNet3D:
    """Build the UNet architecture exactly as during training and load weights."""

    model = UNet3D(
        in_channels=1,
        out_channels=num_classes,
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
    ).to(device)

    utils.load_checkpoint(checkpoint_path=checkpoint_path, model=model)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Sliding‑window inference for 3‑D UNet"
    )
    parser.add_argument(
        "--image_path", required=True, help="Input volume (NIfTI or MHD)"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint .pytorch file"
    )
    parser.add_argument(
        "--out_path", required=True, help="Output segmentation NIfTI filename"
    )
    parser.add_argument(
        "--num_classes", type=int, default=13, help="Number of target classes"
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[170, 170, 170])
    parser.add_argument(
        "--sw_batch_size", type=int, default=1, help="How many patches per GPU batch"
    )

    args = parser.parse_args()

    device = get_device()
    print("Running on", device)

    # 1. Load and normalise volume – keep exactly the preprocessing used during training!
    img = nib.load(args.image_path)
    if get_orientation(img) != ("P", "S", "R"):
        print("Reorienting image to PSR orientation...")
        # Reorient to PSR (Posterior-Superior-Right) orientation
        # This is the same as Ernie Extended's orientation
        # which is required for the model to work correctly
        # as it was trained on images in this orientation.
        img = reorient(img, "PSR")
    vol = img.get_fdata().astype(np.float32)  # Load volume data as float32
    vol_tensor = torch.from_numpy(vol[None, None]).to(device)  # → (1,1,D,H,W)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)  # simple z‑score

    # 2. Load network
    model = load_model(Path(args.checkpoint), args.num_classes, device)

    # 3. Predict
    patch_size = tuple(args.patch_size)

    # probs = sliding_window_predict(
    #     model,
    #     vol,
    #     patch_size=patch_size,
    #     stride=stride,
    #     sw_batch_size=args.sw_batch_size,
    #     device=device,
    # )

    with torch.no_grad():
        seg_probs = sliding_window_inference(
            inputs=vol_tensor,
            roi_size=patch_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=0.5,  # 50% overlap
            mode="gaussian",
            device=device,
        )

    seg = seg_probs.argmax(dim=1).squeeze(0).cpu().numpy()

    # 4. Save segmentation
    out_img = nib.Nifti1Image(seg, img.affine, img.header)
    nib.save(out_img, args.out_path)
    print("Segmentation saved to", args.out_path)


if __name__ == "__main__":
    main()

    # #let's do a simple test
    # # Load the model
    # device = get_device()
    # #let's create a dummy model
    # model = UNet3D(
    #     in_channels=1,
    #     out_channels=2,
    #     f_maps=(32, 64, 128),
    #     layer_order="cgr",
    #     num_groups=8,
    #     final_sigmoid=False,
    #     conv_kernel_size=3,
    #     pool_kernel_size=2,
    #     conv_padding=1,
    #     conv_upscale=2,
    #     upsample="deconv",
    #     num_levels=5,
    #     dropout_prob=0.0,
    #     is_segmentation=True,
    #     is3d=True,
    # ).to(device)
    # model.eval()

    # # let's create a dummy patch size of 32, 32, 32
    # patch_size = (32, 32, 32)
    # # let's create a dummy stride of 16, 16, 16
    # stride = (16, 16, 16)
    # # let's create a dummy sw_batch_size of 4
    # sw_batch_size = 4
    # # let's create a dummy volume of 128, 128, 128
    # volume = torch.randn(128, 128, 128).to(device)

    # print("About to run sliding_window_predict....")

    # probs = sliding_window_predict(
    #     model=model,
    #     volume=volume.cpu().numpy(),
    #     patch_size=patch_size,
    #     stride=stride,
    #     sw_batch_size=sw_batch_size,
    #     device=device,
    # )

    # seg = probs.argmax(0).astype(np.uint8)

    # print("Segmentation shape:", seg.shape)

    # out_img = nib.Nifti1Image(seg, np.eye(4), None)
    # nib.save(out_img, "test_segmentation.nii.gz")
    # print("Segmentation saved to", "test_segmentation.nii.gz")

    # #save also the volume
    # out_img = nib.Nifti1Image(volume.cpu().numpy(), np.eye(4), None)
    # nib.save(out_img, "test_volume.nii.gz")
    # print("Volume saved to", "test_volume.nii.gz")
