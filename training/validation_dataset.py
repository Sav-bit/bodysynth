from itertools import islice
import util
import math
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from training.transform import create_transforms
import torchio as tio
from torch.utils.data import DataLoader



class ValidationDataset(torch.utils.data.Dataset):
    """
    Streams 3-D patches from a (C=1) MRI volume + segmentation.

    Parameters
    ----------
    img, seg : str | Path
        Paths to NIfTI files.
    patch_size : (D, H, W) in voxels.
    stride : optional patch stride.  default = patch_size // 2  (50 % overlap)
    """

    def __init__(
        self,
        img: str | Path,
        seg: str | Path,
        num_classes: int = 10,
        patch_size: Sequence[int] = (128, 128, 128),
        stride: Sequence[int] | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        training_mode: bool = False,
    ):
        self.num_classes = num_classes
        self.patch_size = tuple(int(p) for p in patch_size)
        self.stride = (
            tuple(int(s) for s in stride) if stride is not None else tuple(p // 2 for p in patch_size)
        )
        self.device = torch.device(device)
        self.dtype = dtype

        # ------------------------------------------------------------------ #
        # 1. read & normalise ------------------------------------------------
        img_vol = nib.load(str(img)).get_fdata().astype(np.float32)
        seg_vol = nib.load(str(seg)).get_fdata().astype(np.int64)

        img_vol = (img_vol - img_vol.mean()) / (img_vol.std() + 1e-8)

        # 2. pad so that the last patch fits ---------------------------------
        self.img, self.seg = self._pad_volumes(img_vol, seg_vol)

        # 3. pre-compute all patch start indices -----------------------------
        self.starts = self._compute_patch_grid()
        self.training_mode = training_mode
        if self.training_mode:
            self.transform = create_transforms()

    # ---------------------------------------------------------------------- #
    #                          helper functions                               #
    # ---------------------------------------------------------------------- #
    def _pad_volumes(self, img: np.ndarray, seg: np.ndarray):
        D, H, W = img.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride

        pad_D = (math.ceil(D / sz) * sz + pz - sz) - D
        pad_H = (math.ceil(H / sy) * sy + py - sy) - H
        pad_W = (math.ceil(W / sx) * sx + px - sx) - W
        pad_width = ((0, pad_D), (0, pad_H), (0, pad_W))

        img = np.pad(img, pad_width, mode="edge")
        seg = np.pad(seg, pad_width, mode="edge")

        # keep one copy of each volume in RAM on the target device
        img = torch.as_tensor(img, dtype=self.dtype, device=self.device)
        seg = torch.as_tensor(seg, dtype=torch.long, device=self.device)

        return img, seg

    def _compute_patch_grid(self):
        D, H, W = self.img.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride

        starts = [
            (z, y, x)
            for z in range(0, D - pz + 1, sz)
            for y in range(0, H - py + 1, sy)
            for x in range(0, W - px + 1, sx)
        ]
        return starts
    
    def get_class_frequencies(self) -> dict[int, float]:
        """
        Returns a dictiornay with the class frequencies in the original segmentation.
        The keys are the class labels and the values are the frequencies.
        """
        seg = self.seg.cpu().numpy()
        unique, counts = np.unique(seg, return_counts=True)
        frequencies = dict(zip(unique, counts / seg.size))
        frequencies = {int(k): v for k, v in frequencies.items()}
        return frequencies
    
    def get_num_classes(self) -> int:
        """
        Returns the number of classes in the segmentation.
        """
        return self.num_classes

    # ---------------------------------------------------------------------- #
    #                        PyTorch Dataset API                              #
    # ---------------------------------------------------------------------- #
    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        
        if self.training_mode :
            #Get a random index if in training mode
            idx = torch.randint(0, len(self.starts), (1,)).item()
        
        z, y, x = self.starts[idx]
        pz, py, px = self.patch_size

        iz, iy, ix = slice(z, z + pz), slice(y, y + py), slice(x, x + px)

        # (1, D, H, W) label map
        img_patch = self.img[iz, iy, ix].unsqueeze(0)

        # label map -> one-hot (C, D, H, W) float32
        seg_patch = self.seg[iz, iy, ix]
        seg_patch = F.one_hot(seg_patch, self.num_classes).permute(3, 0, 1, 2).float()
        
        if self.training_mode and self.transform:
            t1 = tio.ScalarImage(tensor=img_patch)
            seg = tio.LabelMap(tensor=seg_patch)
            subject = tio.Subject(t1=t1, seg=seg)
            subject = self.transform(subject)
            img_patch = subject.t1.data.to(device=self.device)
            seg_patch = subject.seg.data.to(device=self.device)
            
        return img_patch, seg_patch


if __name__ == "__main__":
    # Example usage
    dataset = ValidationDataset(
        img="/Users/sav/Documents/Progetti DTU/medical-segmentator/ErnieExtended/m2m_ernie_extended/T1.nii.gz",
        seg="/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz",
        patch_size=[128, 128, 128],
        device="cpu",
        num_classes=13,
        training_mode=True,  # Set to True if you want to apply transformations
    )
    print(f"Dataset size: {len(dataset)}")
    img, seg = dataset[30]
    print(f"Image shape: {img.shape}, Segmentation shape: {seg.shape}")

    # Let's save the first image and segmentation patch to verify
    # util.save_representation(
    #     image=img,
    #     title="test_image",
    #     image_index=0,
    # )
    # util.save_representation(
    #     image=seg,
    #     title="test_segmentation",
    #     image_index=0,
    # )
    
    #test the class frequencies
    frequencies = dataset.get_class_frequencies()
    print(f"Class frequencies: {frequencies}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    
    num_batches_per_epoch = 4
    
    print(f"DataLoader size: {len(data_loader)}")
    for images, segs in islice(data_loader, num_batches_per_epoch):
        print(f"Batch image shape: {images.shape}, Batch segmentation shape: {segs.shape}")
