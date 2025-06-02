import util
import math
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img: str,
        seg: str,
        num_classes: int = 13, # Ernie Extended has 13 classes
        patch_size: list[int] = [128, 128, 128],
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Args
        ----
        img, seg : paths to the NIfTI files
        patch_size : 3-tuple [D, H, W] in voxels
        """
        self.num_classes = num_classes
        self.data, self.labels = self._load_img(
            img, seg, patch_size=patch_size, device=device, dtype=dtype
        )

    # ------------------------------------------------------------------ #
    def _load_img(
        self,
        img_path: str,
        seg_path: str,
        patch_size: list[int],
        device: str,
        dtype,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Read a (C=1) MRI and its segmentation, pad to multiples of
        patch_size, then extract overlapping patches (stride = patch/2).

        Returns
        -------
        images : list[Tensor]  – each tensor has shape (1, D, H, W)
        labels : list[Tensor]  – shape (1, D, H, W) and dtype long
        """
        # ---------- load volumes ---------- #
        img_vol = nib.load(img_path).get_fdata().astype(np.float32)
        seg_vol = nib.load(seg_path).get_fdata().astype(np.int64)

        # --------- optional intensity normalisation ---------- #
        img_vol = (img_vol - img_vol.mean()) / (img_vol.std() + 1e-8)

        # shape: (D, H, W)
        D, H, W = img_vol.shape
        pz, py, px = patch_size
        sz, sy, sx = [p // 2 for p in patch_size]  # 50 % overlap

        # ---------- pad if needed so that the last patch fits ---------- #
        pad_D = (math.ceil(D / sz) * sz + pz - sz) - D
        pad_H = (math.ceil(H / sy) * sy + py - sy) - H
        pad_W = (math.ceil(W / sx) * sx + px - sx) - W
        pad_width = (
            (0, pad_D),
            (0, pad_H),
            (0, pad_W),
        )
        img_vol = np.pad(img_vol, pad_width, mode="edge")
        seg_vol = np.pad(seg_vol, pad_width, mode="edge")

        D_pad, H_pad, W_pad = img_vol.shape

        # ---------- extract patches ---------- #
        img_patches: list[torch.Tensor] = []
        seg_patches: list[torch.Tensor] = []

        for z in range(0, D_pad - pz + 1, sz):
            for y in range(0, H_pad - py + 1, sy):
                for x in range(0, W_pad - px + 1, sx):
                    # slicing
                    iz, iy, ix = slice(z, z + pz), slice(y, y + py), slice(x, x + px)

                    img_patch = torch.tensor(
                        img_vol[iz, iy, ix], dtype=dtype, device=device
                    ).unsqueeze(
                        0
                    )  # (1, D, H, W)

                    seg_patch = torch.tensor(
                        seg_vol[iz, iy, ix], dtype=torch.int64, device=device
                    )  # (D, H, W)

                    seg_patch_oh = F.one_hot(
                        seg_patch, num_classes=self.num_classes
                    )  # (D, H, W, C)

                    seg_patch = seg_patch_oh.permute(3, 0, 1, 2)  # (C, D, H, W)

                    img_patches.append(img_patch)
                    seg_patches.append(seg_patch)

        return img_patches, seg_patches

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    # Example usage
    dataset = ValidationDataset(
        img="/Users/sav/Documents/Progetti DTU/medical-segmentator/ErnieExtended/m2m_ernie_extended/T1.nii.gz",
        seg="/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz",
        patch_size=[128, 128, 128],
        device="cpu",
        num_classes=13,
    )
    print(f"Dataset size: {len(dataset)}")
    img, seg = dataset[200]
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
