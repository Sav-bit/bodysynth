from itertools import islice
from typing import Tuple
import torch
import nibabel as nib
import brainsynth
from training.util import build_CE_weights, save_representation
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader
import numpy as np

"""
Remember in this file:
- N : batch size
- C : channels
- D : depth
- H : height
- W : width
"""


class DataGenerator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        seg_dir,
        out_center_str="image",
        patch_size=[128, 128, 128],
        padding: int = 22,
        device="cpu",
    ):
        self.seg_dir = seg_dir
        self.device = device
        self.patch_size = patch_size
        self.padding = padding

        self.original_data, self.affine = self.load_data()

        # Since the memory is not enough to load the full image, we need to set the out_size
        # The idea here is to set the out_size as a little bit bigger than the patch size
        # so we avoid having the black border around the image in case of non linear transformation
        # then we will crop it to the patch size

        out_size = [x + self.padding for x in self.patch_size]

        self.synth = brainsynth.Synthesizer(
            brainsynth.config.SynthesizerConfig(
                builder="SaverioSynth",
                out_size=out_size,
                out_center_str=out_center_str,
                segmentation_labels="ernie",
                device=self.device,
            )
        )

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the segmentation data from the NIfTI file.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the segmentation data and the affine transformation matrix.
        """

        img = nib.load(self.seg_dir)
        return img.get_fdata().astype(np.int64), img.affine

    def get_original_segmentation(self) -> torch.Tensor:
        """
        Returns the original segmentation data.
        """
        return self.original_data

    def __iter__(self):
        """
        Returns an iterator that yields batches of data.
        """
        while True:
            # Get a random patch from the original segmentation
            # Then we will use the synthesizer to generate a new image from that patch
            to_synth = dict(segmentation=self.get_random_patch())

            result = self.synth(to_synth, unpack=False)

            # This has size (C, D, H, W) where C is the number of channels, 1 in the image case
            image = result["image"]

            # This has size  (C, D, H, W)
            segmentation = result["seg"].to(torch.int64)

            if self.padding > 0:
                sl = slice(self.padding // 2, -self.padding // 2)

                # Crop the image to the patch size
                image = image[
                    :,
                    sl,
                    sl,
                    sl,
                ]

                # Crop the segmentation to the patch size
                segmentation = segmentation[
                    :,
                    sl,
                    sl,
                    sl,
                ]

            yield image, segmentation

    def __repr__(self) -> str:
        """
        Returns a string representation of the DataGenerator.
        """
        return f"DataGenerator(seg_dir={self.seg_dir}, batch_size={self.batch_size}, out_size={self.synth.out_size})"

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and segmentation tensors.
        """
        return self.__getitem__(0)

    def get_num_classes(self) -> int:
        """
        Returns the number of classes in the segmentation data.
        """
        return int(self.get_original_segmentation().max() + 1)

    def get_random_patch(self) -> torch.tensor:

        seg = self.get_original_segmentation()

        # Get the shape of the image
        D, H, W = seg.shape

        isMostBackground = True

        # calculate the total patch size
        # patch size + padding
        total_patch_size = [x + self.padding for x in self.patch_size]

        while isMostBackground:
            # Get random coordinates for the patch
            d = torch.randint(0, D - total_patch_size[0], (1,))
            h = torch.randint(0, H - total_patch_size[1], (1,))
            w = torch.randint(0, W - total_patch_size[2], (1,))

            segmentation_patch = seg[
                d : d + total_patch_size[0],
                h : h + total_patch_size[1],
                w : w + total_patch_size[2],
            ]

            # Check if the segmentation_patch is mostly background
            isMostBackground = self._is_mostly_background(
                segmentation_patch,
                threshold=0.2,
            )
            
        # Convert the patch to a tensor and move it to the device
        segmentation_patch = torch.tensor(segmentation_patch, device=self.device, dtype=torch.int64).unsqueeze(0)

        return segmentation_patch

    def get_affine(self) -> torch.Tensor:
        """
        Returns the affine transformation matrix of the NIfTI file.
        """
        return self.affine

    def _is_mostly_background(
        self,
        patch: np.ndarray,
        threshold=0.2,
    ) -> bool:
        """
        Check if the patch is mostly background
        """
        # Count the number of non-background pixels
        num_non_background = np.count_nonzero(patch > 0)
        # Count the total number of pixels in the patch
        num_total_pixels = patch.size
        # Check if the patch is mostly background
        return num_non_background / num_total_pixels < threshold

    
    def get_class_frequencies(self) -> dict[int, float]:
        """
        Returns a dictiornay with the class frequencies in the original segmentation.
        The keys are the class labels and the values are the frequencies.
        """
        seg = self.get_original_segmentation()
        unique, counts = np.unique(seg, return_counts=True)
        frequencies = dict(zip(unique, counts / seg.size))
        return frequencies

    # ---------------- test ----------------

if __name__ == "__main__":

    device = torch.device("cpu")

    # instantiate the data generator
    data_gen = DataGenerator(
        seg_dir="/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz",
        device=device,
        patch_size=[128, 128, 128],
        padding=22,
    )

    loader = DataLoader(
        data_gen,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
    )

    i = 0

    num_classes = data_gen.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    
    print(f"Class frequencies: {data_gen.get_class_frequencies()}")
    
    ce_weights = build_CE_weights(
        class_frequencies=data_gen.get_class_frequencies(),
        num_classes=num_classes,
    )
    
    print(f"CE weights: {ce_weights}")
    print(f"CE weights shape: {ce_weights.shape}")
    

    model: AbstractUNet = UNet3D(
        in_channels=1,
        out_channels=num_classes,
        final_sigmoid=True,
        # f_maps=[32, 64, 128, 256],
        f_maps=[16, 32, 64],
        layer_order="gcr",
        # num_groups=8,
        # num_levels=4,
        num_groups=4,
        num_levels=3,
        is_segmentation=True,
        conv_padding=1,
        upsample="default",
        dropout_prob=0.5,
        is3d=True,
    ).to(device="cpu")

    # Define your loss configuration
    loss_config = {
        "loss": {
            "name": "DiceLoss",
            "normalization": "sigmoid",
            "weight": ce_weights,
        }
    }

    save_representation(
        image=data_gen.get_random_patch(),
        title="Random_patch",
        affine_matrix=data_gen.get_affine(),
    )

    # save_representation(
    #     image=data_gen.get_original_segmentation(),
    #     title="Original_segmentation",
    #     affine_matrix=data_gen.get_affine(),
    # )

    print(f"Original affine: {data_gen.get_affine()}")

    # Create the loss criterion
    criterion = get_loss_criterion(loss_config)

    num_batches_per_epoch = 3
    num_epochs = 2

    for i in range(num_epochs):
        print(f"Iteration {i + 1}")

        count_batches = 0
        for images, segs in islice(loader, num_batches_per_epoch):
            print(f"I'm in the loop")
            
            predicted = model(images)
            print(f"Predicted shape: {predicted.shape}")
            print(f"Images shape: {images.shape}")
            

            print(f"Image shape: {images.shape}")
            print(f"Segmentation shape: {segs.shape}")
            for j in range(images.shape[0]):
                image = images[j]
                seg = segs[j]

                # Save the image and segmentation
                # save_representation(
                #     image=image,
                #     title=f"image_{i}_{count_batches}_{j}",
                # )
                # save_representation(
                #     image=seg,
                #     title=f"segmentation_{i}_{count_batches}_{j}",
                # )

            # save_representation(
            #     image=image,
            #     title=f"image_{i}",
            # )
            # save_representation(
            #     image=seg,
            #     title=f"segmentation_{i}",
            # )
            count_batches += 1

        print(f"Number of batches in this epoch: {count_batches}")

    print("Out of the loop...")
