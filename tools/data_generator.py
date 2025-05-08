from itertools import islice
import torch
import nibabel as nib
import brainsynth
from tools.util import save_representation
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader

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
        padding=22,
        device="cpu",
    ):
        self.seg_dir = seg_dir
        self.device = device
        self.patch_size = patch_size
        self.padding = padding

        self.original_data = self.load_data()

        # Since the memory is not enough to load the full image, we need to set the out_size
        # The idea here is to set the out_size as a little bit bigger than the patch size
        # so we avoid having the black border around the image in case of non linear transformation
        # then we will crop it to the patch size
        out_size = [x + padding for x in patch_size]

        self.synth = brainsynth.Synthesizer(
            brainsynth.config.SynthesizerConfig(
                builder="SaverioSynth",
                out_size=out_size,
                out_center_str=out_center_str,
                segmentation_labels="ernie",
                device=self.device,
            )
        )

    def _get_out_size(self) -> list[int]:

        segmentation_size = self.get_original_segmentation().shape[1:]

        # The synthesizer requires the output size to be even
        next_even_number = lambda x: x if x % 2 == 0 else x + 1

        out_size = [next_even_number(x) for x in segmentation_size]

        return out_size

    def load_data(self) -> dict[str, torch.Tensor]:
        """
        Loads the segmentation data from the specified NIfTI file.

        Returns:
            dict[str, torch.Tensor]: A dictionary with key "segmentation"
            and its value as a 4D torch.Tensor (1, D, H, W).
        """
        img = nib.load(self.seg_dir)
        data = img.get_fdata()
        data = torch.tensor(data, device=self.device, dtype=torch.int64).unsqueeze(0)
        return dict(segmentation=data)

    def get_original_segmentation(self) -> torch.Tensor:
        """
        Returns the original segmentation data.
        """
        return self.original_data["segmentation"]

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
        return self.get_original_segmentation().max() + 1

    def get_random_patch(self) -> torch.tensor:

        seg = self.get_original_segmentation()

        # Get the shape of the image
        _, D, H, W = seg.shape

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
                :,
                d : d + total_patch_size[0],
                h : h + total_patch_size[1],
                w : w + total_patch_size[2],
            ]

            # Check if the segmentation_patch is mostly background
            isMostBackground = self._is_mostly_background(
                segmentation_patch,
                threshold=0.2,
            )

        return segmentation_patch

    def _is_mostly_background(
        self,
        patch: torch.Tensor,
        threshold=0.2,
    ) -> bool:
        """
        Check if the patch is mostly background
        """
        # Count the number of non-background pixels
        num_non_background = (patch != 0).sum().item()
        # Count the total number of pixels in the patch
        num_total_pixels = patch.numel()
        # Check if the patch is mostly background
        return num_non_background / num_total_pixels < threshold

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

    num_classes = data_gen.get_original_segmentation().max() + 1

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
            # additional parameters can go here if needed...
        }
    }

    # Create the loss criterion
    criterion = get_loss_criterion(loss_config)
    
    num_batches_per_epoch = 3
    num_epochs = 2

    for i in range(num_epochs):
        print(f"Iteration {i + 1}")
        
        count_batches = 0
        for images, segs in islice(loader, num_batches_per_epoch):
            print(f"I'm in the loop")

            print(f"Image shape: {images.shape}")
            print(f"Segmentation shape: {segs.shape}")

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
