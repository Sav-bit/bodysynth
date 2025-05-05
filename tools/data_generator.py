import numpy as np
import torch
import nibabel as nib
import brainsynth
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D

"""
Remember in this file:
- N : batch size (It's always 1)
- C : channels
- D : depth
- H : height
- W : width
"""


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
        self, seg_dir, batch_size=1, out_center_str="image", out_size=None, device="cpu"
    ):
        self.seg_dir = seg_dir
        self.device = device
        self.batch_size = batch_size

        self.original_data = self.load_data()

        if out_size is None:
            out_size = self.get_out_size()

        self.synth = brainsynth.Synthesizer(
            brainsynth.config.SynthesizerConfig(
                builder="SaverioSynth",
                out_size=out_size,
                out_center_str=out_center_str,
                segmentation_labels="ernie",
                device=self.device,
            )
        )

    def get_out_size(self) -> list[int]:

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

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        # The dataset is not iterable, so we return 1
        # to avoid errors in the DataLoader
        return self.batch_size

    def get_original_segmentation(self) -> torch.Tensor:
        """
        Returns the original segmentation data.
        """
        return self.original_data["segmentation"]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and segmentation tensors.
            note: both tensors have size (1, D, H, W)
        """

        if not 0 <= index < self.batch_size:
            raise IndexError("Index out of range")

        result = self.synth(self.original_data, unpack=False)

        # This has size (1, C, D, H, W) where C is the number of channels
        # 1 in this case
        image = result["image"].unsqueeze(0)

        # This has size  (1, C, D, H, W)
        segmentation = result["seg"].to(torch.int64).unsqueeze(0)

        return image, segmentation

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

    def get_random_patch(
        self,
        image: torch.Tensor,
        segmentation: torch.Tensor,
        patch_size=[128, 128, 128],
        delete_original=True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random patch of data.
        """

        # Get the shape of the image
        _, _, D, H, W = image.shape

        isMostBackground = True

        while isMostBackground:
            # Get random coordinates for the patch
            d = torch.randint(0, D - patch_size[0], (1,))
            h = torch.randint(0, H - patch_size[1], (1,))
            w = torch.randint(0, W - patch_size[2], (1,))

            # Get the patch
            image_patch = image[
                :,
                :,
                d : d + patch_size[0],
                h : h + patch_size[1],
                w : w + patch_size[2],
            ]
            segmentation_patch = segmentation[
                :,
                :,
                d : d + patch_size[0],
                h : h + patch_size[1],
                w : w + patch_size[2],
            ]

            # Check if the segmentation_patch is mostly background
            # We consider the background to be 0
            # If the patch is mostly background, we skip it

            # Count the number of non-background pixels
            num_non_background = (segmentation_patch != 0).sum().item()
            # Count the total number of pixels in the patch
            num_total_pixels = patch_size[0] * patch_size[1] * patch_size[2]
            # Check if the patch is mostly background
            # let's say that if the patch has less than 20% of non-background pixels, we consider it as mostly background
            isMostBackground = num_non_background / num_total_pixels < 0.2
            # If the patch is mostly background, we skip it
            # If the patch is not mostly background, we break the loop
            if not isMostBackground:
                break
        # Return the patch

        if delete_original:
            del image, segmentation

        return image_patch, segmentation_patch

    # ---------------- test ----------------


if __name__ == "__main__":

    device = torch.device("cpu")

    # instantiate the data generator
    data_gen = DataGenerator(
        seg_dir="/Users/sav/Documents/Progetti DTU/medical-segmentator/ernie_less_dim.nii.gz",
        batch_size=1,
        device=device,
        out_size=[128, 128, 128],
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

    for image, seg in data_gen:
        print(f"I'm in the loop")
        # prediction = model(image)

        # print(f"Prediction shape: {prediction.shape}")

        patch_size = [70, 70, 70]

        image_patch, seg_patch = data_gen.get_random_patch(image, seg, patch_size, delete_original=False)
        print(f"Image patch shape: {image_patch.shape}")
        print(f"Segmentation patch shape: {seg_patch.shape}")

        # save the image and its patch
        toSave = nib.Nifti1Image(image_patch[0][0].cpu().numpy(), affine=np.eye(4))
        nib.save(toSave, f"image_patch_{i}.nii.gz")
        # save also the original image
        toSave = nib.Nifti1Image(image[0][0].cpu().numpy(), affine=np.eye(4))
        nib.save(toSave, f"image_{i}.nii.gz")
        # Compute the loss
        # loss = criterion(prediction, seg)
        # print(f"Loss: {loss.item()}")
#        del image, seg
