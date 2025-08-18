from itertools import islice
from typing import Tuple
from sklearn.cluster import KMeans
import torch
import nibabel as nib
import brainsynth
from training.util import build_CE_weights, save_representation
from unet3d.losses import get_loss_criterion
from unet3d.model import AbstractUNet, UNet3D
from torch.utils.data import DataLoader
import numpy as np
from monai.transforms import Compose, RandCropByLabelClassesd, ToTensord, EnsureChannelFirstd


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

        max_number_classes = ((self.get_num_classes() - 1) * 3) + 1

        self.synth = brainsynth.Synthesizer(
            brainsynth.config.SynthesizerConfig(
                builder="SaverioSynth",
                out_size=out_size,
                out_center_str=out_center_str,
                #segmentation_labels="ernie",
                segmentation_labels=tuple(range(max_number_classes)), # 1 [background] + (9 [original classes] * 3 [split into triplets]) = 28
                device=self.device,
            )
        )
        
        freq = self.get_class_frequencies()              # {class: fraction}
        ratios = np.ones(self.get_num_classes(), dtype=np.float32)
        ratios[0] = 0.2                                   # small chance to pick background
        for c in range(1, self.get_num_classes()):
            f = float(freq.get(c, 0.0))
            ratios[c] = 1.0 / np.sqrt(f + 1e-8)          # rare classes get higher ratio
        ratios = (ratios / ratios.sum()).tolist()
        
        #implementation of the class_crop
        self.class_crop = Compose([
            # EnsureChannelFirstd(keys=["seg"]),
            RandCropByLabelClassesd(
                keys=["seg"],                            # what to crop
                label_key="seg",                         # use seg to choose centers
                spatial_size=out_size,                   # crop size BEFORE your padding-trim
                ratios=ratios,                           # sampling preference per class id
                num_classes=self.get_num_classes(),            # total classes in seg
                num_samples=1                            # return a single crop (dict, not list)
            ),
            ToTensord(keys=["seg"])                    
        ])

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the segmentation data from the NIfTI file.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the segmentation data and the affine transformation matrix.
        """

        img = nib.load(self.seg_dir)
        return img.get_fdata().astype(np.int64), img.affine

    def get_original_segmentation(self) -> np.ndarray:
        """
        Returns the original segmentation data.
        """
        return self.original_data

    def __iter__(self):
        """
        Returns an iterator that yields batches of data.
        """
        while True:
            
            #get a 50% probability
            if torch.rand(1).item() < 0.5:
                random_patch = self.get_random_patch()
            else:
                random_patch = self._get_class_aware_patch()

            # We will split the labels into three
            random_patch = self._split_labels_into_three(random_patch)
            
            # Convert the patch to a tensor and move it to the device
            segmentation_patch = torch.tensor(segmentation_patch, device=self.device, dtype=torch.int64).unsqueeze(0)

            # Get a random patch from the original segmentation
            # Then we will use the synthesizer to generate a new image from that patch
            to_synth = dict(segmentation=random_patch)

            result = self.synth(to_synth, unpack=False)

            # This has size (C, D, H, W) where C is the number of channels, 1 in the image case
            image = result["image"]

            # This has size  (C, D, H, W)
            segmentation = result["seg"].to(torch.int64)

            # The problem is that now C is 37, so we need to collapse the channels into the 13 Ernie classes
            segmentation = self._collapse_triplet_channels(segmentation)

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

    def get_random_patch(self) -> np.ndarray:

        seg = self.get_original_segmentation()

        # Get the shape of the image
        D, H, W = seg.shape

        isMostBackground = True

        # calculate the total patch size
        # patch size + padding
        total_patch_size = [x + self.padding for x in self.patch_size]
        
        #The network need to see some background, so we will randomly flip the isMostBackground flag
        # with a probability of 0.25
        do_i_want_background = torch.rand(1).item() < 0.25

        while isMostBackground:
            # Get random coordinates for the patch
            d = torch.randint(0, D - total_patch_size[0] + 1, (1,))
            h = torch.randint(0, H - total_patch_size[1] + 1, (1,))
            w = torch.randint(0, W - total_patch_size[2] + 1, (1,))

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
            
            if do_i_want_background:
                # If we want background, we will stop when the patch is mostly background
                isMostBackground = not isMostBackground

        return segmentation_patch
    
    
    def _get_class_aware_patch(self) -> np.ndarray:

        temp_seg = np.expand_dims(self.get_original_segmentation(), axis=0)

        sample = self.class_crop({"seg": temp_seg})
        
        seg_patch = sample[0]["seg"]
    
        return seg_patch.numpy().squeeze(0)

    def _collapse_triplet_channels(self, segC : torch.Tensor, k=3) -> torch.Tensor:
        """
        Collapse channels of a segmentation tensor that has been split into triplets.
        Arguments:
            segC: Tensor with shape (C, D, H, W) where C is the number of channels.
            k: Number of channels per label (default is 3).
        Returns:
            A new tensor with shape ((C-1)//k + 1, D, H, W) where the first channel is the background
            and the rest are collapsed labels.
        Note: Assumes the first channel is background (0) and the rest are labels.
        """
        C = segC.shape[0]
        n_base = (C - 1) // k

        out = segC.new_zeros((n_base + 1, *segC.shape[1:]))
        out[0] = segC[0]  # background stays background

        for L in range(1, n_base + 1):
            start = (L - 1) * k + 1       # e.g. 1,4,7,... for k=3
            out[L] = segC[start:start + k].sum(dim=0)  # or .amax(dim=0).values

        return out
    
    def _split_labels_into_three(self, seg : np.ndarray, background=0, spacing=None, random_state=0):
        """
        Split each non-background label region in `seg` into up to 3 spatial clusters
        using KMeans on voxel coordinates (z, y, x). Returns a new segmentation with
        remapped labels:
        background -> 0
        L -> (L-1)*3 + {1,2,3}
        If a region has <3 voxels, it will produce fewer clusters (no warning).
        """
        out = seg.copy()
        uniq = np.unique(seg)

        # scale coordinates by spacing (in mm) if provided, so clustering respects anisotropy
        spacing = np.asarray(spacing, dtype=np.float32) if spacing is not None else None

        for L in uniq:
            if L == background:
                continue

            coords = np.column_stack(np.nonzero(seg == L))  # (N, 3) with [z, y, x]
            N = coords.shape[0]
            if N == 0:
                continue

            k = min(3, N)  # avoid errors if the object is tiny

            X = coords.astype(np.float32)
            if spacing is not None:
                X = X * spacing  # cluster in physical space if zooms are anisotropic

            labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(
                X
            )

            # New label base: 1.. becomes triples: (L-1)*3 + 1..3
            base = (int(L) - 1) * 3 + 1
            for c in range(k):
                pts = coords[labels == c]
                out[pts[:, 0], pts[:, 1], pts[:, 2]] = base + c

        return out

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
