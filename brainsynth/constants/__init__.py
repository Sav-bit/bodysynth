from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

# neutral labels are labels that are not lateralized
# NNeutralLabels = namedtuple("NNeutralLabels", ("incl_csf", "excl_csf", "wmgm"))
# n_neutral_labels = NNeutralLabels(incl_csf=7, excl_csf=6, wmgm=1)

hemispheres = ("lh", "rh")

# Mapped inputs

MappedInputKeys = namedtuple(
    "MappedInputKeys", ["image", "surface", "initial_vertices", "state", "affine"]
)
mapped_input_keys = MappedInputKeys("image", "surface", "initial_vertices", "state", "affine")


# ---------------------------
# IMAGES
# ---------------------------

# Labeling schemes
LabelingScheme = namedtuple(
    "LabelingScheme",
    (
        "brainseg",
        "brainseg_with_extracerebral",
        "ernie"
    ),
)

# Generation labels
GenerationLabels = namedtuple(
    "GenerationLabels",
    ("n_labels", "label_range", "kmeans", "lesion", "white", "gray", "csf", "pv"),
)
# PV is encoded between 100 and 250 such that
#   100 = 1 lesions
#   150 = 2 white matter
#   200 = 3 gray matter
#   250 = 4 csf
# where
#   RHS = 1 + (LHS-100)/50
#   LHS = 100 + 50(RHS-1).
# For example, 151, 152, ..., 199 encode fractional steps from WM to GM.
PartialVolume = namedtuple("PartialVolume", ("lesion", "white", "gray", "csf"))


# Image name to image filename mapper
Images = namedtuple(
    "ImageFiles",
    (
        "brain_dist_map",
        "brainseg",
        "brainseg_with_extracerebral",
        "generation_labels",
        "generation_labels_dist",
        "mni_reg_x",
        "mni_reg_y",
        "mni_reg_z",
        "mni152_nonlin_forward",
        "mni152_nonlin_backward",
        "lp_dist_map",
        "lw_dist_map",
        "rp_dist_map",
        "rw_dist_map",
        "t1w",
        "t1w_mask",
        "t1w_areg_mni",
        "t1w_areg_mni_mask",
        "t2w",
        "t2w_mask",
        "ct",
        "ct_mask",
        "flair",
        "flair_mask",
    ),
)

@dataclass
class ImageData:
    filename: Path | str
    dtype: torch.dtype
    transform: Callable | None = None
    defacingmask: None | str = None

# distance maps (clipped at [-5, 5] mm) are encoded as
#   distance * 20 + 128
# so
#   28 is -5
#   128 is 0
#   228 is 5.
t_dist_map = lambda x: (x-128.0) / 20.0

# mni_reg* are saved as world coordinates * 100
# stored as int16
t_mni_reg = lambda x: x / 100.0

# brain_dist_map is saved as distance * 10 and saturates at 25.5 mm
# stored as uint8
t_brain_dist_map = lambda x: x / 10.0

# mni152* are saved as *voxel* coordinates * 100
# stored as int16
t_mni152_reg = lambda x: x / 100.0

class ImageSettings:
    def __init__(self):
        self.labeling_scheme = LabelingScheme(
            brainseg=tuple(range(44)),  # brainseg.lut
            brainseg_with_extracerebral=tuple(
                range(57)
            ),  # brainseg_with_extracerebral.lut
            ernie=tuple(range(13)),  # ernie.lut
        )

        self.generation_labels = GenerationLabels(
            n_labels=256,
            label_range=(0, 100),  # the actual labels; the rest are some mix of these
            kmeans=(12, 13, 14, 15, 16, 17, 18, 19), # (12, 13, 14, 15), #
            lesion=1,
            white=2,
            gray=3,
            csf=4,
            pv=PartialVolume(lesion=100, white=150, gray=200, csf=250),
        )

        self.images = Images(
            brain_dist_map=ImageData("brain_dist_map.nii", torch.float, t_brain_dist_map),
            brainseg=ImageData("brainseg.nii", torch.int32),
            brainseg_with_extracerebral=ImageData(
                "brainseg_with_extracerebral.nii", torch.int32,
            ),
            ct=ImageData("CT.nii", torch.float, defacingmask="ct_mask"),
            ct_mask=ImageData("CT.defacingmask.nii", torch.bool),
            flair=ImageData("FLAIR.nii", torch.float, defacingmask="flair_mask"),
            flair_mask=ImageData("FLAIR.defacingmask.nii", torch.float),
            generation_labels=ImageData("generation_labels.nii", torch.int32),
            generation_labels_dist=ImageData("generation_labels_dist.nii", torch.int32),
            lp_dist_map=ImageData("lp_dist_map.nii", torch.float, t_dist_map),
            lw_dist_map=ImageData("lw_dist_map.nii", torch.float, t_dist_map),
            rp_dist_map=ImageData("rp_dist_map.nii", torch.float, t_dist_map),
            rw_dist_map=ImageData("rw_dist_map.nii", torch.float, t_dist_map),
            mni_reg_x=ImageData("mni_reg.x.nii", torch.float, t_mni_reg),
            mni_reg_y=ImageData("mni_reg.y.nii", torch.float, t_mni_reg),
            mni_reg_z=ImageData("mni_reg.z.nii", torch.float, t_mni_reg),
            mni152_nonlin_forward=ImageData(
                "mni152_nonlin_forward.nii", torch.float, t_mni152_reg
            ),
            mni152_nonlin_backward=ImageData(
                "mni152_nonlin_backward.nii", torch.float, t_mni152_reg
            ),
            t1w=ImageData("T1w.nii", torch.float, defacingmask="t1w_mask"),
            t1w_mask=ImageData("T1w.defacingmask.nii", torch.bool),
            # t1w affinely registered to MNI152
            t1w_areg_mni=ImageData("T1w.areg-mni.nii", torch.float, defacingmask="t1w_areg_mni_mask"),
            t1w_areg_mni_mask=ImageData("T1w.areg-mni.defacingmask.nii", torch.bool),
            t2w=ImageData("T2w.nii", torch.float, defacingmask="t2w_mask"),
            t2w_mask=ImageData("T2w.defacingmask.nii", torch.bool),
        )
        self.dist_maps = {"lp_dist_map", "lw_dist_map", "rp_dist_map", "rw_dist_map"}

        self.default_images = [
            "brain_dist_map",
            "brainseg",
            "brainseg_with_extracerebral",
            "generation_labels",
            "generation_labels_dist",
            "mni_reg_x",
            "mni_reg_y",
            "mni_reg_z",
            "mni152_nonlin_backward",
            "mni152_nonlin_forward",
            "lp_dist_map",
            "lw_dist_map",
            "rp_dist_map",
            "rw_dist_map",
            "t1w",
            "t1w_areg_mni",
        ]


# ---------------------------
# SURFACES
# ---------------------------

class SurfaceSettings:
    def __init__(self):
        self.hemispheres: tuple[str, str] = ("lh", "rh")
        self.types: tuple = ("white", "pial")
        self.resolutions: tuple = (0, 1, 2, 3, 4, 5, 6)
        #self.files = SurfaceFiles(self.hemispheres, self.types, self.resolutions)

    @staticmethod
    def get_files(hemispheres, types, resolution, name, extension="pt"):

        hemispheres = [hemispheres] if isinstance(hemispheres, str) else hemispheres

        if isinstance(types, str):
            types = [types]

        remain = [extension]
        remain = remain if name is None else [name] + remain
        remain = remain if resolution is None else [str(resolution)] + remain

        out = {}
        for h in hemispheres:
            if types is None:
                out[h] = ".".join((h, *remain))
            else:
                for t in types:
                    out[(h,t)] = ".".join((h, t, *remain))
        return out


IMAGE = ImageSettings()
SURFACE = SurfaceSettings()
