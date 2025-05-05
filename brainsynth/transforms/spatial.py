from collections.abc import Sequence
import warnings

import torch

from brainsynth.transforms.base import BaseTransform, RandomizableTransform
from brainsynth.transforms import resolution_sampler
from brainsynth.transforms.filters import GaussianSmooth

from brainsynth.transforms.utilities import channel_last, method_recursion_dict


class SpatialCropParameters(BaseTransform):
    def __init__(
        self,
        out_size: torch.Tensor,
        out_center: torch.Tensor,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        # uncorrected slices
        self.raw_slices = self._compute_slice_start_stop(out_center, out_size)

        self.offset = tuple(s[0] for s in self.raw_slices)

    @staticmethod
    def _compute_slice_start_stop(center, size):
        halfsize = 0.5 * (size - 1.0)
        start = torch.ceil(center - halfsize).int()
        stop = torch.floor(center + halfsize).int()
        stop += size - (stop - start)
        return tuple((a.item(), b.item()) for a, b in zip(start, stop))

    def forward(self, in_size: torch.Size):
        # restrict slices to image size (i.e., these slices are sampled in the
        # original image)
        self.slices = tuple(
            slice(max(s[0], 0), min(s[1], ins))
            for s, ins in zip(self.raw_slices, in_size)
        )

        # these commands ...                           adds these dims
        # (assuming x is [C, X, Y, Z] then)

        # torch.nn.functional.pad(x, (0,0,0,0,1,0)) -> x[:,  0,  :,  :]
        # torch.nn.functional.pad(x, (0,0,0,0,0,1)) -> x[:, -1,  :,  :]
        # torch.nn.functional.pad(x, (0,0,1,0,0,0)) -> x[:,  :,  0,  :]
        # torch.nn.functional.pad(x, (0,0,0,1,0,0)) -> x[:,  :, -1,  :]
        # torch.nn.functional.pad(x, (1,0,0,0,0,0)) -> x[:,  :,  :,  0]
        # torch.nn.functional.pad(x, (0,1,0,0,0,0)) -> x[:,  :,  :, -1]

        # therefore

        self.pad = (
            self.slices[2].start - self.raw_slices[2][0],  # inferior
            self.raw_slices[2][1] - self.slices[2].stop,  # superior
            self.slices[1].start - self.raw_slices[1][0],  # posterior
            self.raw_slices[1][1] - self.slices[1].stop,  # anterior
            self.slices[0].start - self.raw_slices[0][0],  # left
            self.raw_slices[0][1] - self.slices[0].stop,  # right
        )

        return dict(offset=self.offset, pad=self.pad, slices=self.slices)


class AdjustAffineToSpatialCrop(BaseTransform):
    def __init__(self, offset, device: None | torch.device = None) -> None:
        super().__init__(device)
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset, dtype=torch.float, device=self.device)
        self.offset = offset

    def forward(self, affine: torch.Tensor):
        out = affine.clone()
        out[:3,3] += out[:3,:3] @ self.offset
        return out


class SpatialCrop(BaseTransform):
    def __init__(
        self,
        size,
        slices=None,
        start=None,
        stop=None,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.size = size  # to validate input
        if slices is None:
            assert (
                start is not None and stop is not None
            ), "`start` and `stop` are required if `slices` is None."
            self.slices = tuple(slice(int(i), int(j)) for i, j in zip(start, stop))
        else:
            self.slices = slices

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert image.size()[-3:] == torch.Size(
            self.size
        ), f"Invalid image size {image.size()[-3:]} expected {torch.Size(self.size)}"
        return image[..., *self.slices]


class PadTransform(BaseTransform):
    def __init__(
        self,
        pad: Sequence[int],
        pad_kwargs: dict | None = None,
        device: None | torch.device = None,
    ) -> None:
        """Construct the amount of padding needed to achieve an image of
        `out_size` centered on `out_center` in an image of `in_size`. E.g.,

            in_size = (176, 256, 256)
            out_size = (192, 192, 192)
            out_center = (87.5000, 127.5000, 127.5000)

        will result in the following amount of padding

            (0, 0, 0, 0, 8, 8)

        To ensure that `out_size` is achieved, apply PadTransform afterwards.
        """
        super().__init__(device)
        self.pad = pad
        self.pad_kwargs = pad_kwargs or {}

        # these commands ...                           adds these dims
        # (assuming x is [C, X, Y, Z] then)

        # torch.nn.functional.pad(x, (0,0,0,0,1,0)) -> x[:,  0,  :,  :]
        # torch.nn.functional.pad(x, (0,0,0,0,0,1)) -> x[:, -1,  :,  :]
        # torch.nn.functional.pad(x, (0,0,1,0,0,0)) -> x[:,  :,  0,  :]
        # torch.nn.functional.pad(x, (0,0,0,1,0,0)) -> x[:,  :, -1,  :]
        # torch.nn.functional.pad(x, (1,0,0,0,0,0)) -> x[:,  :,  :,  0]
        # torch.nn.functional.pad(x, (0,1,0,0,0,0)) -> x[:,  :,  :, -1]

    def forward(self, image):
        # assert image.size()[-3:] == self.size, f"Padding was calculated for an image of size {self.size}; got image of size {image.size()[-3:]}"
        return torch.nn.functional.pad(image, self.pad, **self.pad_kwargs)


class TranslationTransform(BaseTransform):
    def __init__(
        self,
        translation: torch.Tensor,
        invert: bool = False,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.invert = invert
        self.translation = torch.as_tensor(translation).to(self.device)

    def forward(self, x: torch.Tensor):
        if self.invert:
            return x - self.translation
        else:
            return x + self.translation


class RandTranslationTransform(RandomizableTransform):
    def __init__(
        self,
        x_range: tuple | list[float],
        y_range: tuple | list[float],
        z_range: tuple | list[float],
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def rand_in_range(self, r: tuple | list[float]):
        return r[0] + torch.rand(1, device=self.device) * (r[1] - r[0])

    def forward(self, x: torch.Tensor):
        if self.should_apply_transform():
            t = torch.cat(
                (
                    self.rand_in_range(self.x_range),
                    self.rand_in_range(self.y_range),
                    self.rand_in_range(self.z_range),
                )
            )
            return x + t
        else:
            return x


class RandLinearTransform(RandomizableTransform):
    def __init__(
        self,
        max_rotation,
        max_scale,
        max_shear,
        relative_to_input: bool = False,
        device: None | torch.device = None,
        prob: float = 1.0,
    ):
        """_summary_

        Parameters
        ----------
        max_rotation : _type_
            _description_
        max_scale : _type_
            _description_
        max_shear : _type_
            _description_
        relative_to_input : bool, optional
            Apply the transformations relative to the input grid, i.e., in a
            coordinate system centered on the input grid. If true, this will
            center the grid, apply then transformation, and undo the centering.
            (default = False).
        device : None | torch.device, optional
            _description_, by default None
        prob : float, optional
            _description_, by default 1.0
        """
        super().__init__(prob, device)

        if not self.should_apply_transform():
            self.trans = torch.eye(3, device=self.device)
            self.trans_inv = self.trans
            self.scale = 1.0
            return

        self.max_scale = max_scale
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.relative_to_input = relative_to_input

        rotations = (
            (2 * self.max_rotation * torch.rand(3) - self.max_rotation)
            / 180.0
            * torch.pi
        )
        shears = 2 * self.max_shear * torch.rand(3) - self.max_shear
        scalings = 1 + (2 * self.max_scale * torch.rand(3) - self.max_scale)
        # we divide distance maps by this, not perfect, but better than nothing
        scaling_factor_distances = torch.prod(scalings) ** 0.33333333333

        self.trans = self.generate_transform(rotations, shears, scalings)
        self.trans_inv = self.inverse()
        self.scale = scaling_factor_distances

    def generate_transform(
        self,
        rot: torch.Tensor,
        shear: torch.Tensor,
        scale: torch.Tensor,
    ):
        with torch.device(self.device):
            cosR = torch.cos(rot)
            sinR = torch.sin(rot)

            # rotation
            Rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, cosR[0], -sinR[0]],
                    [0, sinR[0], cosR[0]],
                ]
            )
            Ry = torch.tensor(
                [
                    [cosR[1], 0, sinR[1]],
                    [0, 1, 0],
                    [-sinR[1], 0, cosR[1]],
                ]
            )
            Rz = torch.tensor(
                [
                    [cosR[2], -sinR[2], 0],
                    [sinR[2], cosR[2], 0],
                    [0, 0, 1],
                ]
            )

            # shear
            SHx = torch.tensor([[1, 0, 0], [shear[1], 1, 0], [shear[2], 0, 1]])
            SHy = torch.tensor([[1, shear[0], 0], [0, 1, 0], [0, shear[2], 1]])
            SHz = torch.tensor([[1, 0, shear[0]], [0, 1, shear[1]], [0, 0, 1]])

            # collect
            A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
            A[0] *= scale[0]
            A[1] *= scale[1]
            A[2] *= scale[2]

        return A

    def inverse(self):
        return torch.linalg.inv(self.trans)

    def forward(self, grid, inverse: bool = False):
        """`grid` has coordinates in the last dimension!"""
        if self.should_apply_transform():
            A = self.trans_inv if inverse else self.trans
            if self.relative_to_input:
                gm = grid.mean(tuple(range(grid.ndim - 1)))
                return ((grid - gm) @ A.T) + gm
            else:
                return grid @ A.T
        else:
            return grid



class ScaleAndSquare(BaseTransform):
    def __init__(
        self,
        grid: torch.Tensor,
        out_size: torch.Size | torch.Tensor,
        n_steps: int = 8,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.grid = grid
        self.out_size = out_size
        assert n_steps >= 0
        self.n_steps = n_steps
        self.step_size = 1.0 / (2.0**self.n_steps)

    def _apply(self, field, n_steps: int):
        """Scale and square for matrix exponentiation.

        Parameters
        ----------
        field : _type_
            _description_
        n_steps : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        References
        ----------
        Ashburner (2007). A fast diffeomorphic image registration algorithm.
        Arsigny (2006). A Log-Euclidean Framework for Statistics on
            Diffeomorphisms.
        """
        if n_steps == 0:
            return field
        elif n_steps > 0:
            resample = GridSample(self.grid + channel_last(field), self.out_size)
            field = field + resample(field)
            return self._apply(field, n_steps - 1)
        else:
            raise ValueError(
                f"`n_steps` must be greater than or equal to zero (got {n_steps})"
            )

    def forward(self, field):
        return self._apply(field * self.step_size, self.n_steps)


class RandNonlinearTransform(RandomizableTransform):
    def __init__(
        self,
        out_size,
        scale_min: float,
        scale_max: float,
        std_max: float,
        exponentiate_field: bool = False,
        scale_and_square_steps: int = 8,
        grid: None | torch.Tensor = None,
        photo_mode: bool = False,
        device: None | torch.device = None,
        prob: float = 1.0,
    ):
        """
        To be consistent with torch convention that the channel dimension is
        before the spatial dimensions, we create the deformation field such
        that the deformations are [C,W,H,D] although when applying it in
        `compute_deformed_grid` we use it as [W,H,D,C].

        compute_backward_field
            Integrate SVF.
        n_steps
            Number of integration steps for

        """
        super().__init__(prob, device)

        self.out_size = out_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_max = std_max
        self.grid = grid
        self.exponentiate_field = exponentiate_field
        self.scale_and_square_steps = scale_and_square_steps

        if not self.should_apply_transform():
            self.trans = {}
            return

        nonlin_scale = scale_min + torch.rand(1, device=self.device) * (
            scale_max - scale_min
        )
        size_F_small = torch.round(nonlin_scale * out_size).to(torch.int)

        if photo_mode:
            size_F_small[1] = torch.round(out_size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = std_max * torch.rand(1, device=self.device)

        # Channel first (as is normal)
        V = nonlin_std * torch.randn([3, *size_F_small], device=self.device)
        V = Resize(out_size)(V)

        if photo_mode:
            # no deformation in AP direction
            V[1] = 0

        # this is slow on the CPU
        if exponentiate_field:
            self.sas = ScaleAndSquare(self.grid, self.out_size, scale_and_square_steps)

            # forward integration
            # U = V * steplength
            # U = self.scale_and_square(U, self.scale_and_square_steps)
            U = self.sas(V)

            # backward integration
            # U_inv = -V * steplength
            # U_inv = self.scale_and_square(U_inv, self.scale_and_square_steps)
            U_inv = self.sas(-V)

            self.trans = dict(forward=U, backward=U_inv)
        else:
            self.trans = dict(forward=V)

    def forward(self, grid, direction="forward"):
        if self.should_apply_transform():
            match direction:
                case "forward":
                    return grid + channel_last(self.trans[direction])
                case "backward":
                    sampler = GridSample(grid, self.out_size)
                    return grid + sampler(self.trans[direction]).squeeze().T
        else:
            return grid


"""
# NOTE - in case we want to implement the affine sub to sub transform as well

A1 = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine_forward.txt")).float()
A2 = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine_forward.txt")).float()
A1B = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine_backward.txt")).float()
A2B = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine_backward.txt")).float()

S1A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.T1w.nii").affine).float()
S1AB = torch.linalg.inv(S1A)
S2A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s2}.T1w.nii").affine).float()
S2AB = torch.linalg.inv(S2A)

# SUB1 > SUB2

grid = torch.stack(torch.meshgrid([torch.arange(i) for i in s2_size], indexing="ij")).float()
grid = grid.permute((1,2,3,0))
Ax = S1AB @ A1 @ A2B @ S2A

# applying `Ax` to `grid` from right to left we go:
#
#   sub2-vox > sub2-ras > mni-ras > sub1-ras > sub1-vox
#
# that is, we deform grid from sub2 to sub1 space - then we sample thereby
# pulling values from S1 to S2

grid = grid @ Ax[:3,:3].T + Ax[:3,3]

sampler = GridSample(grid, s1_size)
out11 = sampler(s1_t1)
nib.Nifti1Image(out11[0].numpy(), S2A).to_filename(d / f"affine_{s1}-to-{s2}.nii")

# SUB2 > SUB1

grid = torch.stack(torch.meshgrid([torch.arange(i) for i in s1_size], indexing="ij")).float()
grid = grid.permute((1,2,3,0))
Ay = S2AB @ A2 @ A1B @ S1A
grid = grid @ Ay[:3,:3].T + Ay[:3,3]

sampler = GridSample(grid, s2_size)
out11 = sampler(s2_t1)
nib.Nifti1Image(out11[0].numpy(), S2A).to_filename(d / f"affine_{s2}-to-{s1}.nii")

"""


class XSubWarpImage(BaseTransform):
    def __init__(
        self,
        s1_deform: torch.Tensor,
        s2_deform_backward: torch.Tensor,
        s1_size: torch.Tensor | torch.Size,
        device: None | torch.device = None,
    ) -> None:
        """Warp an image from subject 1 (voxel) space to subject 2 (voxel)
        space. The mapping is achieved by composing the supplied deformation
        fields, i.e.,

                      s1_deform            s2_deform_backward
            S1[vox]     <---     MNI[vox]         <---          S2[vox]

        such that, when applied to an image in subject 1 space, they will pull
        the values into the space of subject 2.

        Parameters
        ----------
        s1_deform: torch.Tensor
            Forward deformation field of subject 1. The forward deformation
            maps each voxel in MNI152 space to the corresponding coordinate in
            subject 1 *voxel* space. Forward deformations has MNI affine and
            MNI size.
        s2_deform_backward: torch.Tensor
            Backward deformation field of subject 2. The backward deformation
            maps each voxel in subject 2 space to the corresponding coordinate
            in MNI *voxel* space. Backward deformations has subject affine and
            subject size.
        s1_size: torch.Tensor | torch.Size
            The spatial dimensions of subject 1 image space.
        """
        # images are (C,W,H,D) !!

        super().__init__(device)
        mni_image_shape = s1_deform.shape[-3:]

        # MNI voxel coordinates
        sampler = GridSample(channel_last(s2_deform_backward), mni_image_shape)
        # S1 voxel coordinates
        grid = sampler(s1_deform)

        # Invalid coordinates (S2 coordinates mapped outside MNI FOV) gets
        # mapped to zeros (or border). What we really wanted, but which is not
        # yet supported, would be something like
        #
        #   grid_sample(..., padding_mode="constant", value=1e6)
        #
        # Therefore, we identify the invalid coordinates and map them to
        # something outside of S1's FOV which means they will eventually become
        # zero.

        # norm_grid is (X,Y,Z,C,N) (N = batch dim)
        oob = torch.any(sampler.norm_grid.squeeze().abs() > 1.0, dim=-1)
        grid[:, oob] = 1e6

        # The grid now contains the voxel coordinates in S1 corresponding to
        # each voxel in S2, i.e., we have a S1[vox] to S2[vox] mapping.
        self.sampler = GridSample(channel_last(grid), s1_size)

    def forward(self, image):
        return self.sampler(image)


class XSubWarpSurface_v1(BaseTransform):
    def __init__(
        self,
        s1_deform_backward: torch.Tensor,
        s2_deform: torch.Tensor,
        device: None | torch.device = None,
    ) -> None:
        """Warp a surface from subject 1 (voxel) space to subject 2 (voxel)
        space. The mapping is achieved by composing the supplied deformation
        fields, i.e.,

                      s2_deform            s1_deform_backward
            S2[vox]     <---     MNI[vox]         <---          S1[vox]

        such that, when applied to a surface in subject 1 space, they will push
        the vertices into the space of subject 2. Note that this is the inverse
        operation of `XSubWarpImage`: XSubWarpImage deforms the image
        grid of S2 to S1 space and pulls the values. XSubWarpSurface pushes
        coordinates in S1 space to S2 space.

        Parameters
        ----------
        s1_deform_backward :
            Backward deformation field of subject 1. The backward deformation
            maps each voxel in subject 1 space to the corresponding coordinate
            in MNI *voxel* space. Backward deformations has subject affine and
            subject size.
        s2_deform :
            Forward deformation field of subject 2. The forward deformation
            maps each voxel in MNI152 space to the corresponding coordinate in
            subject 2 *voxel* space. Forward deformations has MNI affine and
            MNI size.
        """
        super().__init__(device)
        mni_image_shape = s2_deform.shape[-3:]

        # MNI voxel coordinates
        sampler = GridSample(channel_last(s1_deform_backward), mni_image_shape)
        # The field now contains the voxel coordinates in S2 space corresponding
        # to each voxel in S1, i.e., we have a S2[vox] to S1[vox] mapping.
        # S2 voxel coordinates
        self.deformation_field = sampler(s2_deform, padding_mode="border")
        self.size = self.deformation_field.size()[-3:]

    def forward(self, surface):
        sampler = GridSample(surface, self.size)
        return sampler(self.deformation_field, padding_mode="border").squeeze().T


class XSubWarpSurface_v2(BaseTransform):
    def __init__(
        self,
        s1_deform_backward: torch.Tensor,
        s2_deform: torch.Tensor,
        device: None | torch.device = None,
    ) -> None:
        """Warp a surface from subject 1 (voxel) space to subject 2 (voxel)
        space. The mapping is achieved by composing the supplied deformation
        fields, i.e.,

                      s2_deform            s1_deform_backward
            S2[vox]     <---     MNI[vox]         <---          S1[vox]

        such that, when applied to a surface in subject 1 space, they will push
        the vertices into the space of subject 2. Note that this is the inverse
        operation of `XSubWarpImage`: XSubWarpImage deforms the image
        grid of S2 to S1 space and pulls the values. XSubWarpSurface pushes
        coordinates in S1 space to S2 space.

        Parameters
        ----------
        s1_deform_backward :
            Backward deformation field of subject 1. The backward deformation
            maps each voxel in subject 1 space to the corresponding coordinate
            in MNI *voxel* space. Backward deformations has subject affine and
            subject size.
        s2_deform :
            Forward deformation field of subject 2. The forward deformation
            maps each voxel in MNI152 space to the corresponding coordinate in
            subject 2 *voxel* space. Forward deformations has MNI affine and
            MNI size.
        """
        super().__init__(device)
        self.s1_deform_backward = s1_deform_backward
        self.s2_deform = s2_deform

    def forward(self, surface):
        self2mni = GridSample(surface, self.s1_deform_backward.size()[-3:])
        surface_mni = self2mni(self.s1_deform_backward, padding_mode="border").T
        mni2other = GridSample(surface_mni, self.s2_deform.size()[-3:])
        return mni2other(self.s2_deform, padding_mode="border").T


class CheckCoordsInside(BaseTransform):
    def __init__(
        self, size, raise_on_outsize: bool = False, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.size = torch.as_tensor(size, device=self.device)
        self.raise_on_outsize = raise_on_outsize
        self.msg_header = "Cortical surface is partly outside of FOV."

    def forward(self, coords):
        msg = [self.msg_header]
        outside = False
        if torch.any((smin := coords.amin(0)) < 0):
            msg.append(f"  {torch.round(smin, decimals=2)} < (0, 0, 0)")
            outside = True
        if torch.any((smax := coords.amax(0)) > self.size):
            msg.append(f"  {torch.round(smax, decimals=2)} > {self.size}")
            outside = True

        if outside:
            msg = "\n".join(msg)
            if self.raise_on_outsize:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        return coords


# class SurfaceDeformation(BaseTransform):
#     def __init__(
#         self,
#         linear_trans: RandLinearTransform,
#         nonlinear_trans: RandNonlinearTransform,
#         # nonlinear_field: ,
#         # out_size: ,
#         out_center: torch.Tensor,
#         device: None | torch.device = None,
#     ) -> None:
#         super().__init__(device)
#         self.linear_trans = linear_trans
#         self.nonlinear_trans = nonlinear_trans
#         self.out_size = self.nonlinear_trans.out_size
#         # the center of the final FOV
#         self.out_center = out_center
#         # center in the output image
#         # self.center_in_out_image = GridCentering(self.out_size, device=self.device)

#     def deform_surface(self, s):
#         # surface_center = 0.5 * (s.amax(0) - s.amin(0))

#         # center around (0,0,0)
#         # s -= surface_center
#         s -= self.out_center

#         s = self.linear_trans(s, inverse=True)

#         # place appropriately in output FOV
#         # s = self.center_in_out_image(s + self.out_center + surface_center)
#         s = s + center_from_shape(self.out_size)

#         if self.nonlinear_trans.should_apply_transform():
#             sample = GridSample(s, self.nonlinear_trans.out_size)
#             s += sample(self.nonlinear_trans.trans["backward"])

#         if torch.any((smin := s.amin(0)) < 0) or torch.any(
#             (smax := s.amax(0)) > self.out_size
#         ):
#             smax = s.amax(0) if "smax" not in vars() else smax
#             warnings.warn(
#                 (
#                     "Cortical surface is partly outside of FOV.\n"
#                     f"  {tuple(i.item() for i in smin)} > (0,0,0)"
#                     f"  {tuple(i.item() for i in smax)} < {self.out_size}."
#                 )
#             )

#         return s

#     def forward(self, surface: dict | torch.Tensor):
#         if isinstance(surface, torch.Tensor):
#             return self.deform_surface(surface)
#         else:
#             return {k: self.forward(v) for k, v in surface.items()}


class Resize(BaseTransform):
    def __init__(self, size: torch.Tensor | tuple | list, mode="trilinear"):
        super().__init__()
        self.size = tuple(size)
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image[None]
        if image.ndim == 4:
            image = image[None]

        return torch.nn.functional.interpolate(
            input=image,
            size=self.size,
            mode=self.mode,
            align_corners=True,
        ).squeeze(0)


# class RandFlipImage(RandomizableTransform):
#     def __init__(self, prob: float = 1):
#         super().__init__(prob)

#     def forward(self):
#         raise NotImplementedError


# class RandLRFlipSurface(RandomizableTransform):
#     def __init__(self, size, prob: float = 1.0):
#         super().__init__(prob)
#         self.flip_hemi = dict(zip(constants.HEMISPHERES, constants.HEMISPHERES[::-1]))
#         self.dim = 0
#         self.width = size[self.dim] # assume RAS : [W,H,D]

#     def _flip_surface_coords(self, s):
#         s[:, self.dim] *= -1.0
#         s[:, self.dim] += self.width - 1.0

#     def forward(self, surfaces):
#         if not self.should_apply_transform():
#             return surfaces

#         for hemi, surfs in surfaces.items():
#             if isinstance(surfs, dict):
#                 for s in surfs:
#                     self._flip_surface_coords(surfaces[hemi][s])
#             else:
#                 self._flip_surface_coords(surfs)

#         # NOTE
#         # swap label (e.g., lh -> rh) when flipping image. Not sure if
#         # this is the best thing to do but at least ensures some kind
#         # of consistency: when flipping then faces need to be reordered
#         # which is also the case for rh compared to lh.
#         return {self.flip_hemi[h]: v for h, v in surfaces.items()}


class SpatialSize(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: torch.Tensor):
        return torch.as_tensor(image.shape[-3:], device=image.device)


class Grid(BaseTransform):
    def __init__(self, size, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.size = size

    def forward(self):
        return torch.stack(
            torch.meshgrid(
                [
                    torch.arange(s, dtype=torch.float, device=self.device)
                    for s in self.size
                ],
                indexing="ij",
            ),
            dim=-1,
        )


class GridCentering(BaseTransform):
    def __init__(
        self, size, align_corners: bool = True, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.align_corners = align_corners
        self.size = torch.as_tensor(size, device=self.device)
        if self.align_corners:
            self.center = 0.5 * (self.size - 1.0)
        else:
            self.center = 0.5 * self.size

    def forward(self, grid: torch.Tensor):
        return grid - self.center


class GridNormalization(GridCentering):
    def __init__(
        self, size, align_corners: bool = True, device: None | torch.device = None
    ) -> None:
        super().__init__(size, align_corners, device)

    def forward(self, grid: torch.Tensor):
        return super().forward(grid) / self.center


class GridSample(BaseTransform):
    def __init__(
        self,
        grid: torch.Tensor,
        size: None | torch.Tensor | torch.Size = None,
        assume_normalized: bool = False,
        # device: None | torch.device = None,
    ):
        """_summary_

        Parameters
        ----------
        grid : torch.Tensor
            Grid of shape ([N, ]V, 3) or ([N, ]W, H, D, 3).
        size : None | torch.Tensor, optional
           Size of the image to be interpolated from (i.e., the size of the
           image used when calling GridSample). The grid is normalized by this
           shape.
        assume_normalized : bool, optional
            Assume that `grid` is already normalized in which case `size` is
            not required (default = False).
        """
        super().__init__()

        # size
        self.size = size
        if not assume_normalized:
            assert (
                size is not None
            ), "`size` must be provided when `assume_normalized = False`"
            assert len(size) == 3
        self.assume_normalized = assume_normalized

        # grid
        self.grid = grid
        self.grid_ndim = len(self.grid.shape)
        assert self.grid_ndim in {2, 3, 4, 5}

        # normalized grid
        if self.assume_normalized:
            self.norm_grid = grid
        else:
            self.norm_grid = GridNormalization(self.size, device=grid.device)(grid)

        match self.grid_ndim:
            case 2:
                self.norm_grid = self.norm_grid[None, :, None, None]  # 1,W,1,1,3
            case 3:
                # includes batch dim
                self.norm_grid = self.norm_grid[:, :, None, None]  # 1,W,1,1,3
            case 4:
                self.norm_grid = self.norm_grid[None]  # 1,W,H,D,3
            case 5:
                # includes batch dim so assume it is OK
                pass  # N,W,H,D,3

    def forward(self, image, padding_mode="border"):
        """_summary_

        image : (1, 3, ...) (2, 3, ...)
        grid: (..., 3), (1, ..., 3), (2, ..., 3)
        grid: (V, 3), (2, V, 3)

        Parameters
        ----------
        image : _type_
            Image to interpolate from. The dimensions are ([N, ]C, H, W, D)
            where (H, W, D) should match the spatial size provided at
            initialization (if any).

        Returns
        -------
        _type_
            _description_
        """
        # image : ([N, ]C, W, H, D)
        # grid  : (V, 3) or (WW, HH, DD, 3)
        if self.grid is None:
            return image

        if not self.assume_normalized:
            assert (
                (image_size := tuple(image.size()[-3:])) == tuple(self.size)
            ), f"Expected image with spatial dimensions {self.size} but got {image_size}."

        assert (image_ndim := len(image.shape)) in {3, 4, 5}

        # image
        # [n,c,]x,y,z -> n,c,z,y,x (!)
        if image_ndim == 3:
            image = image[None, None]
        elif image_ndim == 4:
            image = image[None]
        elif image_ndim == 5:
            pass  # already has batch dim
        image = image.transpose(2, 4)  # NOTE xyz -> zyx (!)

        # NOTE `sample` has original xyz ordering! (N,C,W,H,D)
        if image.is_floating_point():
            sample = torch.nn.functional.grid_sample(
                image,
                self.norm_grid,
                align_corners=True,
                mode="bilinear",
                padding_mode=padding_mode,
            )
        else:
            # Cast to float and back since grid_sample doesn't support int (at
            # least not on cuda)
            sample = torch.nn.functional.grid_sample(
                image.float(),
                self.norm_grid,
                align_corners=True,
                mode="nearest",
                padding_mode=padding_mode,
            ).to(image.dtype)

        if self.grid_ndim in {2, 3}:
            sample.squeeze_((3, 4))  # remove H,D dims
        sample.squeeze_(0)  # remove batch dim if not present

        return sample


def _reduce_bbox(bbox: dict[str, torch.Tensor]):
    """Reduce to the total bounding box."""
    it = tuple(bbox.values())
    agg = torch.empty_like(it[0])
    assert agg.shape == (2, 3)
    agg.copy_(it[0])
    for v in it[1:]:
        torch.minimum(agg[0], v[0], out=agg[0])
        torch.maximum(agg[1], v[1], out=agg[1])
    return agg


class RestrictBoundingBox(BaseTransform):
    def __init__(
        self,
        size: torch.Size | torch.Tensor | tuple,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.zeros = torch.zeros(3, device=self.device)
        self.size = torch.as_tensor(size, device=self.device)

    def forward(self, bbox: torch.Tensor):
        torch.maximum(bbox[0], self.zeros, out=bbox[0])
        torch.minimum(bbox[1], self.size, out=bbox[1])
        return bbox


class SurfaceBoundingBox(BaseTransform):
    def __init__(
        self,
        floor_and_ceil: bool = False,
        pad: float = 0.0,
        reduce: bool = False,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.floor_and_ceil = floor_and_ceil
        self.pad = pad
        self.reduce = reduce

    @method_recursion_dict
    def compute_bbox(self, x: torch.Tensor):
        """Bounding box of 2D array with (points, coordinates)."""
        return torch.stack((x.amin(0), x.amax(0)))

    @method_recursion_dict
    def _floor_and_ceil(self, bbox: torch.Tensor):
        torch.floor(bbox[0], out=bbox[0])
        torch.ceil(bbox[1], out=bbox[1])
        return bbox

    @method_recursion_dict
    def _pad(self, bbox: torch.Tensor):
        bbox[0] -= self.pad
        bbox[1] += self.pad
        return bbox

    def forward(self, surface: dict[str, torch.Tensor]):
        bbox = self.compute_bbox(surface)
        bbox = _reduce_bbox(bbox) if self.reduce else bbox
        bbox = self._floor_and_ceil(bbox) if self.floor_and_ceil else bbox
        bbox = self._pad(bbox) if self.pad > 0 else bbox
        return bbox


class BoundingBoxSize(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, bbox):
        return bbox[1] - bbox[0]


class BoundingBoxCorner(BaseTransform):
    def __init__(self, corner: str, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.corner_index = self.corner_to_index(corner)

    @staticmethod
    def corner_to_index(corner):
        match corner:
            case "lower left":
                return 0
            case "upper right":
                return 1
            case _:
                raise ValueError(f"Invalid corner specification `{corner}`")

    def forward(self, bbox):
        if isinstance(bbox, dict):
            return {k: self.forward(v) for k, v in bbox.items()}
        else:
            return bbox[self.corner_index]


class CenterFromString(BaseTransform):
    def __init__(
        self,
        size: torch.Tensor,
        surface_bbox: dict[str, torch.Tensor] | None = None,
        align_corners: bool = True,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.size = size
        self.surface_bbox = surface_bbox
        self.align_corners = align_corners
        if self.surface_bbox is None:
            self.valid_centers = {"image", "random"}
        else:
            self.valid_centers = None

    def forward(self, center):
        if self.valid_centers is not None:
            assert center in self.valid_centers

        match center:
            case "brain":
                res = _reduce_bbox(self.surface_bbox).mean(0)
            case "image":
                res = 0.5 * (self.size - 1.0)
            case "lh":
                res = self.surface_bbox["lh"].mean(0)
            case "random":
                # get the property of the tensor "self"
                # print("Image size",self.size) #this is the size of the image
                res = torch.rand(3, device=self.device) * self.size
                res = torch.clamp(res, min=torch.tensor(0, device=self.device), max=self.size - 1)
            case "rh":
                res = self.surface_bbox["rh"].mean(0)

        # When align_corners = True (in grid sampling), this, together with the
        # assertion that the FOV size is divisible by 2, ensures that we sample
        # exactly in the center of a voxel when there is no linear or nonlinear
        # deformation (we could get almost identical results with cropping).
        # This is important, especially for label images, as otherwise we are
        # always sampling exactly in between two voxels which will result in
        # label images that look downsampled. (Also, the exact center is not
        # important.)
        return res.floor() + 0.5 if self.align_corners else res


class RandResolution(RandomizableTransform):
    """Simulate a"""

    def __init__(
        self,
        out_size: torch.Tensor | tuple,
        in_res: torch.Tensor | tuple,
        res_sampler: str = "ResolutionSamplerDefault",
        photo_mode: bool = False,
        photo_spacing: float | None = None,
        photo_thickness: float = 0.001,
        photo_res_sampler: str = "ResolutionSamplerPhoto",
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)

        self.in_res = torch.as_tensor(in_res, device=self.device)
        self.out_size = torch.as_tensor(out_size, device=self.device)
        self.photo_mode = photo_mode
        self.photo_spacing = photo_spacing
        self.resize = Resize(out_size)

        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        if self.should_apply_transform():
            if self.photo_mode:
                out_res, thickness = getattr(resolution_sampler, photo_res_sampler)(
                    self.in_res,
                    photo_spacing,
                    photo_thickness,
                    self.device,
                )
            else:
                out_res, thickness = getattr(resolution_sampler, res_sampler)(
                    self.device
                )()

            sigma = (
                (0.85 + 0.3 * torch.rand(1, device=self.device))
                * torch.log(torch.tensor([5.0], device=self.device))
                / torch.pi
                * thickness
                / self.in_res
            )
            # no blur if thickness is equal to the resolution of the training
            # data
            sigma[thickness <= self.in_res] = 0.0
            self.out_res = out_res
            self.sigma = tuple(s.item() for s in sigma)
            self.gaussian_blur = GaussianSmooth(self.sigma, device=self.device)
        else:
            self.out_res = in_res
            self.sigma = None
            self.gaussian_blur = None

    def compute_voxel_indices(self, image, ds_size):
        nones = [None] * len(image.shape[1:])
        sv = tuple(
            torch.arange(s, dtype=torch.float, device=self.device) for s in ds_size
        )
        ds_factor = self.out_size/ds_size

        small_voxel = torch.stack(torch.meshgrid(*sv, indexing="ij"))
        out_voxel = self.resize(small_voxel)

        # Distance from each *upsampled* voxel to the closest voxel in the
        # *original* (downsampled) image
        # (along each dim)
        dist_dim = (out_voxel - out_voxel.round()).abs() * ds_factor[:, *nones]
        # (overall)
        # dist_euc = dist_dim.pow(2).sum(0,keepdims=True).sqrt()

        return torch.cat((image, dist_dim))

        # out_voxel = out_voxel.round()
        # out_voxel = out_voxel / (ds_size[:, *nones] - 1.0)

        # # np.linspace(0,1,176)
        # freq = tuple(
        #     torch.cos(
        #         (upsize-1.0) / (upsize/downsize) * torch.pi * torch.linspace(0.0, 1.0, upsize, device=self.device)
        #         ) for downsize,upsize in zip(ds_size, self.out_size)
        # )
        # freq_grid = torch.stack(torch.meshgrid(*freq, indexing="ij"))
        # return torch.cat((image, freq_grid), dim=0)

        # return torch.cat((image, out_voxel), dim=0)

    def forward(self, image):
        if not self.should_apply_transform():
            return image

        # sanity check
        # assert tuple(self.out_size) == tuple(image.size()[-3:])

        ds_size = torch.round(self.out_size * self.in_res / self.out_res).to(torch.int)

        # blur, downsample, upsample (back to original size)
        image = self.resize(Resize(ds_size)(self.gaussian_blur(image)))

        return image
