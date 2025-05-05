from functools import partial
import torch

from brainsynth.config.synthesizer import SynthesizerConfig
from brainsynth.transforms import *
from brainsynth import resources_dir
from brainsynth.constants import IMAGE


class SynthBuilder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device

    def initialize_spatial_transform(self):
        """Optional."""
        pass

    def build_spatial_transform(self):
        raise NotImplementedError

    def build_intensity_transform(self):
        raise NotImplementedError

    def build_resolution_transform(self):
        raise NotImplementedError

    def build_state(self):
        raise NotImplementedError

    def build_output(self):
        raise NotImplementedError

    def build(self):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
        the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """

        self.initialize_spatial_transform()
        state = self.build_state()

        # The following relies (may rely) on the state
        self.build_spatial_transform()
        self.build_intensity_transform()
        self.build_resolution_transform()

        output = self.build_output()

        return state, output


class PredictionBuilder(SynthBuilder):
    """Process an image called `image` and"""

    def build_state(self):
        return dict(
            in_size=Pipeline(
                SelectImage("image"),
                SpatialSize(),
            ),
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=self.config.align_corners,
                ),
            ),
            crop_params=Pipeline(
                SelectState("in_size"),
                PipelineModule(
                    SpatialCropParameters,
                    self.config.out_size,
                    SelectState("out_center"),
                ),
            ),
        )

    def build_spatial_transform(self):
        self.image_crop = SubPipeline(
            PipelineModule(
                SpatialCrop,
                SelectState("in_size"),
                SelectState("crop_params", "slices"),
            ),
            PipelineModule(
                PadTransform,
                SelectState("crop_params", "pad"),
            ),
        )

        self.surface_translation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("crop_params", "offset"),
                invert=True,
                device=self.device,
            ),
            CheckCoordsInside(self.config.out_size, device=self.device),
        )

        self.affine_adjuster = SubPipeline(
            PipelineModule(
                AdjustAffineToSpatialCrop,
                SelectState("crop_params", "offset"),
                device=self.device,
            ),
        )

    def build_intensity_transform(self):
        self.intensity_normalization = IntensityNormalization()

    def build_output(self):
        return dict(
            image=Pipeline(
                SelectImage("image"),
                self.image_crop,
                self.intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                self.surface_translation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                self.surface_translation,
                skip_on_InputSelectorError=True,
            ),
            affine=Pipeline(SelectAffine("image"), self.affine_adjuster),
        )


class DefaultSynth(SynthBuilder):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

        - linear deformation
        - nonlinear deformation
        - synthesis of an image as the input image for training

        """
        super().__init__(config)

    def initialize_spatial_transform(self):
        self.nonlinear_transform = RandNonlinearTransform(
            self.config.out_size,
            scale_min=0.03,  # 0.10,
            scale_max=0.06,  # 0.15,
            std_max=4.0,
            exponentiate_field=True,
            grid=self.config.grid,
            prob=1.0,
            device=self.device,
        )
        self.linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.2,
            max_shear=0.2,
            prob=1.0,
            device=self.device,
        )

        self.inv_nonlinear_transform = partial(
            self.nonlinear_transform, direction="backward"
        )
        self.inv_linear_transform = partial(self.linear_transform, inverse=True)

    def build_state(self):
        return dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*self.config.photo_spacing_range, device=self.device),
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    SelectState("available_images", 0),
                ),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            extracerebral_mask=Pipeline(
                SelectImage("generation_labels_dist"),
                PipelineModule(
                    MaskFromLabelImage,
                    labels=[0] + list(IMAGE.generation_labels.kmeans),
                    device=self.device,
                ),
                skip_on_InputSelectorError=True,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=self.config.align_corners,
                ),
                # RandTranslationTransform(
                #     x_range=[-10, 10],
                #     y_range=[-10, 10],
                #     z_range=[-10, 10],
                #     prob=0.5,
                #     device=self.device,
                # ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(self.config.centered_grid),
                self.nonlinear_transform,
                self.linear_transform,
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=self.device,
                ),
            ),
        )

    def build_spatial_transform(self):
        self.image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        self.surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=self.device,
            ),
            self.inv_linear_transform,
            TranslationTransform(self.config.center, device=self.device),
            self.inv_nonlinear_transform,
            CheckCoordsInside(self.config.out_size, device=self.device),
        )

    def build_intensity_transform(self):
        self.skullstrip = PipelineModule(
            SwitchTransform,
            IdentityTransform(),
            # skull strip
            PipelineModule(
                RandMaskRemove,
                SelectState("extracerebral_mask"),
            ),
            # salt and pepper noise (e.g., like MP2RAGE)
            PipelineModule(
                RandSaltAndPepperNoise,
                SelectState("extracerebral_mask"),
                scale=255.0,
                device=self.device,
            ),
            prob=(0.8, 0.1, 0.1),
        )
        self.intensity_normalization = IntensityNormalization()

    def build_resolution_transform(self):
        self.resolution_augmentation = PipelineModule(
            RandResolution,
            self.config.out_size,
            self.config.in_res,
            res_sampler="RandClinicalSlice",
            # res_sampler_kwargs = dict(slice_idx=2),
            photo_mode=self.config.photo_mode,
            photo_res_sampler_kwargs=dict(
                spacing=SelectState("photo_spacing"),
                slice_thickness=self.config.photo_thickness,
            ),
            prob=1.0,
            device=self.device,
        )

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage("generation_labels_dist"),
            SynthesizeIntensityImage(
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=self.config.photo_mode,
                device=self.device,
            ),
            RandGammaTransform(prob=0.33),
            self.skullstrip,
            self.image_deformation,  # Transform to output FOV
            RandBiasfield(
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.03,
                std_min=0.0,
                std_max=0.3,
                photo_mode=self.config.photo_mode,
                prob=0.75,
                device=self.device,
            ),
            # Small, local intensity variations
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.10,
            #     scale_max=0.20,
            #     std_min=0.0,
            #     std_max=0.15,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.5,
            #     device=self.device,
            # ),
            # self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_surfaces(self):
        return dict(
            surface=Pipeline(
                SelectSurface(),
                self.surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                self.surface_deformation,
                # Noise on initial vertices
                RandLinearTransform(
                    max_rotation=5.0,
                    max_scale=0.05,
                    max_shear=0.05,
                    relative_to_input=True,
                    prob=0.5,
                    device=self.device,
                ),
                RandTranslationTransform(
                    x_range=[-3, 3],
                    y_range=[-3, 3],
                    z_range=[-3, 3],
                    prob=0.5,
                    device=self.device,
                ),
                skip_on_InputSelectorError=True,
            ),
        )

    def build_images(self):
        return dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.skullstrip,
                self.image_deformation,
                self.intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                self.skullstrip,
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
            # t2w=Pipeline(
            #     SelectImage("t2w"),
            #     self.image_deformation,
            #     self.intensity_normalization,
            #     skip_on_InputSelectorError=True,
            # ),
            # t2w_mask=Pipeline(
            #     SelectImage("t2w_mask"),
            #     self.image_deformation,
            #     skip_on_InputSelectorError=True,
            # ),
            # brainseg=Pipeline(
            #     SelectImage("brainseg"),
            #     self.image_deformation,
            #     OneHotEncoding(self.config.segmentation_num_labels),
            #     skip_on_InputSelectorError=True,
            # ),
            brain_dist_map=Pipeline(
                SelectImage("brain_dist_map"),
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

    def build_output(self):
        return dict(
            image_hires=self.build_image(),
            image=Pipeline(
                SelectOutput("image_hires"),
                self.resolution_augmentation,
            ),
            **self.build_images(),
            **self.build_surfaces(),
        )


class OnlySynth(DefaultSynth):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

            - synthesis of an image as the input image for training
            - resolution augmentation from DefaultSynth

        and thus *no* spatial augmentation (only extracting a particular FOV).
        """
        super().__init__(config)

    def initialize_spatial_transform(self):
        """No spatial transformation."""
        self.linear_transform = IdentityTransform()
        self.nonlinear_transform = IdentityTransform()
        self.inv_linear_transform = IdentityTransform()
        self.inv_nonlinear_transform = IdentityTransform()


class OnlySynthIso(OnlySynth):
    def build_resolution_transform(self):
        self.resolution_augmentation = IdentityTransform()


class OnlySelect(OnlySynth):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

            - selection of an image as the input image for training
            - resolution augmentation from DefaultSynth

        and thus *no* spatial augmentation (only extracting a particular FOV).
        """
        super().__init__(config)

    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )
        return state

    def build_image(self):
        """No intensity augmentation."""

        return Pipeline(
            # Select one of the available (valid) images
            PipelineModule(
                SelectImage,
                PipelineModule(
                    RandomChoice,
                    # equal probability of selecting each image
                    SelectState("selectable_images"),
                ),
            ),
            self.skullstrip,
            self.image_deformation,  # Transform to output FOV
            # self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_images(self):
        # return {}
        return dict(
            brain_dist_map=Pipeline(
                SelectImage("brain_dist_map"),
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t1w=Pipeline(
                SelectImage("t1w"),
                self.skullstrip,
                self.image_deformation,
                self.intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
        )

    def build_output(self):
        return dict(
            image_hires=self.build_image(),
            image=Pipeline(
                SelectOutput("image_hires"),
                self.resolution_augmentation,
            ),
            **self.build_images(),
            **self.build_surfaces(),
        )


class OnlySelectNoSkullStrip(OnlySelect):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

            - selection of an image as the input image for training
            - resolution augmentation from DefaultSynth

        and thus *no* spatial augmentation (only extracting a particular FOV).
        """
        super().__init__(config)

    def build_intensity_transform(self):
        # remove skull-stripping
        self.skullstrip = IdentityTransform()
        self.intensity_normalization = IntensityNormalization()


class OnlySelectNoSkullStripIso(OnlySelectNoSkullStrip):
    def build_resolution_transform(self):
        self.resolution_augmentation = IdentityTransform()


class OnlySelectIso(OnlySelect):
    def build_resolution_transform(self):
        self.resolution_augmentation = IdentityTransform()


class OnlySynthIsoT1w(OnlySynthIso):
    def build_intensity_transform(self):
        self.intensity_normalization = IntensityNormalization()

    def build_image(self):
        prefix = "T1w_HCP20_multivariate_normal"
        return Pipeline(
            SelectImage("generation_labels"),
            SynthesizeFromMultivariateNormal(
                mean_loc=torch.load(
                    resources_dir / "contrast_distributions" / f"{prefix}_mean_loc.pt"
                ),
                mean_scale_tril=torch.load(
                    resources_dir
                    / "contrast_distributions"
                    / f"{prefix}_mean_scale_tril.pt"
                ),
                sigma_loc=torch.load(
                    resources_dir / "contrast_distributions" / f"{prefix}_sigma_loc.pt"
                ),
                sigma_scale_tril=torch.load(
                    resources_dir
                    / "contrast_distributions"
                    / f"{prefix}_sigma_scale_tril.pt"
                ),
                device=self.device,
            ),
            PipelineModule(RandBlendImages, SelectImage("t1w"), prob=0.5),
            self.image_deformation,  # Transform to output FOV
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.01,
            #     scale_max=0.03,
            #     std_min=0.0,
            #     std_max=0.3,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.75,
            #     device=self.device,
            # ),
            self.resolution_augmentation(),
            self.intensity_normalization,
        )


class CropSynth(SynthBuilder):
    """Process an image called `image` and"""

    def build_state(self):
        return dict(
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                unpack_inputs=False,
            ),
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    SelectState("available_images", 0),
                ),
                SpatialSize(),
            ),
            extracerebral_mask=Pipeline(
                SelectImage("generation_labels_dist"),
                PipelineModule(
                    MaskFromLabelImage,
                    labels=[0] + list(IMAGE.generation_labels.kmeans),
                    device=self.device,
                ),
                skip_on_InputSelectorError=True,
            ),
            out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    align_corners=self.config.align_corners,
                ),
            ),
            crop_params=Pipeline(
                SelectState("in_size"),
                PipelineModule(
                    SpatialCropParameters,
                    self.config.out_size,
                    SelectState("out_center"),
                ),
            ),
        )

    def build_intensity_transform(self):
        self.skullstrip = PipelineModule(
            SwitchTransform,
            IdentityTransform(),
            # skull strip
            PipelineModule(
                RandMaskRemove,
                SelectState("extracerebral_mask"),
            ),
            # salt and pepper noise (e.g., like MP2RAGE)
            PipelineModule(
                RandSaltAndPepperNoise,
                SelectState("extracerebral_mask"),
                scale=255.0,
                device=self.device,
            ),
            prob=(0.8, 0.1, 0.1),
        )
        self.intensity_normalization = IntensityNormalization()

    def build_resolution_transform(self):
        self.resolution_augmentation = IdentityTransform()
        # PipelineModule(
        #     RandResolution,
        #     self.config.out_size,
        #     self.config.in_res,
        #     res_sampler = "RandClinicalSlice",
        #     # res_sampler_kwargs = dict(slice_idx=2),
        #     photo_mode=self.config.photo_mode,
        #     photo_res_sampler_kwargs = dict(
        #         spacing=SelectState("photo_spacing"),
        #         slice_thickness=self.config.photo_thickness,
        #     ),
        #     prob=1.0,
        #     device=self.device,
        # )

    def build_spatial_transform(self):
        self.image_crop = SubPipeline(
            PipelineModule(
                SpatialCrop,
                SelectState("in_size"),
                SelectState("crop_params", "slices"),
            ),
            PipelineModule(
                PadTransform,
                SelectState("crop_params", "pad"),
            ),
        )

        self.surface_translation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("crop_params", "offset"),
                invert=True,
                device=self.device,
            ),
            CheckCoordsInside(self.config.out_size, device=self.device),
        )

        self.affine_adjuster = SubPipeline(
            PipelineModule(
                AdjustAffineToSpatialCrop,
                SelectState("crop_params", "offset"),
                device=self.device,
            ),
        )

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage("generation_labels_dist"),
            SynthesizeIntensityImage(
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=self.config.photo_mode,
                device=self.device,
            ),
            RandGammaTransform(prob=0.33),
            self.skullstrip,
            self.image_crop,  # Transform to output FOV
            RandBiasfield(
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.03,
                std_min=0.0,
                std_max=0.3,
                photo_mode=self.config.photo_mode,
                prob=0.75,
                device=self.device,
            ),
            # Small, local intensity variations
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.10,
            #     scale_max=0.20,
            #     std_min=0.0,
            #     std_max=0.15,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.5,
            #     device=self.device,
            # ),
            # self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_images(self):
        return dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.skullstrip,
                self.image_crop,
                self.intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
        )

    def build_output(self):
        return dict(
            image=self.build_image(),
            **self.build_images(),
            affine=Pipeline(
                SelectAffine("generation_labels_dist"), self.affine_adjuster
            ),
        )


class CropSelect(CropSynth):
    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )
        return state

    def build_image(self):
        return Pipeline(
            # Select one of the available (valid) images
            PipelineModule(
                SelectImage,
                PipelineModule(
                    RandomChoice,
                    # equal probability of selecting each image
                    SelectState("selectable_images"),
                ),
            ),
            RandGammaTransform(prob=0.33),
            self.skullstrip,
            self.image_crop,  # Transform to output FOV
            RandBiasfield(
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.03,
                std_min=0.0,
                std_max=0.3,
                photo_mode=self.config.photo_mode,
                prob=0.75,
                device=self.device,
            ),
            # Small, local intensity variations
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.10,
            #     scale_max=0.20,
            #     std_min=0.0,
            #     std_max=0.15,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.5,
            #     device=self.device,
            # ),
            # self.resolution_augmentation,
            self.intensity_normalization,
        )


class SynthOrSelectImage(DefaultSynth):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

        - linear deformation
        - nonlinear deformation
        - A choice between a real input image or the generation of a
          synthetic image as the input image for training

        """
        super().__init__(config)

    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )

    def build_image(self):
        return Pipeline(
            # Synthesize or select an image
            RandomChoice(
                [
                    # Synthesize image
                    Pipeline(
                        SelectImage("generation_labels_dist"),
                        SynthesizeIntensityImage(
                            mu_low=25.0,
                            mu_high=200.0,
                            sigma_global_scale=0.0,
                            sigma_local_scale=10.0,
                            pv_sigma_range=[0.1, 1.5],
                            min_cortical_contrast=20.0,
                            photo_mode=self.config.photo_mode,
                            device=self.device,
                        ),
                        # RandGaussianNoise(),
                        RandGammaTransform(prob=0.25),
                    ),
                    # Select one of the available (valid) images
                    Pipeline(
                        PipelineModule(
                            SelectImage,
                            PipelineModule(
                                RandomChoice,
                                # equal probability of selecting each image
                                SelectState("selectable_images"),
                            ),
                        ),
                    ),
                ],
                prob=(0.9, 0.1),
            ),
            self.image_deformation,  # Transform to output FOV
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.02,
            #     scale_max=0.04,
            #     std_min=0.1,
            #     std_max=0.6,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.9,
            #     device=self.device,
            # ),
            self.resolution_augmentation,
            self.intensity_normalization,
        )


class XSubSynth(DefaultSynth):
    def __init__(self, config: SynthesizerConfig) -> None:
        super().__init__(config)

    def initialize_spatial_transform(self):
        self.warp_surface = SubPipeline(
            PipelineModule(
                XSubWarpSurface_v2,  # or XSubWarpSurface_v1
                # SelectState("self_mni_backward_cropped"),
                SelectImage("mni152_nonlin_backward"),
                SelectImage("other:mni152_nonlin_forward"),
            ),
        )

    def build_spatial_transform(self):
        self.image_deformation = SubPipeline(
            PipelineModule(
                XSubWarpImage,
                SelectImage("mni152_nonlin_forward"),
                SelectState("other_mni_backward_cropped"),
                SelectState("self_size"),  # size of the input we sample *from*
            ),
            # We used a cropped image in `other` space so pad to achieve
            # desired output size
            PipelineModule(
                PadTransform,
                SelectState("crop_params", "pad"),
            ),
        )

        self.surface_deformation = SubPipeline(
            # compensate for offset caused by cropping in self space
            # PipelineModule(
            #     TranslationTransform,
            #     SelectState("padded_surface_bbox", 0),  # lower left
            #     invert=True,
            #     device=device,
            # ),
            # PipelineModule(
            #     XSubWarpSurface_v1,
            #     SelectState("self_mni_backward_cropped"),
            #     SelectImage("other:mni152_nonlin_forward"),
            # ),
            self.warp_surface,
            # compensate for offset caused by cropping in other space (because
            # we cropped to out_size)
            PipelineModule(
                TranslationTransform,
                SelectState("crop_params", "offset"),
                invert=True,
                device=self.device,
            ),
            CheckCoordsInside(self.config.out_size, device=self.device),
        )

    def build_state(self):
        return dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*self.config.photo_spacing_range, device=self.device),
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                unpack_inputs=False,
            ),
            selectable_images=Pipeline(
                SelectState("available_images"),
                Intersection(self.config.selectable_images),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            self_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    SelectState("available_images", 0),
                ),
                SpatialSize(),
            ),
            # the spatial dimensions of the image to warp *to*
            other_size=Pipeline(
                SelectImage("other:mni152_nonlin_backward"),
                SpatialSize(),
            ),
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # padded_surface_bbox=Pipeline(
            #     SelectInitialVertices(),
            #     # pad so hopefully the pial surface is inside...
            #     SurfaceBoundingBox(floor_and_ceil=True, pad=20.0, reduce=True),
            #     PipelineModule(
            #         RestrictBoundingBox,
            #         SelectState("self_size"),
            #         device=device,
            #     ),
            #     unpack_inputs=False,
            # ),
            # where to center the (output) FOV in the input image space
            self_out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("self_size"),
                    SelectState("surface_bbox"),
                    align_corners=self.config.align_corners,
                ),
                # RandTranslationTransform(
                #     x_range=[-10, 10],
                #     y_range=[-10, 10],
                #     z_range=[-10, 10],
                #     prob=0.5,
                #     device=self.device,
                # ),
            ),
            # transform center to `other`
            other_out_center=Pipeline(
                SelectState("self_out_center"),
                ApplyFunction(torch.atleast_2d),
                self.warp_surface,
                ApplyFunction(torch.squeeze),
            ),
            # other_out_center=Pipeline(
            #     ServeValue(config.out_center_str),
            #     PipelineModule(
            #         CenterFromString,
            #         SelectState("other_size"),
            #         align_corners=config.align_corners,
            #     ),
            # ),
            # cropping parameters in `other` space
            crop_params=Pipeline(
                SelectState("other_size"),
                PipelineModule(
                    SpatialCropParameters,
                    self.config.out_size,
                    SelectState("other_out_center"),
                ),
            ),
            # we don't need the entire space of `self` in order to transform
            # the surface so crop to a bounding box around the surface(s)
            # self_mni_backward_cropped=Pipeline(
            #     SelectImage("mni152_nonlin_backward"),
            #     PipelineModule(
            #         SpatialCrop,
            #         size=SelectState("self_size"),
            #         start=SelectState("padded_surface_bbox", 0),  # recursive selection
            #         stop=SelectState("padded_surface_bbox", 1),
            #     ),
            # ),
            # depending on the desired output FOV, we don't need to fill the
            # entire space of `other` so crop it
            other_mni_backward_cropped=Pipeline(
                SelectImage("other:mni152_nonlin_backward"),
                PipelineModule(
                    SpatialCrop,
                    size=SelectState("other_size"),
                    slices=SelectState("crop_params", "slices"),
                ),
            ),
        )

    def build_image(self):
        """No intensity augmentation."""

        return Pipeline(
            # Select one of the available (valid) images
            PipelineModule(
                SelectImage,
                PipelineModule(
                    RandomChoice,
                    # equal probability of selecting each image
                    SelectState("selectable_images"),
                ),
            ),
            self.image_deformation,  # Transform to output FOV
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.02,
            #     scale_max=0.04,
            #     std_min=0.1,
            #     std_max=0.6,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.9,
            #     device=self.device,
            # ),
            self.resolution_augmentation(),
            self.intensity_normalization,
        )

    def build_output(self):
        return dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.image_deformation,
                self.intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                self.image_deformation,
                self.intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                self.image_deformation,
                OneHotEncoding(self.config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=self.build_image(),
            surface=Pipeline(
                SelectSurface(),
                self.surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                self.surface_deformation,
                # Noise on initial vertices
                RandLinearTransform(
                    max_rotation=5.0,
                    max_scale=0.05,
                    max_shear=0.05,
                    relative_to_input=True,
                    prob=0.5,
                    device=self.device,
                ),
                RandTranslationTransform(
                    x_range=[-3, 3],
                    y_range=[-3, 3],
                    z_range=[-3, 3],
                    prob=0.5,
                    device=self.device,
                ),
                skip_on_InputSelectorError=True,
            ),
        )


class XSubSynthIso(XSubSynth):
    def resolution_augmentation(self):
        return IdentityTransform()


class SaverioSynth(SynthBuilder):
    def __init__(self, config: SynthesizerConfig) -> None:
        super().__init__(config)

    def initialize_spatial_transform(self):
        self.nonlinear_transform = RandNonlinearTransform(
            self.config.out_size,
            scale_min=0.03,  # 0.10,
            scale_max=0.06,  # 0.15,
            std_max=4.0,
            exponentiate_field=True,
            grid=self.config.grid,
            prob=1.0,
            device=self.device,
        )
        self.linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.2,
            max_shear=0.2,
            prob=1.0,
            device=self.device,
        )

    def build_state(self):
        return dict(
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    SelectState("available_images", 0),
                ),
                SpatialSize(),
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    align_corners=self.config.align_corners,
                ),
                # RandTranslationTransform(
                #     x_range=[-10, 10],
                #     y_range=[-10, 10],
                #     z_range=[-10, 10],
                #     prob=0.5,
                #     device=self.device,
                # ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(self.config.centered_grid),
                self.nonlinear_transform,
                self.linear_transform,
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=self.device,
                ),
            ),
        )

    def build_spatial_transform(self):
        self.image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

    def build_intensity_transform(self):
        self.intensity_normalization = IntensityNormalization()

    def build_resolution_transform(self):
        # self.resolution_augmentation = PipelineModule(
        #     RandResolution,
        #     self.config.out_size,
        #     self.config.in_res,
        #     prob=0.75,
        #     device=self.device,
        # )
        self.resolution_augmentation = IdentityTransform()

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage("segmentation"),
            SynthesizeIntensityImage(
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=self.config.photo_mode,
                device=self.device,
            ),
            RandGammaTransform(prob=0.33),
            self.image_deformation,  # Transform to output FOV
            RandBiasfield(
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.03,
                std_min=0.0,
                std_max=0.3,
                photo_mode=self.config.photo_mode,
                prob=0.75,
                device=self.device,
            ),
            # Small, local intensity variations
            # RandBiasfield(
            #     self.config.out_size,
            #     scale_min=0.10,
            #     scale_max=0.20,
            #     std_min=0.0,
            #     std_max=0.15,
            #     photo_mode=self.config.photo_mode,
            #     prob=0.5,
            #     device=self.device,
            # ),
            self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_images(self):
        return dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.image_deformation,
                self.intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            seg=Pipeline(
                SelectImage("segmentation"),
                self.image_deformation,
                OneHotEncoding(self.config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
        )

    def build_output(self):
        return dict(
            image = self.build_image(),
            **self.build_images(),
        )