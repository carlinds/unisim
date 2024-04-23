# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UniSim model.

https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf

"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import numpy as np
import torch
from jaxtyping import Float, Int
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.lidars import transform_points_pairwise
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.dataparsers.pandaset_dataparser import ALLOWED_RIGID_CLASSES
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.cnns import BasicBlock
from nerfstudio.model_components.losses import L1Loss, MSELoss, VGGPerceptualLossPix2Pix
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, FeatureRenderer, NormalsRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.model_components.utils import SigmoidDensity
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.math import GaussiansStd, chamfer_distance, erf_approx
from nerfstudio.viewer.server.viewer_elements import ViewerCheckbox, ViewerSlider
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from unisim.ray_samplers import UniSimSampler, UniSimSamplerConfig, UniSimSampling
from unisim.unisim_field import GaussianUniSimField, UniSimField

RGBDecoderType = Literal["cnn", "mlp"]
EPS = 1e-7


@dataclass
class UniSimModelConfig(ADModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: UniSimModel)
    nff_out_dim: int = 32
    """Dimensionality of the neural feature field output."""
    nff_hidden_dim: int = 32
    """Dimensionality of the neural feature field hidden layers."""
    hashgrid_dim: int = 2
    """Number of dimensions of each hashgrid feature (per level)."""
    sdf_to_density_slope: float = 20.0  # TODO: tune this (maybe 20.0 like in NeuSim)
    """Slope of the sigmoid function used to convert SDF to density."""
    learnable_beta: bool = True
    """Whether to learn the beta (sdf to density) parameter or not."""
    sampler: UniSimSamplerConfig = field(default_factory=UniSimSamplerConfig)
    """Sampler configuration."""
    update_occ_grid_every_n: int = -1
    """Update the occupancy grid every n iterations."""
    traj_opt_start_phase: int = 1
    """The phase at which to start optimizing the trajectories."""

    static_num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    static_base_res: int = 16
    """Resolution of the base grid for the hasgrid."""
    static_max_res: int = 4096
    """Maximum resolution of the hashmap for the base mlp."""
    static_log2_hashmap_size: int = 21
    """Maximum size of the static-world hashmap (per level)."""
    static_use_gauss_field: bool = True
    """Whether to use a gaussian field for the static-world field or not."""

    sky_res: int = 4096
    """Resolution of the sky hashmap."""
    sky_log2_hashmap_size: int = 19
    """Maximum size of the sky hash table."""

    use_dynamic_field: bool = True
    """Whether to use the dynamic field or not."""
    dynamic_num_levels: int = 8
    """Number of levels of the hashmap for each dynamic (actor) base mlp."""
    dynamic_base_res: int = 16
    """Resolution of the base grid for each dynamic (actor) hasgrid."""
    dynamic_max_res: int = 256
    """Maximum resolution of the hashmap for each dynamic (actor) base mlp."""
    dynamic_log2_hashmap_size: int = 14
    """Maximum size of the dynamic (actor) hashmap (per level)."""
    dynamic_use_gauss_field: bool = False
    """Whether to use a gaussian field for the dynamic (actor) field or not."""
    use_hypernet: bool = True
    """Whether actor feature grids should be computed using a hypernet or not."""
    actor_embedding_dim: int = 64
    """Dimensionality of the actor embedding."""
    hypernet_layers: int = 2
    """Number of layers in the hypernet."""
    hypernet_hidden_dim: int = 64
    """Dimensionality of the hidden layers in the hypernet."""
    use_random_flip: bool = True
    """Whether to randomly flip the sign of positions and directions fed to actor networks."""
    use_class_specific_embedding: bool = False
    """Whether to use class-specific embeddings for actors."""
    use_shared_actor_embedding: bool = False

    rgb_decoder_type: RGBDecoderType = "cnn"
    """What type of image (rgb) decoder to use."""
    rgb_upsample_factor: int = 3
    """Upsampling factor for the rgb decoder."""
    rgb_hidden_dim: int = 32  # TODO: 64? (for correct param count)
    """Dimensionality of the hidden layers in the CNN."""
    # Conv-only settings. TODO: Replace with some full CNN config?
    conv_use_bn: bool = True
    """Whether to use batch norm in the CNN decoder."""

    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    predict_normals: bool = False
    """Whether to predict normals or not."""

    vgg_loss_mult: float = 0.05
    """Multipier for VGG perceptual loss."""
    vgg_uniform: bool = False
    """Whether to uniform weights across the VGG scales (otherwise we decrease the weight of early layers)."""
    rgb_loss_mult: float = 5.0
    """Multipier for RGB loss."""
    lidar_loss_mult: float = 0.01
    """Multipier for lidar loss."""
    lidar_use_mse: bool = True
    """Whether to use MSE or L1 for the lidar loss."""
    ray_drop_loss_mult: float = 0.0
    """Multipier for lidar point drop loss. NOTE: Automatically disabled if add_missing_points in dataparser is False."""
    regularization_loss_mult: float = 0.01
    """Multiplier for regularization loss."""
    regularization_epsilon: float = 0.1
    """Epsilon for regularization loss, in meters."""
    quantile_threshold: float = 0.95
    """Quantile threshold for lidar and regularization losses."""
    actor_norm_loss_mult: float = 0.1
    """Multipier for actor norm loss"""
    actor_motion_model_regularization_loss_mult: float = 0.0
    """Multipier for actor motion model regularization loss"""

    numerical_gradients_delta: float = 0.01
    """Delta for numerical gradients, in meters."""
    use_numerical_gradients: bool = True
    """Use numerical gradients for the regularization loss."""
    compensate_upsampling_when_rendering: bool = True
    """Compensate for upsampling when asked to render an image of some given resolution."""
    normalize_depth: bool = False
    """Whether to normalize depth by dividing by accumulation."""

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""

    sdf_offset: float = 0.0
    """Offset to add to the SDF values."""
    learn_sdf_offset: bool = False
    """Whether to learn the SDF offset or not."""

    model_lidar_return_prob: bool = False
    """Whether to model the lidar return probability or not."""
    non_return_lidar_distance: float = 150.0
    """Distance at which we assume that the lidar does not return points."""

    def __post_init__(self) -> None:
        if self.rgb_decoder_type == "mlp":
            self.rgb_upsample_factor = 1
            self.vgg_loss_mult = 0.0  # no patches -> no vgg loss


class UniSimModel(ADModel):
    """UniSim model.

    Args:
        config: UniSim configuration to instantiate model
    """

    config: UniSimModelConfig

    def update_phase(self, phase: int):
        if phase >= self.config.traj_opt_start_phase and self.dynamic_actors.config.optimize_trajectories:
            self.dynamic_actors.requires_grad_(True)
        else:
            self.dynamic_actors.requires_grad_(False)
        if self.config.rgb_decoder_type == "cnn":
            # freeze the cnn decoder during phase 1
            self.rgb_decoder.requires_grad_(phase > 0)
            self.use_vgg_loss = phase > 0 and self.config.vgg_loss_mult > 0.0
        else:
            self.use_vgg_loss = False

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.use_vgg_loss = False

        # Fields
        self.sdf_to_density = SigmoidDensity(
            self.config.sdf_to_density_slope, learnable_beta=self.config.learnable_beta
        )
        static_field = GaussianUniSimField if self.config.static_use_gauss_field else UniSimField
        self.static_field = static_field(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            implementation=self.config.implementation,
            hidden_dim=self.config.nff_hidden_dim,
            nff_out_dim=self.config.nff_out_dim,
            sdf_to_density=self.sdf_to_density,
            hashgrid_dim=self.config.hashgrid_dim,
            # static-scene specifics
            num_levels=self.config.static_num_levels,
            base_res=self.config.static_base_res,
            max_res=self.config.static_max_res,
            log2_hashmap_size=self.config.static_log2_hashmap_size,
            inverted_sphere=False,
            numerical_gradients_delta=self.config.numerical_gradients_delta,  # TODO: 1.0 / self.config.static_max_res?
            use_numerical_gradients=self.config.use_numerical_gradients,
            sdf_offset=self.config.sdf_offset,
            learn_sdf_offset=self.config.learn_sdf_offset,
            model_lidar_return_prob=self.config.model_lidar_return_prob,
        )
        self.sky_field = UniSimField(
            aabb=self.scene_box.aabb,  # But not used with spatial distortion
            num_images=self.num_train_data,
            implementation=self.config.implementation,
            hidden_dim=self.config.nff_hidden_dim,
            nff_out_dim=self.config.nff_out_dim,
            sdf_to_density=self.sdf_to_density,
            hashgrid_dim=self.config.hashgrid_dim,
            # sky specifics
            num_levels=self.config.static_num_levels,
            base_res=self.config.static_base_res,
            max_res=self.config.sky_res,
            log2_hashmap_size=self.config.sky_log2_hashmap_size,
            inverted_sphere=True,
            # TODO: numerical_gradients_delta=1.0 / self.config.sky_res?
            use_numerical_gradients=self.config.use_numerical_gradients,
            sdf_offset=self.config.sdf_offset,
            learn_sdf_offset=self.config.learn_sdf_offset,
            model_lidar_return_prob=self.config.model_lidar_return_prob,
        )

        assert (
            "trajectories" in self.kwargs["metadata"].keys()
        ), "Metadata must contain trajectories for dynamic agents."
        trajectories = self.kwargs["metadata"]["trajectories"]
        aabbs = torch.stack(
            [-self.dynamic_actors.actor_sizes / 2, self.dynamic_actors.actor_sizes / 2],
            dim=1,
        )
        self.register_buffer("actor_aabbs", aabbs)
        # self.dynamic_actors.n_actors = len(trajectories)
        if self.config.use_class_specific_embedding:
            # TODO: ugly import of classes, only supports pandaset for now
            self.class_idx_map = {cls_name: i for i, cls_name in enumerate(ALLOWED_RIGID_CLASSES)}
            actor_classes = torch.tensor([self.class_idx_map[traj["label"]] for traj in trajectories])
            self.register_buffer("actor_classes", actor_classes)

        assert not (
            self.config.use_class_specific_embedding and self.config.use_shared_actor_embedding
        ), "Cannot use both class-specific and shared actor embeddings"

        if self.config.use_class_specific_embedding:
            self.class_slider = ViewerSlider(
                "Class index",
                -1.0,
                -1.0,
                len(self.class_idx_map) - 1,
                1.0,
            )
            self.only_render_class = ViewerCheckbox("Only render class embedding", False)
            self.only_render_actor = ViewerCheckbox("Only render actor embedding", False)

        if self.config.use_shared_actor_embedding:
            self.only_render_shared = ViewerCheckbox("Only render shared embedding", False)
            self.only_render_actor = ViewerCheckbox("Only render actor embedding", False)

        dynamic_field = GaussianUniSimField if self.config.dynamic_use_gauss_field else UniSimField
        self.dynamic_field = dynamic_field(
            num_images=self.num_train_data,
            implementation=self.config.implementation,
            hidden_dim=self.config.nff_hidden_dim,
            nff_out_dim=self.config.nff_out_dim,
            sdf_to_density=self.sdf_to_density,
            hashgrid_dim=self.config.hashgrid_dim,
            # dynamic-scene specifics
            num_levels=self.config.dynamic_num_levels,
            base_res=self.config.dynamic_base_res,
            max_res=self.config.dynamic_max_res,
            log2_hashmap_size=self.config.dynamic_log2_hashmap_size,
            inverted_sphere=False,
            # TODO: numerical_gradients_delta=1.0 / self.config.dynamic_max_res ?
            use_numerical_gradients=self.config.use_numerical_gradients,
            sdf_offset=self.config.sdf_offset,
            learn_sdf_offset=self.config.learn_sdf_offset,
            model_lidar_return_prob=self.config.model_lidar_return_prob,
        )
        if self.config.use_hypernet:
            self.actor_embedding = torch.nn.Embedding(self.dynamic_actors.n_actors, self.config.actor_embedding_dim)

            self.actor_embedding.weight = torch.nn.Parameter(self.actor_embedding.weight * 0.001)  # TODO: 0.003?

            hypernet_implementation = "torch" if not self.config.use_numerical_gradients else self.config.implementation
            self.actor_hypernet = MLP(
                in_dim=self.config.actor_embedding_dim,
                layer_width=self.config.hypernet_hidden_dim,
                out_dim=self._compute_hypernet_out_dim(),
                num_layers=self.config.hypernet_layers,
                implementation=hypernet_implementation,
            )

            if self.config.use_class_specific_embedding:
                self.actor_embedding = torch.nn.Embedding(self.dynamic_actors.n_actors, self.config.actor_embedding_dim)
                self.actor_embedding.weight = torch.nn.Parameter(self.actor_embedding.weight * 0.001)
                self.class_embedding = torch.nn.Embedding(len(ALLOWED_RIGID_CLASSES), self.config.actor_embedding_dim)
                self.class_embedding.weight = torch.nn.Parameter(self.class_embedding.weight * 0.001)

            if self.config.use_shared_actor_embedding:
                self.actor_embedding = torch.nn.Embedding(self.dynamic_actors.n_actors, self.config.actor_embedding_dim)
                self.actor_embedding.weight = torch.nn.Parameter(self.actor_embedding.weight * 0.001)
                self.shared_actor_embedding = torch.nn.Embedding(1, self.config.actor_embedding_dim)
                self.shared_actor_embedding.weight = torch.nn.Parameter(self.shared_actor_embedding.weight * 0.001)
        else:
            # TODO: delete the current hashgrid? slight memory optimization
            # self.dynamic_field.hash_encoding = None
            self.actor_hashgrids = torch.nn.ModuleList(
                [
                    HashEncoding(
                        num_levels=self.config.dynamic_num_levels,
                        min_res=self.config.dynamic_base_res,
                        max_res=self.config.dynamic_max_res,
                        log2_hashmap_size=self.config.dynamic_log2_hashmap_size,
                        features_per_level=self.config.hashgrid_dim,
                        implementation=self.config.implementation,
                    )
                    for _ in range(self.dynamic_actors.n_actors)
                ]
            )

        self.rgb_decoder = self._build_rgb_decoder()
        self.lidar_decoder = MLP(
            in_dim=self.config.nff_out_dim,
            layer_width=32,
            out_dim=2,
            num_layers=3,
            implementation=self.config.implementation,
            out_activation=None,
        )  # TODO: add ray drop probability?

        # Sampler
        # Note! results in unequal number of samples per ray
        alpha_fn = self.static_field.alpha_fn if self.config.update_occ_grid_every_n > 0 else None
        self.sampler: UniSimSampler = self.config.sampler.setup(
            aabb=self.scene_box.aabb,
            dynamic_actors=self.dynamic_actors,
            alpha_fn=alpha_fn,
        )
        if not self.config.use_dynamic_field:
            self.sampler.dynamic_enabled.value = False  # haxx

        # Collider - will be used in forward() to set ray nears and fars
        self.collider = AABBBoxCollider(self.scene_box)

        # renderers
        self.renderer_feat = FeatureRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected") if self.config.normalize_depth else render_depth_simple
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.depth_loss = MSELoss(reduction="none") if self.config.lidar_use_mse else L1Loss(reduction="none")
        self.intensity_loss = MSELoss(reduction="none")
        self.vgg_loss = VGGPerceptualLossPix2Pix(weights=(1.0, 1.0, 1.0, 1.0, 1.0) if self.config.vgg_uniform else None)
        self.ray_drop_loss = BCEWithLogitsLoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.median_l2 = lambda pred, gt: torch.median((pred - gt) ** 2)
        self.mean_rel_l2 = lambda pred, gt: torch.mean(((pred - gt) / gt) ** 2)
        self.rmse = lambda pred, gt: torch.sqrt(torch.mean((pred - gt) ** 2))
        self.chamfer_distance = lambda pred, gt: chamfer_distance(pred, gt, 1_000, True)

    def _build_rgb_decoder(self) -> Callable:
        hidden_dim = self.config.rgb_hidden_dim
        if self.config.rgb_decoder_type == "cnn":
            return torch.nn.Sequential(
                torch.nn.Conv2d(self.config.nff_out_dim, hidden_dim, kernel_size=1, padding=0),
                # TODO: torch.nn.BatchNorm2d(hidden_dim) if self.config.conv_use_bn else torch.nn.Identity() ?
                torch.nn.ReLU(inplace=True),
                BasicBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=7,
                    padding=3,
                    use_bn=self.config.conv_use_bn,
                ),
                BasicBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=7,
                    padding=3,
                    use_bn=self.config.conv_use_bn,
                ),
                torch.nn.ConvTranspose2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=self.config.rgb_upsample_factor,
                    stride=self.config.rgb_upsample_factor,
                ),
                BasicBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=7,
                    padding=3,
                    use_bn=self.config.conv_use_bn,
                ),
                BasicBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=7,
                    padding=3,
                    use_bn=self.config.conv_use_bn,
                ),
                torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
                torch.nn.Sigmoid(),
            )
        elif self.config.rgb_decoder_type == "mlp":
            return MLP(
                in_dim=self.config.nff_out_dim,
                layer_width=self.config.rgb_hidden_dim,
                out_dim=3,
                num_layers=3,
                implementation=self.config.implementation,
                out_activation=torch.nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown rgb decoder type: {self.config.rgb_decoder_type}")

    def _compute_hypernet_out_dim(self):
        growth_factor = np.exp(
            (np.log(self.config.dynamic_max_res) - np.log(self.config.dynamic_base_res))
            / (self.config.dynamic_num_levels - 1)
        )
        hashgrid_levels = [
            math.ceil(self.config.dynamic_base_res * growth_factor**i) for i in range(self.config.dynamic_num_levels)
        ]
        assert hashgrid_levels[-1] == self.config.dynamic_max_res, "Last level should be max_res"
        clipped_hashgrid_levels = [min(level**3, 2**self.config.dynamic_log2_hashmap_size) for level in hashgrid_levels]
        hypernet_out_dim = sum(clipped_hashgrid_levels) * self.config.hashgrid_dim
        return hypernet_out_dim

    def disable_ray_drop(self):
        self.config.ray_drop_loss_mult = 0.0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        model_groups = {}
        model_groups["fields"] = (
            list(self.static_field.parameters())
            + list(self.sky_field.parameters())
            + list(self.lidar_decoder.parameters())
            + list(self.dynamic_field.parameters())
        )
        if self.config.use_hypernet:
            model_groups["fields"] += list(self.actor_embedding.parameters()) + list(self.actor_hypernet.parameters())
            if self.config.use_class_specific_embedding:
                model_groups["fields"] += list(self.class_embedding.parameters())
            if self.config.use_shared_actor_embedding:
                model_groups["fields"] += list(self.shared_actor_embedding.parameters())
        else:
            model_groups["fields"] += list(self.actor_hashgrids.parameters())

        # RGB decoder is handled differently depending on the type
        if self.config.rgb_decoder_type == "cnn":
            model_groups["cnn"] = self.rgb_decoder.parameters()
        elif self.config.rgb_decoder_type == "mlp":
            model_groups["fields"] += list(self.rgb_decoder.parameters())

        return model_groups

    def forward(
        self,
        ray_bundle: RayBundle,
        patch_size: Tuple[int, int] = (1, 1),
        intensity_for_cam: bool = False,
        calc_lidar_losses: bool = True,
    ):
        ray_bundle = self.collider(ray_bundle)  # Check where each ray intersects with the static scene bounds
        return self.get_outputs(ray_bundle, patch_size, intensity_for_cam, calc_lidar_losses)

    def get_outputs(
        self,
        ray_bundle: RayBundle,
        patch_size: Tuple[int, int],
        intensity_for_cam: bool = False,
        calc_lidar_losses: bool = True,
    ):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        nff_outputs = self.get_nff_outputs(ray_bundle, calc_lidar_losses)
        rgb, intensity, ray_drop_logits = self.decode_features(
            features=nff_outputs["features"],
            patch_size=patch_size,
            is_lidar=ray_bundle.metadata.get("is_lidar"),
            intensity_for_cam=intensity_for_cam,
        )
        final_outputs = {
            "rgb": rgb,
            "intensity": intensity,
            "ray_drop_logits": ray_drop_logits,
            **nff_outputs,
        }
        return final_outputs

    def decode_features(
        self,
        features: Tensor,
        patch_size: Tuple[int, int],
        is_lidar: Optional[Tensor] = None,
        intensity_for_cam: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Take the neural feature field feature, and render them to rgb and/or intensity."""
        if is_lidar is None:
            lidar_features = features.new_empty((0, *features.shape[1:]))
            cam_features = features
        else:
            lidar_features, cam_features = (
                features[is_lidar[..., 0]],
                features[~is_lidar[..., 0]],
            )

        # Decode lidar features
        if intensity_for_cam:  # Useful for visualization
            intensity, ray_drop_logit = self.lidar_decoder(features).float().split(1, dim=-1)  # TODO: hmm
        else:
            intensity, ray_drop_logit = self.lidar_decoder(lidar_features).split(1, dim=-1)

        intensity = intensity.sigmoid()
        # Decode camera features
        if self.config.rgb_decoder_type == "mlp":
            rgb = self.rgb_decoder(cam_features)
            rgb = rgb.view(-1, *patch_size, 3)  # B x D x D x 3
        elif self.config.rgb_decoder_type == "cnn":
            cam_feature_patches = cam_features.view(-1, *patch_size, cam_features.shape[-1])  # B x D x D x C
            cam_feature_patches = cam_feature_patches.permute(0, 3, 1, 2)  # B x C x D x D
            rgb = self.rgb_decoder(cam_feature_patches)  # B x 3 x upsample x upsample
            rgb = rgb.permute(0, 2, 3, 1)  # B x upsample x upsample x 3
        else:
            raise NotImplementedError(f"Unsupported {self.config.rgb_decoder_type} and {self.training}")

        return rgb, intensity, ray_drop_logit

    def get_nff_outputs(self, ray_bundle: RayBundle, calc_lidar_losses: bool = False) -> Dict[str, Tensor]:
        """Run the neural feature field, and return the rendered outputs."""
        ray_bundle = self._scale_pixel_area(ray_bundle)
        sampling: UniSimSampling = self.sampler(ray_bundle)
        assert sampling.ray_samples.metadata is not None, "Ray samples must have metadata"
        if self.training and "is_lidar" in ray_bundle.metadata:
            self._compute_is_close_to_lidar(sampling)
        ray_indices = sampling.ray_samples.metadata["ray_indices"][..., 0]
        num_rays = ray_bundle.shape[0]
        predict_normals = self.config.predict_normals or self.training

        outputs = self._forward_fields(sampling, predict_normals)  # Note! not sorted, use sampling.ordering
        if self.config.model_lidar_return_prob and "is_lidar" in ray_bundle.metadata:
            assert FieldHeadNames.LIDAR_RETURN_PROB in outputs
            outputs[FieldHeadNames.LIDAR_RETURN_PROB] = outputs[FieldHeadNames.LIDAR_RETURN_PROB].reshape(
                outputs[FieldHeadNames.ALPHA].shape
            )
            is_lidar = (
                sampling.ray_samples.metadata["is_lidar"]
                .reshape(outputs[FieldHeadNames.ALPHA].shape)
                .to(outputs[FieldHeadNames.LIDAR_RETURN_PROB])
            )
            outputs[FieldHeadNames.ALPHA] = outputs[FieldHeadNames.ALPHA] * (
                1 - (1 - outputs[FieldHeadNames.LIDAR_RETURN_PROB]) * is_lidar
            )

        if str(self.device) in ("cpu", "mps"):
            # Note: for debugging on devices without cuda
            weights = torch.zeros_like(outputs[FieldHeadNames.ALPHA][..., 0]) + 0.5
        else:
            weights, _ = nerfacc.render_weight_from_alpha(
                outputs[FieldHeadNames.ALPHA][sampling.ordering][..., 0],
                ray_indices=ray_indices,
                n_rays=num_rays,
            )

        # Render rays
        nff_outputs = {
            "features": self.renderer_feat(
                features=outputs[FieldHeadNames.FEATURE][sampling.ordering],
                weights=weights[..., None],
                ray_indices=ray_indices,
                num_rays=num_rays,
            ),
            "depth": self.renderer_depth(
                weights=weights[..., None],
                ray_samples=sampling.ray_samples,
                ray_indices=ray_indices,
                num_rays=num_rays,
            ),
            "accumulation": self.renderer_accumulation(
                weights=weights[..., None], ray_indices=ray_indices, num_rays=num_rays
            ),
        }
        if not self.training and FieldHeadNames.NORMALS in outputs:
            nff_outputs["normals"] = self.renderer_normals(
                normals=outputs[FieldHeadNames.NORMALS][sampling.ordering],
                weights=weights[..., None],
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
        if self.training and calc_lidar_losses:
            nff_outputs["gradients"] = outputs[FieldHeadNames.GRADIENT]
            metadata = sampling.ray_samples.metadata
            weights_mask = ((~metadata["is_close_to_lidar"]) & metadata["is_lidar"] & metadata["did_return"]).squeeze(
                -1
            )
            nff_outputs["non_nearby_weights"] = weights[weights_mask]
            lidar_start_ray = ray_indices[metadata["is_lidar"].squeeze(-1)].min()
            nff_outputs["non_nearby_lidar_ray_indices"] = ray_indices[weights_mask] - lidar_start_ray

            """lidar_samples = sampling.ray_samples[sampling.ray_samples.metadata["is_lidar"].squeeze()]
            lidar_weights = (weights[metadata["is_lidar"].squeeze()]+1e-5).log()
            diff = ((lidar_samples.frustums.ends-lidar_samples.frustums.starts)/2 - lidar_samples.metadata["directions_norm"])
            variance = 0.1
            lidar_values = (-diff.pow(2)/(2*variance))
            lidar_ray_indices = lidar_samples.metadata["ray_indices"][..., 0]
            depth_loss  = -nerfacc.accumulate_along_rays(weights=lidar_weights, values=lidar_values.exp(), ray_indices=lidar_ray_indices, n_rays=num_rays)[ray_bundle.metadata["is_lidar"].squeeze()]"""

        return nff_outputs

    def _get_actor_embeddings(self, actor_indices: Tensor) -> Tensor:
        actor_embeddings = self.actor_embedding(actor_indices)
        if self.config.use_class_specific_embedding:
            if int(self.class_slider.value) == -1:
                actor_classes = self.actor_classes[actor_indices]
            else:
                actor_classes = torch.ones_like(actor_indices) * int(self.class_slider.value)

            class_embeddings = self.class_embedding(actor_classes)
            if self.only_render_class.value:
                z = class_embeddings
            elif self.only_render_actor.value:
                z = actor_embeddings
            else:
                z = actor_embeddings + class_embeddings
        elif self.config.use_shared_actor_embedding:
            if self.only_render_shared.value:
                z = self.shared_actor_embedding(torch.tensor(0, device=actor_embeddings.device))
            elif self.only_render_actor.value:
                z = actor_embeddings
            else:
                z = actor_embeddings + self.shared_actor_embedding(torch.tensor(0, device=actor_embeddings.device))
        else:
            z = actor_embeddings
        return z

    def _forward_dynamic_gauss_field(self, sampling: UniSimSampling, predict_normals: bool):
        assert sampling.dynamic_ray_samples is not None, "Dynamic ray samples must be provided"
        non_empty_sample_idx = sampling.dynamic_ray_samples.metadata[
            "object_indices"
        ].unique_consecutive()  # faster, as it's sorted
        if self.config.use_hypernet:
            actor_embeddings = self._get_actor_embeddings(non_empty_sample_idx)
            hashgrid_weights = self.actor_hypernet(actor_embeddings)

        w2b = sampling.dynamic_ray_samples.metadata["world2box"]
        sampling.dynamic_ray_samples.frustums.directions = transform_points_pairwise(
            sampling.dynamic_ray_samples.frustums.directions,
            w2b,
            with_translation=False,
        ).squeeze(1)  # TODO: make sure these are not used anywhere else
        sampling.dynamic_ray_samples.frustums.directions = sampling.dynamic_ray_samples.frustums.directions / (
            torch.linalg.norm(
                sampling.dynamic_ray_samples.frustums.directions,
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )
        sampling.dynamic_ray_samples.frustums.origins = transform_points_pairwise(
            sampling.dynamic_ray_samples.frustums.origins, w2b
        ).squeeze(1)  # TODO: make sure these are not used anywhere else

        aabbs = self.actor_aabbs[sampling.dynamic_ray_samples.metadata["object_indices"]].squeeze(1)
        aabbs = aabbs + torch.stack(
            (-sampling.bbox_padding.unsqueeze(0), sampling.bbox_padding.unsqueeze(0)),
            dim=1,
        )
        gaussians: GaussiansStd = self.dynamic_field.get_positions(sampling.dynamic_ray_samples, aabb=aabbs)

        if self.training and self.config.use_random_flip:
            # randomly flip sign of x for directions and positions
            symmetric = sampling.dynamic_ray_samples.metadata["symmetric"].squeeze(-1)
            flip = torch.ones_like(gaussians.mean)[:, 0, :]
            flip[..., 0] = (
                1 - symmetric * torch.randint(0, 2, (gaussians.mean.shape[0],), device=gaussians.mean.device) * 2
            )
            gaussians.mean = (gaussians.mean - 0.5) * flip.unsqueeze(1) + 0.5  # positions are in [0,1]
            sampling.dynamic_ray_samples.frustums.directions = (
                sampling.dynamic_ray_samples.frustums.directions * flip.squeeze(1)
            )

        all_feats = []
        all_positions = []
        for i, i_actor in enumerate(non_empty_sample_idx):
            curr_mask = (sampling.dynamic_ray_samples.metadata["object_indices"] == i_actor).view(-1)
            gauss = gaussians[curr_mask]
            prefix_shape = list(gauss.mean.shape[:-1])

            if self.config.use_hypernet:
                param_dict = {"tcnn_encoding.params": hashgrid_weights[i]}
                grid_feats = torch.func.functional_call(
                    self.dynamic_field.hash_encoding, param_dict, gauss.mean.view(-1, 3)
                )
            else:
                grid_feats = self.actor_hashgrids[i_actor].forward(gauss.mean.view(-1, 3))

            grid_feats = grid_feats.view(
                prefix_shape
                + [self.dynamic_field.hash_encoding.num_levels * self.dynamic_field.hash_encoding.features_per_level]
            ).unflatten(
                -1,
                (
                    self.dynamic_field.hash_encoding.num_levels,
                    self.dynamic_field.hash_encoding.features_per_level,
                ),
            )  # [..., "n_samples", "num_levels", "features_per_level"]

            weights = erf_approx(
                1
                / (8**0.5 * gauss.std[..., None] * self.dynamic_field.hash_encoding.scalings.view(-1))
                .abs()
                .clamp_min(EPS)  # type: ignore
            )  # [..., "n_samples", "num_levels"]
            grid_feats = (
                (grid_feats * weights[..., None]).mean(dim=-3).flatten(-2, -1)
            )  # [..., "n_samples", "num_levels * features_per_level"]
            all_feats.append(grid_feats)
            all_positions.append(gauss)

        actor_outputs = self.dynamic_field.forward(
            sampling.dynamic_ray_samples,
            positions=all_positions[0].cat(all_positions[1:]),
            compute_normals=predict_normals,
            override_grid_features=torch.cat(all_feats, dim=0),
        )
        return actor_outputs

    def _forward_dynamic_field(self, sampling: UniSimSampling, predict_normals: bool):
        assert sampling.dynamic_ray_samples is not None, "Dynamic ray samples must be provided"
        non_empty_sample_idx = sampling.dynamic_ray_samples.metadata[
            "object_indices"
        ].unique_consecutive()  # faster, as it's sorted
        if self.config.use_hypernet:
            actor_embeddings = self._get_actor_embeddings(non_empty_sample_idx)
            hashgrid_weights = self.actor_hypernet(actor_embeddings)

        w2b = sampling.dynamic_ray_samples.metadata["world2box"]
        sampling.dynamic_ray_samples.frustums.directions = transform_points_pairwise(
            sampling.dynamic_ray_samples.frustums.directions,
            w2b,
            with_translation=False,
        ).squeeze(1)  # TODO: make sure these are not used anywhere else
        sampling.dynamic_ray_samples.frustums.directions = sampling.dynamic_ray_samples.frustums.directions / (
            torch.linalg.norm(
                sampling.dynamic_ray_samples.frustums.directions,
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )
        sampling.dynamic_ray_samples.frustums.origins = transform_points_pairwise(
            sampling.dynamic_ray_samples.frustums.origins, w2b
        ).squeeze(1)  # TODO: make sure these are not used anywhere else

        aabbs = self.actor_aabbs[sampling.dynamic_ray_samples.metadata["object_indices"]].squeeze(1)
        aabbs = aabbs + torch.stack(
            (-sampling.bbox_padding.unsqueeze(0), sampling.bbox_padding.unsqueeze(0)),
            dim=1,
        )
        positions = self.dynamic_field.get_positions(sampling.dynamic_ray_samples, aabb=aabbs)

        if self.training and self.config.use_random_flip:
            # randomly flip sign of x for directions and positions
            symmetric = sampling.dynamic_ray_samples.metadata["symmetric"].squeeze(-1)
            flip = torch.ones_like(positions)
            flip[:, 0] = 1 - symmetric * torch.randint(0, 2, (positions.shape[0],), device=positions.device) * 2
            positions = (positions - 0.5) * flip + 0.5  # positions are in [0,1]
            sampling.dynamic_ray_samples.frustums.directions = sampling.dynamic_ray_samples.frustums.directions * flip

        all_feats = []
        all_positions = []
        for i, i_actor in enumerate(non_empty_sample_idx):
            curr_mask = (sampling.dynamic_ray_samples.metadata["object_indices"] == i_actor).view(-1)
            pos = positions[curr_mask]

            if self.config.use_hypernet:
                param_dict = {"tcnn_encoding.params": hashgrid_weights[i]}
                grid_feats = torch.func.functional_call(self.dynamic_field.hash_encoding, param_dict, pos)
            else:
                grid_feats = self.actor_hashgrids[i_actor].forward(pos)
            all_feats.append(grid_feats)
            all_positions.append(pos)

        actor_outputs = self.dynamic_field.forward(
            sampling.dynamic_ray_samples,
            positions=torch.cat(all_positions, dim=0),
            compute_normals=predict_normals,
            override_grid_features=torch.cat(all_feats, dim=0),
        )
        return actor_outputs

    def _forward_fields(self, sampling: UniSimSampling, predict_normals: bool):
        outs = []
        if sampling.static_ray_samples is not None:
            outs.append(self.static_field.forward(sampling.static_ray_samples, compute_normals=predict_normals))
        if sampling.sky_ray_samples is not None:
            outs.append(self.sky_field.forward(sampling.sky_ray_samples, compute_normals=predict_normals))
        if sampling.dynamic_ray_samples is not None:
            if self.config.dynamic_use_gauss_field:
                outs.append(self._forward_dynamic_gauss_field(sampling, predict_normals))
            else:
                outs.append(self._forward_dynamic_field(sampling, predict_normals))
        keys = set(outs[0].keys())
        assert all([set(o.keys()) == keys for o in outs])
        return {k: torch.cat([d[k] for d in outs], dim=0) for k in keys}

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        if "image" in batch:
            image, rgb = batch["image"].to(self.device), outputs["rgb"]
            metrics_dict["psnr"] = self.psnr(rgb.detach(), image)
        if "lidar" in batch:
            mask = (batch["is_lidar"].squeeze(-1) & batch["did_return"].squeeze(-1)).to(self.device)
            points_did_return = batch["did_return"][batch["is_lidar"].squeeze(-1)].squeeze(-1).to(self.device)
            points_intensities = batch["lidar"][..., 3:4].to(self.device)
            termination_depth = batch["distance"].to(self.device)
            ray_drop_logits = outputs["ray_drop_logits"]

            pred_depth = outputs["depth"][batch["is_lidar"].squeeze(-1)]
            pred_intensity = outputs["intensity"]

            # eval metrics
            metrics_dict["depth_median_l2"] = self.median_l2(
                pred_depth[points_did_return], termination_depth[points_did_return]
            )
            metrics_dict["depth_mean_rel_l2"] = self.mean_rel_l2(
                pred_depth[points_did_return], termination_depth[points_did_return]
            )
            metrics_dict["intensity_rmse"] = self.rmse(
                pred_intensity[points_did_return], points_intensities[points_did_return]
            )
            # TODO: should consider if we actually train for ray drop
            metrics_dict["ray_drop_accuracy"] = (
                ((ray_drop_logits.sigmoid() > 0.5).squeeze(-1) == ~points_did_return).float().mean()
            )

            # train metrics / losses
            if self.training:
                unreduced_depth_loss = self.depth_loss(termination_depth, pred_depth)
                quantile = torch.quantile(
                    unreduced_depth_loss[points_did_return],
                    self.config.quantile_threshold,
                )
                quantile_mask = (unreduced_depth_loss < quantile).squeeze(-1) & points_did_return

                metrics_dict["depth_loss"] = torch.mean(unreduced_depth_loss[quantile_mask])
                metrics_dict["intensity_loss"] = self.intensity_loss(
                    points_intensities[quantile_mask], pred_intensity[quantile_mask]
                ).mean()

                quantile_weights_mask = quantile_mask[outputs["non_nearby_lidar_ray_indices"]].squeeze(-1)
                weights_loss = (outputs["non_nearby_weights"][quantile_weights_mask] ** 2).sum()
                # no need to mask gradients, they are only computed for samples that are nearby (for efficiency)
                grad_norm = torch.linalg.norm(outputs["gradients"], dim=-1)
                eikonal_loss = ((grad_norm - 1) ** 2).sum()
                metrics_dict["regularization_loss"] = (weights_loss + eikonal_loss) / mask.sum()  # avg per ray
                metrics_dict["ray_drop_loss"] = self.ray_drop_loss(
                    ray_drop_logits,
                    (~points_did_return).unsqueeze(-1).to(ray_drop_logits),
                )

        metrics_dict["sdf_to_density"] = self.sdf_to_density.beta.item()
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        if self.config.use_class_specific_embedding:
            metrics_dict["actor_embedding_norm"] = self.actor_embedding.weight.norm(dim=-1).mean()
            metrics_dict["class_embedding_norm"] = self.class_embedding.weight.norm(dim=-1).mean()

        if self.config.use_shared_actor_embedding:
            metrics_dict["actor_embedding_norm"] = self.actor_embedding.weight.norm(dim=-1).mean()
            metrics_dict["shared_actor_embedding_norm"] = self.shared_actor_embedding.weight.norm(dim=-1).mean()

        metrics_dict["motion_model_log_prob"] = self.dynamic_actors.get_motion_model_log_prob_for_trajectories()

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        if "image" in batch:
            image, rgb = batch["image"].to(self.device), outputs["rgb"]
            loss_dict["rgb_loss"] = self.rgb_loss(image, rgb) * self.config.rgb_loss_mult
            if self.use_vgg_loss:
                loss_dict["vgg_loss"] = self.vgg_loss(rgb, image) * self.config.vgg_loss_mult
        if self.training:
            assert metrics_dict
            if "depth_loss" in metrics_dict and "intensity_loss" in metrics_dict:
                loss_dict["lidar_loss"] = self.config.lidar_loss_mult * (
                    metrics_dict["depth_loss"] + metrics_dict["intensity_loss"]
                )
            if "regularization_loss" in metrics_dict:
                loss_dict["regularization_loss"] = (
                    self.config.regularization_loss_mult * metrics_dict["regularization_loss"]
                )
            self.camera_optimizer.get_loss_dict(loss_dict)
            if "ray_drop_loss" in metrics_dict:
                loss_dict["ray_drop_loss"] = self.config.ray_drop_loss_mult * metrics_dict["ray_drop_loss"]
        # TODO: add all unisim losses
        if self.config.use_class_specific_embedding or self.config.use_shared_actor_embedding:
            loss_dict["actor_norm_loss"] = self.config.actor_norm_loss_mult * metrics_dict["actor_embedding_norm"]

        loss_dict["motion_model_regularization_loss"] = (
            -self.config.actor_motion_model_regularization_loss_mult * metrics_dict["motion_model_log_prob"]
        )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict = {}
        images_dict = {}
        if "image" in batch:
            image, rgb = batch["image"].to(self.device), outputs["rgb"]
            acc = colormaps.apply_colormap(outputs["accumulation"])
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                # accumulation=outputs["accumulation"],
            )

            combined_rgb = torch.cat([image, rgb], dim=1)
            combined_acc = torch.cat([acc], dim=1)
            combined_depth = torch.cat([depth], dim=1)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            # all of these metrics will be logged as scalars
            metrics_dict["psnr"] = float(self.psnr(image, rgb).item())
            metrics_dict["ssim"] = float(self.ssim(image, rgb))  # type: ignore
            metrics_dict["lpips"] = float(self.lpips(image, rgb))
            images_dict.update(
                {
                    "img": combined_rgb,
                    "accumulation": combined_acc,
                    "depth": combined_depth,
                }
            )

        if "lidar" in batch:
            points = batch["lidar"].to(self.device)
            if "is_lidar" not in batch:
                batch["is_lidar"] = torch.ones(*batch["lidar"].shape[:-1], 1, dtype=torch.bool, device=self.device)
            if "did_return" not in batch:
                batch["did_return"] = torch.ones(*batch["lidar"].shape[:-1], 1, dtype=torch.bool, device=self.device)

            ray_drop_logits = outputs["ray_drop_logits"]
            pred_depth = outputs["depth"]
            did_return = batch["did_return"][:, 0].to(self.device)
            is_lidar = batch["is_lidar"][:, 0].to(self.device)
            metrics_dict["depth_median_l2"] = float(
                self.median_l2(pred_depth[is_lidar][did_return], batch["distance"][did_return])
            )
            metrics_dict["depth_mean_rel_l2"] = float(
                self.mean_rel_l2(pred_depth[is_lidar][did_return], batch["distance"][did_return])
            )
            metrics_dict["intensity_rmse"] = float(self.rmse(outputs["intensity"][did_return], points[did_return, 3:4]))
            if self.config.ray_drop_loss_mult > 0.0:
                pred_points_did_return = ~(ray_drop_logits.sigmoid() > 0.5).squeeze(-1)
            else:
                pred_points_did_return = (pred_depth < self.config.non_return_lidar_distance).squeeze(-1)
            metrics_dict["ray_drop_accuracy"] = float((pred_points_did_return == did_return).float().mean())
            if pred_points_did_return.any() and points.shape[0] > 0 and did_return.any():
                pred_points = outputs["points"][is_lidar][pred_points_did_return]
                metrics_dict["chamfer_distance"] = float(
                    self.chamfer_distance(pred_points[..., :3], points[did_return, :3])
                )
            else:
                metrics_dict["chamfer_distance"] = points[did_return, :3].norm(dim=-1).mean()

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        if len(camera_ray_bundle.shape) == 1:  # lidar
            output_size, patch_size = (camera_ray_bundle.shape[0],), (1, 1)
            is_lidar = torch.ones_like(camera_ray_bundle.pixel_area, dtype=torch.bool)
        else:  # camera
            assert len(camera_ray_bundle.shape) == 2, "Raybundle should be 2d (an image/patch)"
            if self.config.compensate_upsampling_when_rendering:
                # shoot rays at a lower resolution and upsample the output to the target resolution
                step = self.config.rgb_upsample_factor
                camera_ray_bundle = camera_ray_bundle[step // 2 :: step, step // 2 :: step]
            output_size = patch_size = (
                camera_ray_bundle.shape[0],
                camera_ray_bundle.shape[1],
            )
            camera_ray_bundle = camera_ray_bundle.reshape((-1,))
            is_lidar = None

        camera_ray_bundle = self.collider(camera_ray_bundle)

        # Run chunked forward pass through NFF only
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.get_nff_outputs(ray_bundle, calc_lidar_losses=False)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(*output_size, -1)  # type: ignore

        features = outputs["features"].view(-1, outputs["features"].shape[-1])
        rgb, intensity, ray_drop_logit = self.decode_features(
            features, patch_size=patch_size, is_lidar=is_lidar, intensity_for_cam=True
        )
        if rgb is not None:
            outputs["rgb"] = rgb.squeeze(0)
        if intensity is not None:
            outputs["intensity"] = intensity.view(*output_size, -1)
        if ray_drop_logit is not None:
            outputs["ray_drop_logits"] = ray_drop_logit.view(*output_size, -1)
            outputs["ray_drop_prob"] = ray_drop_logit.view(*output_size, -1).sigmoid()
        return outputs

    @torch.no_grad()
    def _compute_is_close_to_lidar(self, sampling: UniSimSampling) -> None:
        """Compute which rays are close to the lidar."""
        for ray_samples in (
            sampling.static_ray_samples,
            sampling.sky_ray_samples,
            sampling.dynamic_ray_samples,
            sampling.ray_samples,
        ):
            if ray_samples is None:
                continue
            assert ray_samples.metadata is not None
            metadata, frustums = ray_samples.metadata, ray_samples.frustums
            mask = metadata["is_lidar"]
            if "did_return" in metadata:
                mask = mask & metadata["did_return"]
            mask_clone = mask.clone()
            # directions_norm, in case of lidar, is the distance
            start_dist = metadata["directions_norm"][mask] - frustums.starts[mask]
            end_dist = metadata["directions_norm"][mask] - frustums.ends[mask]
            dist = torch.min(start_dist.abs(), end_dist.abs())
            # if only one of them is negative, the point is in the frustum
            dist[(start_dist * end_dist).sign() < 0] = 0
            # same as mask[mask] since it will write to itself in place and break
            mask[mask_clone] = dist < self.config.regularization_epsilon
            metadata["is_close_to_lidar"] = mask

    def _scale_pixel_area(self, ray_bundle: RayBundle) -> RayBundle:
        is_lidar = ray_bundle.metadata.get("is_lidar")
        scaling = torch.ones_like(ray_bundle.pixel_area)
        if is_lidar is not None:
            scaling[~is_lidar] = self.config.rgb_upsample_factor**2
        else:
            scaling = self.config.rgb_upsample_factor**2
        ray_bundle.pixel_area = ray_bundle.pixel_area * scaling
        return ray_bundle

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.sampler._volumetric_sampler.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.static_field.alpha_fn(x),
                n=self.config.update_occ_grid_every_n,
                warmup_steps=0,
            )

        if self.sampler.config.use_nerfacc and self.config.update_occ_grid_every_n > 0:
            callback = [
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_occupancy_grid,
                ),
            ]
        else:
            callback = []

        return callback + super().get_training_callbacks(training_callback_attributes)


def render_depth_simple(
    weights: Float[Tensor, "*batch num_samples 1"],
    ray_samples: RaySamples,
    ray_indices: Optional[Int[Tensor, " num_samples"]] = None,
    num_rays: Optional[int] = None,
) -> Float[Tensor, "*batch 1"]:
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    return nerfacc.accumulate_along_rays(weights[..., 0], values=steps, ray_indices=ray_indices, n_rays=num_rays)
