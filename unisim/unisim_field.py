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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from contextlib import nullcontext
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.model_components.utils import SigmoidDensity
from nerfstudio.utils.math import GaussiansStd, erf_approx
from torch import Tensor, nn

EPS = 1.0e-7


class UniSimField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        inverted_sphere: whether to use inverted sphere spatial distortion
        nff_out_dim: output dimension of the neural feature field
        hashgrid_dim: number of features per level (for the hash grid)


    """

    def __init__(
        self,
        num_images: int,
        aabb: Optional[Tensor] = None,
        num_layers: int = 2,  # TODO: 3?
        hidden_dim: int = 32,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 4096,
        log2_hashmap_size: int = 21,
        num_layers_color: int = 3,  # TODO: 4?
        inverted_sphere: bool = False,
        nff_out_dim: int = 32,
        hashgrid_dim: int = 2,
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        numerical_gradients_delta: float = 1e-3,
        use_numerical_gradients: bool = True,
        sdf_to_density: nn.Module = SigmoidDensity(10.0),
        regularize_hash_function: Callable[[Tensor], Tensor] = torch.square,
        sdf_offset: float = 0.0,
        learn_sdf_offset: bool = False,
        model_lidar_return_prob: bool = False,
    ) -> None:
        super().__init__()
        self.regularize_hash_function = regularize_hash_function
        self.sdf_offset = nn.Parameter(torch.tensor(sdf_offset), requires_grad=learn_sdf_offset)
        self.model_lidar_return_prob = model_lidar_return_prob

        if aabb is not None:
            self.register_buffer("aabb", aabb)
        else:
            self.aabb = None
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.register_buffer("numerical_gradients_delta", torch.tensor(numerical_gradients_delta))

        self.num_images = num_images
        self.base_res = base_res
        self.use_inverted_sphere = inverted_sphere
        self.use_numerical_gradients = use_numerical_gradients
        self.geo_feat_dim = hidden_dim - 1

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )
        self.hash_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=hashgrid_dim,
            implementation=implementation,
        )
        mlp_base_implementation = "torch" if not use_numerical_gradients else implementation
        self.mlp_base = MLP(
            in_dim=self.hash_encoding.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=mlp_base_implementation,
        )
        mlp_head_out_dim = nff_out_dim + 1 if self.model_lidar_return_prob else nff_out_dim
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim,
            out_dim=mlp_head_out_dim,
            activation=nn.ReLU(),
            out_activation=out_activation,
            implementation=implementation,
        )

        self.sdf_to_density = sdf_to_density

    def _get_numerical_gradients_delta(self) -> torch.Tensor:
        delta = self.numerical_gradients_delta  # self.hash_encoding.scalings[-1]
        return delta

    def set_numerical_gradients_delta(self, delta: float) -> None:
        self.numerical_gradients_delta = torch.tensor(delta)

    def get_sdf_grad(
        self, sample_locations: Float[Tensor, "*batch 3"], signed_distances: Float[Tensor, "*batch 1"]
    ) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of sdf gradients."""
        if sample_locations.nelement() == 0:
            return torch.empty_like(sample_locations)
        if self.use_numerical_gradients:
            device = sample_locations.device
            delta = self._get_numerical_gradients_delta() / np.sqrt(3)
            k1 = torch.as_tensor([1, -1, -1], device=device, dtype=sample_locations.dtype)
            k2 = torch.as_tensor([-1, -1, 1], device=device, dtype=sample_locations.dtype)
            k3 = torch.as_tensor([-1, 1, -1], device=device, dtype=sample_locations.dtype)
            k4 = torch.as_tensor([1, 1, 1], device=device, dtype=sample_locations.dtype)

            points = torch.stack(
                [
                    sample_locations + k1 * delta,
                    sample_locations + k2 * delta,
                    sample_locations + k3 * delta,
                    sample_locations + k4 * delta,
                ],
                dim=-2,
            )
            points_sdf = self.geonetwork_forward(points)[0]
            gradients = (
                k1 * points_sdf[..., 0, :]
                + k2 * points_sdf[..., 1, :]
                + k3 * points_sdf[..., 2, :]
                + k4 * points_sdf[..., 3, :]
            ) / (4 * delta)
        else:
            gradients = torch.autograd.grad(
                outputs=signed_distances,
                inputs=sample_locations,
                grad_outputs=torch.ones_like(signed_distances, requires_grad=False),
                retain_graph=True,
                create_graph=True,
            )[0]

        return gradients

    def geonetwork_forward(self, x: Tensor, override_grid_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Computes and returns the signed distance value and the base mlp output."""
        if override_grid_features is not None:
            grid_feats = override_grid_features.view(-1, self.mlp_base.in_dim)
        else:
            grid_feats = self.hash_encoding(x.view(-1, 3))
        h = self.mlp_base(grid_feats)
        h = h.view(*x.shape[:-1], 1 + self.geo_feat_dim)
        signed_distance, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        signed_distance = signed_distance + self.sdf_offset
        return signed_distance, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = dict()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        feature = self.mlp_head(h).view(*outputs_shape, -1).to(directions)  # TODO: relu?
        if self.model_lidar_return_prob:
            lidar_return_prob, feature = torch.split(feature, [1, feature.shape[-1] - 1], dim=-1)
            lidar_return_prob = lidar_return_prob.view(*outputs_shape, 1).sigmoid()
            outputs[FieldHeadNames.LIDAR_RETURN_PROB] = lidar_return_prob
        outputs[FieldHeadNames.FEATURE] = feature
        return outputs

    def forward(
        self,
        ray_samples: RaySamples,
        positions: Optional[Float[Tensor, "*batch 3"]] = None,
        compute_normals: bool = False,
        override_grid_features: Optional[Tensor] = None,
        aabb: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if positions is None:
            positions: Float[Tensor, "*batch 3"] = self.get_positions(ray_samples, aabb)

        maybe_grad = torch.enable_grad() if compute_normals and not self.use_numerical_gradients else nullcontext()
        with maybe_grad:
            signed_distance, geo_embedding = self.geonetwork_forward(positions, override_grid_features)
            # signed_distance = signed_distance.to(positions)  # TODO: Needed?
        # TODO: neurangelo?
        alpha = self.sdf_to_density(signed_distance)

        field_outputs = self.get_outputs(ray_samples, density_embedding=geo_embedding)
        field_outputs[FieldHeadNames.SDF] = signed_distance
        field_outputs[FieldHeadNames.ALPHA] = alpha

        if compute_normals:
            with maybe_grad:
                # During training we only compute gradients for lidar rays (for speed).
                if self.training and ray_samples.metadata and "is_close_to_lidar" in ray_samples.metadata:
                    mask = ray_samples.metadata["is_close_to_lidar"][..., 0]
                    positions = (
                        positions[mask] if self.use_numerical_gradients else positions
                    )  # analytical gradients break if we mask out points
                    signed_distance = signed_distance[mask] if self.use_numerical_gradients else signed_distance
                else:
                    mask = torch.ones_like(signed_distance, dtype=torch.bool)
                gradients = self.get_sdf_grad(positions, signed_distance)
                gradients = gradients[mask] if not self.use_numerical_gradients else gradients
                normals = -gradients / (1e-8 + torch.linalg.norm(gradients, dim=-1, keepdim=True))
            field_outputs[FieldHeadNames.NORMALS] = normals
            field_outputs[FieldHeadNames.GRADIENT] = gradients
        return field_outputs

    def get_positions(self, ray_samples: RaySamples, aabb: Optional[Tensor] = None) -> Float[Tensor, "*batch 3"]:
        if self.use_inverted_sphere:
            positions = ray_samples.frustums.get_positions()
            # go from x,y,z to azimuth, elevation, disparity and normalize
            distance = torch.linalg.norm(positions, dim=-1)
            # azimuth has range [-pi, pi] -> [0, 1]
            azimuth = torch.atan2(positions[..., 1], positions[..., 0]) / (2 * torch.pi) + 0.5
            # elevation has range [0, pi] -> [0, 1]
            elevation = torch.asin(positions[..., 2] / distance) / torch.pi + 0.5
            disparity = 1 / distance
            positions = torch.stack([azimuth, elevation, disparity], dim=-1)
        else:
            aabb = aabb if aabb is not None else self.aabb
            assert aabb is not None, "AABB must be provided in __init__ or forward."
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        positions = torch.clamp(positions, 0.0, 1.0)
        positions.requires_grad_()
        return positions

    def alpha_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None
    ) -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        del times
        # Need to figure out a better way to describe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]) + EPS,
                pixel_area=torch.ones_like(positions[..., :1]),
            )
        )
        pos_emb = self.get_positions(ray_samples)
        sdf, _ = self.geonetwork_forward(pos_emb)
        sdf = sdf + self.sdf_offset
        alpha = self.sdf_to_density(sdf)
        return alpha


class GaussianUniSimField(UniSimField):
    def __init__(
        self,
        num_images: int,
        aabb: Optional[Tensor] = None,
        num_layers: int = 2,  # TODO: 3?
        hidden_dim: int = 32,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 4096,
        log2_hashmap_size: int = 21,
        num_layers_color: int = 3,  # TODO: 4?
        inverted_sphere: bool = False,
        nff_out_dim: int = 32,
        hashgrid_dim: int = 2,
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        numerical_gradients_delta: float = 1e-3,
        use_numerical_gradients: bool = True,
        sdf_to_density: nn.Module = SigmoidDensity(10.0),
        regularize_hash_function: Callable[[Tensor], Tensor] = torch.square,
        sdf_offset: float = 0.0,
        learn_sdf_offset: bool = False,
        model_lidar_return_prob: bool = False,
    ):
        super().__init__(
            num_images=num_images,
            aabb=aabb,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            inverted_sphere=inverted_sphere,
            nff_out_dim=nff_out_dim,
            hashgrid_dim=hashgrid_dim,
            out_activation=out_activation,
            implementation=implementation,
            numerical_gradients_delta=numerical_gradients_delta,
            use_numerical_gradients=use_numerical_gradients,
            sdf_to_density=sdf_to_density,
            regularize_hash_function=regularize_hash_function,
            sdf_offset=sdf_offset,
            learn_sdf_offset=learn_sdf_offset,
            model_lidar_return_prob=model_lidar_return_prob,
        )
        self.hashgrid_dim = hashgrid_dim

    def _get_normalized_mean_and_std(self, gaussians: GaussiansStd, aabb: Tensor) -> Tuple[Tensor, Tensor]:
        means = SceneBox.get_normalized_positions(gaussians.mean, aabb, per_dim_norm=False)
        # aabb = aabb.unsqueeze(1) if len(aabb.shape) > 2 else aabb
        aabb_lengths = aabb[..., 1, :] - aabb[..., 0, :]
        aabb_lengths = aabb_lengths.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        std = gaussians.std / aabb_lengths

        return means, std

    def get_positions(
        self,
        ray_samples: RaySamples,
        aabb: Optional[Tensor] = None,
        multisampled: bool = False,
        normalized: bool = True,
    ) -> GaussiansStd:
        if multisampled:
            gaussians = ray_samples.frustums.get_multisampled_gaussian_blob(rand=self.training)
        else:
            gaussians = ray_samples.frustums.get_conical_gaussian_blob()
            gaussians = gaussians.to_std()
            gaussians.mean = gaussians.mean.unsqueeze(1)
            # collapse to isotropic gaussian using geometric mean
            gaussians.std = gaussians.std.prod(dim=-1, keepdim=True).pow(1 / 3).unsqueeze(1)

        if self.use_inverted_sphere:
            means = gaussians.mean
            std = gaussians.std
            # go from x,y,z to azimuth, elevation, disparity and normalize
            distance = torch.linalg.norm(means, dim=-1)
            # azimuth has range [-pi, pi] -> [0, 1]
            azimuth = torch.atan2(means[..., 1], means[..., 0]) / (2 * torch.pi) + 0.5
            # elevation has range [0, pi] -> [0, 1]
            elevation = torch.acos(means[..., 2] / distance) / torch.pi
            disparity = 1 / distance
            means = torch.stack([azimuth, elevation, disparity], dim=-1)
        else:
            aabb = aabb if aabb is not None else self.aabb
            assert aabb is not None, "AABB must be provided in __init__ or forward."
            if normalized:
                means, std = self._get_normalized_mean_and_std(gaussians, aabb)
            else:
                means = gaussians.mean
                std = gaussians.std

        # Make sure the tcnn gets inputs between 0 and 1.
        means = torch.clamp(means, 0.0, 1.0) if normalized else means
        # means.requires_grad_() # this gives OOM, why?
        return GaussiansStd(means, std)

    def forward(
        self,
        ray_samples: RaySamples,
        positions: Optional[GaussiansStd] = None,
        compute_normals: bool = False,
        override_grid_features: Optional[Tensor] = None,
        aabb: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if positions is None:
            positions: GaussiansStd = self.get_positions(ray_samples, aabb)

        maybe_grad = torch.enable_grad() if compute_normals and not self.use_numerical_gradients else nullcontext()
        with maybe_grad:
            signed_distance, geo_embedding = self.geonetwork_forward(positions, override_grid_features)
            # signed_distance = signed_distance.to(positions)  # TODO: Needed?
        # TODO: neurangelo?
        alpha = self.sdf_to_density(signed_distance)

        field_outputs = self.get_outputs(ray_samples, density_embedding=geo_embedding)
        field_outputs[FieldHeadNames.SDF] = signed_distance
        field_outputs[FieldHeadNames.ALPHA] = alpha

        if compute_normals:
            with maybe_grad:
                # During training we only compute gradients for lidar rays (for speed).
                if self.training and ray_samples.metadata and "is_close_to_lidar" in ray_samples.metadata:
                    mask = ray_samples.metadata["is_close_to_lidar"][..., 0]
                    positions = (
                        positions[mask] if self.use_numerical_gradients else positions
                    )  # analytical gradients break if we mask out points
                    signed_distance = signed_distance[mask] if self.use_numerical_gradients else signed_distance
                else:
                    mask = torch.ones_like(signed_distance, dtype=torch.bool)
                gradients = self.get_sdf_grad(
                    GaussiansStd(
                        mean=positions.mean.mean(dim=-2),
                        std=positions.std.pow(2).sum(dim=-2).sqrt() / positions.std.shape[-1],
                    ),
                    signed_distance,
                )
                gradients = gradients[mask] if not self.use_numerical_gradients else gradients
                normals = -gradients / (1e-8 + torch.linalg.norm(gradients, dim=-1, keepdim=True))
            field_outputs[FieldHeadNames.NORMALS] = normals
            field_outputs[FieldHeadNames.GRADIENT] = gradients
        return field_outputs

    def weighted_positional_enc(self, x: GaussiansStd) -> Tensor:
        prefix_shape = list(x.mean.shape[:-1])
        grid_feats = (
            self.hash_encoding(x.mean.view(-1, 3))
            .view(prefix_shape + [self.hash_encoding.num_levels * self.hash_encoding.features_per_level])  # type: ignore
            .unflatten(-1, (self.hash_encoding.num_levels, self.hash_encoding.features_per_level))
        )  # [..., "n_samples", "num_levels", "features_per_level"]
        weights = erf_approx(
            1 / (8**0.5 * x.std * self.hash_encoding.scalings.view(-1)).abs().clamp_min(EPS)  # type: ignore
        )  # [..., "n_samples", "num_levels"]
        grid_feats = (
            (grid_feats * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        )  # [..., "n_samples", "num_levels * features_per_level"]
        grid_feats = grid_feats.view(-1, self.hash_encoding.get_out_dim())
        return grid_feats

    def geonetwork_forward(
        self, x: GaussiansStd, override_grid_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes and returns the signed distance value and the base mlp output."""
        if override_grid_features is not None:
            grid_feats = override_grid_features.view(-1, self.mlp_base.in_dim)
        else:
            grid_feats = self.weighted_positional_enc(x)
        h = self.mlp_base(grid_feats)
        h = h.view(*x.mean.shape[:-2], 1 + self.geo_feat_dim)
        signed_distance, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        signed_distance = signed_distance + self.sdf_offset
        return signed_distance, base_mlp_out

    def get_sdf_grad(
        self, sample_gaussians: GaussiansStd, signed_distances: Float[Tensor, "*batch 1"]
    ) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of sdf gradients."""
        sample_locations = sample_gaussians.mean
        if sample_locations.nelement() == 0:
            return torch.empty_like(sample_locations)
        if self.use_numerical_gradients:
            device = sample_locations.device
            delta = self._get_numerical_gradients_delta() / np.sqrt(3)
            k1 = torch.as_tensor([1, -1, -1], device=device, dtype=sample_locations.dtype)
            k2 = torch.as_tensor([-1, -1, 1], device=device, dtype=sample_locations.dtype)
            k3 = torch.as_tensor([-1, 1, -1], device=device, dtype=sample_locations.dtype)
            k4 = torch.as_tensor([1, 1, 1], device=device, dtype=sample_locations.dtype)

            points = torch.stack(
                [
                    sample_locations + k1 * delta,
                    sample_locations + k2 * delta,
                    sample_locations + k3 * delta,
                    sample_locations + k4 * delta,
                ],
                dim=-2,
            )
            points_sdf = self.geonetwork_forward(
                GaussiansStd(
                    mean=points.unsqueeze(2), std=sample_gaussians.std.unsqueeze(2).repeat(1, 4, 1).unsqueeze(2)
                )
            )[0]  # unsqueeze(2) to make dummy "number of samples" dim
            gradients = (
                k1 * points_sdf[..., 0, :]
                + k2 * points_sdf[..., 1, :]
                + k3 * points_sdf[..., 2, :]
                + k4 * points_sdf[..., 3, :]
            ) / (4 * delta)
        else:
            gradients = torch.autograd.grad(
                outputs=signed_distances,
                inputs=sample_locations,
                grad_outputs=torch.ones_like(signed_distances, requires_grad=False),
                retain_graph=True,
                create_graph=True,
            )[0]

        return gradients
