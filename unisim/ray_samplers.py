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
UniSim's ray sampling strategies.
"""

from dataclasses import dataclass, field
from math import prod
from typing import Any, Optional, Type, Union

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from nerfstudio.cameras.lidars import transform_points_pairwise
from nerfstudio.cameras.rays import RayBundle, RaySamples, merge_raysamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.model_components.dynamic_actors import DynamicActors
from nerfstudio.model_components.ray_samplers import AlphaFn, LinearDisparitySampler, Sampler, VolumetricSampler
from nerfstudio.utils.math import intersect_aabb_batch
from nerfstudio.viewer.server.viewer_elements import ViewerCheckbox, ViewerSlider
from torch import Tensor

from unisim.occupancy_grids import OccupancyGrid


class EqualSpaceSampler(Sampler):
    """A sampler that samples rays equally spaced in Euclidean space."""

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
        )
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        sample_spacing: float = 1.0,
    ) -> Any:
        """Generates position samples according to sample spacing.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray
            sample_spacing: Spacing between samples

        Returns:
            Positions and deltas for samples along a ray
        """

        assert ray_bundle is not None
        assert ray_bundle.nears is not None

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1, device=ray_bundle.origins.device)[
            None, ...
        ]  # [1, num_samples+1]
        bins = bins * (sample_spacing * num_samples)  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        # add nears to bins
        bins = bins + ray_bundle.nears

        ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[..., :-1, None], bin_ends=bins[..., 1:, None])

        return ray_samples

    def forward(self, *args, **kwargs) -> Any:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class ActorSampler(Sampler):
    """Sampler that samples rays from a set of (dynamic) actors."""

    def __init__(self, actors: DynamicActors, stepsize: float, single_jitter: bool = False):
        super().__init__()
        self.actors = actors
        self.stepsize = stepsize
        self._uniform_sampler = EqualSpaceSampler(single_jitter=single_jitter)

        self.dynamic_scene_stepsize = ViewerSlider(
            name="Dynamic Stepsize (m)",
            default_value=stepsize,
            min_value=0.01,
            max_value=1.0,
            step=0.01,
        )

    def generate_ray_samples(self, ray_bundle: RayBundle) -> Union[RaySamples, None]:
        if self.actors.n_actors == 0:
            return None
        assert ray_bundle.times is not None

        world2boxes, ray_indices, object_indices = self.actors.get_world2boxes(ray_bundle.times)

        object_sizes = self.actors.actor_sizes[object_indices] + 2 * torch.tensor(
            self.actors.config.actor_bbox_padding, device=self.actors.actor_sizes.device
        )

        aabb = torch.cat([-object_sizes / 2.0, object_sizes / 2.0], dim=-1)  # [num_rays*n_valid_objects, 6]

        with torch.no_grad():
            ray_origs_box = transform_points_pairwise(ray_bundle.origins[ray_indices], world2boxes)
            ray_dirs_box = transform_points_pairwise(
                ray_bundle.directions[ray_indices], world2boxes, with_translation=False
            )
        ray_dirs_box = ray_dirs_box / torch.norm(
            ray_dirs_box, dim=-1, keepdim=True
        )  # renormalize in case of numerical errors

        max_bound = 1e10
        t_min, t_max = intersect_aabb_batch(
            ray_origs_box.unsqueeze(1), ray_dirs_box.unsqueeze(1), aabb, max_bound=max_bound, invalid_value=max_bound
        )
        t_min = t_min.squeeze(-1)
        t_max = t_max.squeeze(-1)
        hits = (t_min != max_bound) & (t_max != max_bound)
        if hits.sum() == 0:
            return None

        hits_idx = hits.nonzero(as_tuple=False).squeeze(-1)
        ray_indices = ray_indices[hits_idx]
        object_indices = object_indices[hits_idx]
        world2boxes = world2boxes[hits_idx]
        t_min = t_min[hits_idx]
        t_max = t_max[hits_idx]

        rays_to_sample = ray_bundle[ray_indices]
        rays_to_sample.nears = t_min.unsqueeze(-1)
        dynamic_n_samples = ((t_max - t_min) / self.dynamic_scene_stepsize.value).ceil().int().max()
        dynamic_n_samples = dynamic_n_samples.clamp(min=1)
        samples = self._uniform_sampler(
            ray_bundle=rays_to_sample,
            num_samples=dynamic_n_samples,
            sample_spacing=self.dynamic_scene_stepsize.value,
        )

        actor_sorting = torch.argsort(object_indices)

        samples.metadata["ray_indices"] = ray_indices.view(-1, 1, 1).expand(-1, dynamic_n_samples, 1).contiguous()
        samples.metadata["object_indices"] = object_indices.view(-1, 1, 1).expand(-1, dynamic_n_samples, 1).contiguous()
        samples.metadata["symmetric"] = self.actors.actor_symmetric[samples.metadata["object_indices"]]
        samples.metadata["deformable"] = self.actors.actor_deformable[samples.metadata["object_indices"]]
        samples.metadata["world2box"] = world2boxes.view(-1, 1, 3, 4).expand(-1, dynamic_n_samples, 3, 4).contiguous()
        samples = samples[actor_sorting]
        samples_in_range_mask = samples.frustums.ends.squeeze(-1) < t_max[actor_sorting].unsqueeze(-1)
        samples = samples[samples_in_range_mask]
        return samples if prod(samples.shape) > 0 else None


@dataclass
class UniSimSampling:
    """Sampling result for UniSim."""

    ray_samples: RaySamples
    ordering: torch.Tensor
    static_ray_samples: Optional[RaySamples]
    sky_ray_samples: Optional[RaySamples]
    dynamic_ray_samples: Optional[RaySamples]
    sample_types: torch.Tensor
    bbox_padding: torch.Tensor

    @property
    def is_static(self) -> Tensor:
        return self.sample_types == 0

    @property
    def is_sky(self) -> Tensor:
        return self.sample_types == 1

    @property
    def is_dynamic(self) -> Tensor:
        return self.sample_types == 2

    @property
    def num_static(self) -> int:
        return prod(self.static_ray_samples.shape) if self.static_ray_samples is not None else 0

    @property
    def num_sky(self) -> int:
        return prod(self.sky_ray_samples.shape) if self.sky_ray_samples is not None else 0

    @property
    def num_dynamic(self) -> int:
        return prod(self.dynamic_ray_samples.shape) if self.dynamic_ray_samples is not None else 0


@dataclass
class UniSimSamplerConfig(InstantiateConfig):
    """Configuration of UniSim's sampler."""

    _target: Type = field(default_factory=lambda: UniSimSampler)

    occupancy_voxel_size: float = 0.5
    """Size of each voxel in the occupancy grid (in meters)."""
    occupancy_dilation_kernel_size: int = 2  # TODO: tune this
    """Size of the dilation kernel for the occupancy grid."""
    static_scene_stepsize: float = 0.2
    """How tightly the static scene should be sampled (in meters)."""
    dynamic_scene_stepsize: float = 0.05
    """How tightly the dynamic (actor) scene should be sampled (in meters)."""
    num_sky_samples: int = 16
    """Number of samples to use when sampling the sky (how? we simply cannot know)."""
    max_sky_distance: float = 3000.0
    """Maximum distance to sample the sky (in meters)."""
    use_nerfacc: bool = True
    """Use nerfacc for sampling rays. Slightly faster but not tested as much."""
    use_single_jitter: bool = False
    """Use the same random jitter for all samples in the same ray (in training)."""
    dont_sample_static_inside_actors: bool = False
    """Whether to sample the static scene inside actor boxes or not."""
    dont_sample_sky_under_ground: bool = False
    """Whether to sample the static scene under the ground or not."""


class UniSimSampler(Sampler):
    """The sampler used for UniSim.

    It samples the static scene every 20cm and samples the dynamic scene every 5cm.
    Static samples are discarded based on accupancy.
    Finally, a sky field is sampled uniformly in disparity (from the end of the static scene).
    """

    config: UniSimSamplerConfig

    def __init__(
        self,
        config: UniSimSamplerConfig,
        aabb: Float[Tensor, "2 3"],
        dynamic_actors: DynamicActors,
        alpha_fn: Optional[AlphaFn] = None,
    ):
        """
        Args:
            config: The sampler configuration settings.
            aaab: The axis-aligned bounding box of the scene.
            dynamic_actors: The dynamic actors in the scene.

        """
        super().__init__()
        self.config = config
        self.occ_grid = OccupancyGrid(
            aabb=aabb,
            voxel_size=self.config.occupancy_voxel_size,
            dilation_kernel_size=self.config.occupancy_dilation_kernel_size,
        )
        self.actor_sampler = ActorSampler(
            dynamic_actors, self.config.dynamic_scene_stepsize, self.config.use_single_jitter
        )

        self.static_scene_stepsize = ViewerSlider(
            name="Static Stepsize (m)",
            default_value=self.config.static_scene_stepsize,
            min_value=0.01,
            max_value=1.0,
            step=0.01,
        )
        self._uniform_sampler = EqualSpaceSampler(single_jitter=self.config.use_single_jitter)
        self._sky_sampler = LinearDisparitySampler(
            single_jitter=self.config.use_single_jitter, num_samples=self.config.num_sky_samples
        )
        if self.config.use_nerfacc:
            self._volumetric_sampler = VolumetricSampler(
                occupancy_grid=OccGridEstimator(
                    roi_aabb=self.occ_grid.aabb.flatten(), resolution=list(self.occ_grid.grid.shape), levels=1
                ),
                alpha_fn=alpha_fn,
            )
            self._volumetric_sampler.occupancy_grid.binaries = self.occ_grid.grid.unsqueeze(0)
            self._volumetric_sampler.occupancy_grid.occs = (
                self._volumetric_sampler.occupancy_grid.binaries.float().view(-1)
            )
        self.static_enabled = ViewerCheckbox("Sample Static Field", True)
        self.sky_enabled = ViewerCheckbox("Sample Sky Field", True)
        self.dynamic_enabled = ViewerCheckbox("Sample Dynamic Field(s)", True)

    def initialize_occupancy(self, lidar_points):
        """Initialize the occupancy grid."""
        self.occ_grid.populate_occupancy(lidar_points)
        if self.config.use_nerfacc:
            self._volumetric_sampler.occupancy_grid.binaries = self.occ_grid.grid.unsqueeze(0)
            self._volumetric_sampler.occupancy_grid.occs = (
                self._volumetric_sampler.occupancy_grid.binaries.float().view(-1)
            )

    def generate_ray_samples(self, ray_bundle: RayBundle) -> UniSimSampling:
        """Sample the ray bundle."""
        assert ray_bundle.fars is not None and ray_bundle.nears is not None, "RayBundle must have nears and fars."

        # Always sample everything during training
        static_ray_samples = self._sample_static(ray_bundle) if (self.training or self.static_enabled.value) else None
        sky_ray_samples = self._sample_sky(ray_bundle) if (self.training or self.sky_enabled.value) else None
        dyn_ray_samples = self.actor_sampler(ray_bundle) if (self.training or self.dynamic_enabled.value) else None

        # Merge and determine ordering
        ray_samples = [s for s in [static_ray_samples, sky_ray_samples, dyn_ray_samples] if s is not None]
        ray_samples, ordering = merge_raysamples(ray_samples, sort=True)
        sample_types = torch.cat(
            [
                torch.full(samples.shape, type_val, dtype=torch.int32, device=samples.frustums.origins.device)
                for type_val, samples in enumerate([static_ray_samples, sky_ray_samples, dyn_ray_samples])
                if samples is not None
            ]
        )[ordering]
        sampling = UniSimSampling(
            ray_samples,
            ordering,
            static_ray_samples,
            sky_ray_samples,
            dyn_ray_samples,
            sample_types=sample_types,
            bbox_padding=torch.tensor(
                self.actor_sampler.actors.config.actor_bbox_padding, device=ray_bundle.origins.device
            ),
        )
        if self.config.dont_sample_static_inside_actors and sampling.num_static > 0:
            self._drop_interleaved_static_samples_(sampling)
        return sampling

    def _drop_interleaved_static_samples_(self, sampling: UniSimSampling):
        device = sampling.sample_types.device
        padded_is_dynamic = torch.cat(
            (torch.tensor([True], device=device), sampling.is_dynamic, torch.tensor([True], device=device))
        )  # padding with True so that static sample is dropped if ray begins/ends inside object
        is_static_inside_dynamic = sampling.is_static & padded_is_dynamic[:-2] & padded_is_dynamic[2:]
        indices_to_keep = torch.where(~is_static_inside_dynamic)[0]

        sampling.sample_types = sampling.sample_types[indices_to_keep]
        sampling.static_ray_samples = sampling.ray_samples[indices_to_keep][sampling.sample_types == 0]
        sampling.ray_samples, sampling.ordering = merge_raysamples(
            [
                s
                for s in [sampling.static_ray_samples, sampling.sky_ray_samples, sampling.dynamic_ray_samples]
                if s is not None
            ],
            sort=True,
        )  # need to reorder because old ordering is broken now

    def _sample_static(self, ray_bundle: RayBundle) -> Optional[RaySamples]:
        if self.config.use_nerfacc:
            samples = self._sample_static_nerfacc(ray_bundle)
        else:
            samples = self._sample_static_occgrid(ray_bundle)
        return samples if prod(samples.shape) > 0 else None

    def _sample_static_nerfacc(self, ray_bundle: RayBundle) -> RaySamples:
        ray_samples, ray_indices = self._volumetric_sampler(
            ray_bundle, render_step_size=self.static_scene_stepsize.value
        )
        metadata = ray_samples.metadata if ray_samples.metadata is not None else dict()
        metadata.update({k: v[ray_indices].view(-1, v.shape[-1]) for k, v in ray_bundle.metadata.items()})
        ray_samples.metadata = metadata
        ray_samples.metadata["ray_indices"] = ray_indices.view(-1, 1)
        return ray_samples

    def _sample_static_occgrid(self, ray_bundle: RayBundle) -> RaySamples:
        num_rays, device = ray_bundle.shape[0], ray_bundle.origins.device
        # Sample the static scene every 20cm between nears and fars (which are already set based on scene box).
        static_n_samples = ((ray_bundle.fars - ray_bundle.nears) / self.static_scene_stepsize.value).ceil().int().max()
        static_ray_samples = self._uniform_sampler(
            ray_bundle=ray_bundle, num_samples=static_n_samples, sample_spacing=self.static_scene_stepsize.value
        )
        ray_indices = torch.arange(num_rays, device=device).view(-1, 1, 1).repeat(1, static_n_samples, 1)
        static_ray_samples.metadata["ray_indices"] = ray_indices
        # Reject all samples outside occupancy. TODO: maybe more efficient to check occupancy before this?
        static_sample_occupancies = self.occ_grid.is_occupied(static_ray_samples.frustums.get_positions().detach())
        static_ray_samples = static_ray_samples[static_sample_occupancies]
        return static_ray_samples

    def _sample_sky(self, ray_bundle: RayBundle) -> Optional[RaySamples]:
        num_rays, device = ray_bundle.shape[0], ray_bundle.origins.device
        # TODO: is this copy too slow?
        sky_ray_bundle = ray_bundle[:]
        sky_ray_bundle.nears = ray_bundle.fars
        sky_ray_bundle.fars = torch.full_like(ray_bundle.fars, self.config.max_sky_distance)
        sky_ray_samples = self._sky_sampler(ray_bundle=sky_ray_bundle)
        sky_ray_samples.metadata["ray_indices"] = (
            torch.arange(num_rays, device=device).view(-1, 1, 1).repeat(1, self.config.num_sky_samples, 1)
        )
        sky_ray_samples = sky_ray_samples.flatten()
        if self.config.dont_sample_sky_under_ground:
            frustums = sky_ray_samples.frustums
            sample_zs = (frustums.origins + (frustums.directions * frustums.starts))[:, 2]
            sky_ray_samples = sky_ray_samples[sample_zs > self.occ_grid.aabb[0, 2]]

        return sky_ray_samples if prod(sky_ray_samples.shape) > 0 else None
