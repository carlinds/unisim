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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.pipelines.ad_pipeline import ADPipeline, ADPipelineConfig
from nerfstudio.utils import profiler
from torch.nn import Parameter

from unisim.discriminators import ConvDiscriminator
from unisim.unisim import UniSimModel


@dataclass
class UniSimPipelineConfig(ADPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: UniSimPipeline)
    """target class to instantiate"""
    steps_per_stage: Tuple[int, int, int] = (2500, 7500, 10001)
    """specifies number of steps for each stage of training."""
    adversarial_loss_mult: float = 0.0  # off
    """Multiplier for adversarial loss"""
    remove_dynamic_points: bool = True
    """specifies whether to remove dynamic points from the lidar data used in the occupancy grid."""


class UniSimPipeline(ADPipeline):
    """Pipeline for training UniSim."""

    def __init__(self, config: UniSimPipelineConfig, **kwargs):
        # Override patch size to be 1x1 in the first phase
        self.phase1_patch_size = config.ray_patch_size
        config.ray_patch_size = (1, 1)
        super().__init__(config, **kwargs)

        self.config: UniSimPipelineConfig = config
        assert isinstance(self.model, UniSimModel)
        self.phase = -1  # Set this to undefined, so that all update_phase calls are propagated correctly
        # Initialize sampler
        self.model.sampler.initialize_occupancy(
            self.datamanager.get_accumulated_lidar_points(self.config.remove_dynamic_points)
        )
        self.discriminator = ConvDiscriminator().to(self.device) if self.config.adversarial_loss_mult > 0 else None

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self._update_phase(step)

        # This section a copy of super().get_train_loss_dict, since we need access to the ray_bundle
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle, patch_size=self.config.ray_patch_size)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if (actors := self.model.dynamic_actors).config.optimize_trajectories:
            pos_norm = (actors.actor_positions - actors.initial_positions).norm(dim=-1)
            metrics_dict["traj_opt_translation"] = pos_norm[pos_norm > 0].mean().nan_to_num()
            metrics_dict["traj_opt_rotation"] = (
                (actors.actor_rotations_6d - actors.initial_rotations_6d)[pos_norm > 0].norm(dim=-1).mean().nan_to_num()
            )
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if self.phase == 2 and self.discriminator:
            gan_losses = self._run_gan_step(ray_bundle=ray_bundle, model_outputs=model_outputs)
            loss_dict.update(gan_losses)
        return model_outputs, loss_dict, metrics_dict

    def _run_gan_step(self, ray_bundle: RayBundle, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Only image rays
        fake_ray_bundle = ray_bundle[~ray_bundle.metadata["is_lidar"][:, 0]]._apply_fn_to_fields(torch.detach)

        # Add noise to the ray origins
        fake_ray_bundle = fake_ray_bundle.reshape((-1, *self.config.ray_patch_size))
        fake_ray_bundle.origins = fake_ray_bundle.origins + torch.randn(
            (fake_ray_bundle.shape[0], 3), device=fake_ray_bundle.origins.device
        ).clamp(-3, 3).view(-1, 1, 1, 3)
        fake_ray_bundle = fake_ray_bundle.reshape((-1,))
        fake_ray_bundle = fake_ray_bundle.cat(
            [ray_bundle[ray_bundle.metadata["is_lidar"][:, 0]][:1]]
        )  # TODO: figure out why we need to send in at least one lidar ray to avoid CUDA errors

        fake_model_outputs = self._model(
            fake_ray_bundle, calc_lidar_losses=False, patch_size=self.config.ray_patch_size
        )

        rgb_real = model_outputs["rgb"]
        rgb_fake = fake_model_outputs["rgb"]
        discriminator_loss = self.discriminator.get_loss(rgb_real, rgb_fake)
        self.discriminator.requires_grad_(False)
        discriminator_out_fake = self.discriminator(rgb_fake.permute(0, 3, 1, 2))
        adversarial_loss = (
            F.binary_cross_entropy_with_logits(discriminator_out_fake, torch.ones_like(discriminator_out_fake))
            * self.config.adversarial_loss_mult
        )
        self.discriminator.requires_grad_(True)

        return {"discriminator_loss": discriminator_loss, "adversarial_loss": adversarial_loss}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if self.discriminator:
            param_groups["discriminator"] = list(self.discriminator.parameters())
        return param_groups

    def _update_phase(self, step: int):
        new_phase = (
            0 if step < self.config.steps_per_stage[0] else 1 if step < sum(self.config.steps_per_stage[:2]) else 2
        )

        if self.phase == new_phase:
            return

        if new_phase == 1:
            self.config.ray_patch_size = self.phase1_patch_size
            self.datamanager.change_patch_sampler(
                patch_scale=self.model.config.rgb_upsample_factor,
                patch_size=self.config.ray_patch_size[0],
            )
        self.model.update_phase(new_phase)

        self.phase = new_phase
