from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.ad_datamanager import ADDataManagerConfig
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components.dynamic_actors import DynamicActorsConfig
from nerfstudio.plugins.types import MethodSpecification

from unisim.unisim import UniSimModelConfig
from unisim.unisim_pipeline import UniSimPipelineConfig

unisim = MethodSpecification(
    config=TrainerConfig(
        method_name="unisim",
        steps_per_eval_batch=500,
        steps_per_eval_all_images=5000,
        steps_per_save=2000,
        max_num_iterations=20001,
        mixed_precision=True,
        pipeline=UniSimPipelineConfig(
            datamanager=ADDataManagerConfig(
                dataparser=PandaSetDataParserConfig(
                    correct_cuboid_time=False,
                    allow_per_point_times=False,
                    rolling_shutter_offsets=(0.0, 0.0),
                    cameras=("front",),
                    add_missing_points=False,
                ),
                train_num_lidar_rays_per_batch=8192,
            ),
            steps_per_stage=(2500, 7500, 10001),
            model=UniSimModelConfig(
                static_use_gauss_field=False,
                dynamic_use_gauss_field=False,
                rgb_loss_mult=1.0,
                eval_num_rays_per_chunk=1 << 15,
                dynamic_actors=DynamicActorsConfig(),
                camera_optimizer=CameraOptimizerConfig(mode="off"),  # SO3xR3
                traj_opt_start_phase=1,
            ),
            # adversarial_loss_mult=0.001,
        ),
        optimizers={
            "trajectory_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001),
            },
            "discriminator": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=20001),
            },
            "cnn": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20001),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="UniSim, as described in the paper.",
)

unisim_plusplus = MethodSpecification(
    config=TrainerConfig(
        method_name="unisim++",
        steps_per_eval_batch=500,
        steps_per_eval_all_images=5000,
        steps_per_save=2000,
        max_num_iterations=20001,
        mixed_precision=True,
        pipeline=UniSimPipelineConfig(
            datamanager=ADDataManagerConfig(
                dataparser=PandaSetDataParserConfig(),
                train_num_lidar_rays_per_batch=8192,
            ),
            model=UniSimModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),  # SO3xR3
            ),
        ),
        optimizers={
            "trajectory_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
            },
            "cnn": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20001, warmup_steps=500),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="UniSim with some tweaks inspired by NeuRAD.",
)
