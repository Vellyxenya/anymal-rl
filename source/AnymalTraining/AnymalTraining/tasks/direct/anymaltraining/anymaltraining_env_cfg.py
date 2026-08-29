# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class AnymaltrainingSceneCfg(InteractiveSceneCfg):
    """Scene configuration using the template-based spawning path."""

    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class AnymaltrainingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # scene
    scene: AnymaltrainingSceneCfg = AnymaltrainingSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # Joint-position commands are offsets from ANYmal's default standing pose.
    action_scale = 0.5  # [rad]
    target_base_height = 0.6  # [m]
    min_base_height = 0.35  # [m]
    max_base_tilt = 0.7  # norm of projected gravity in the base x-y plane

    # Standing reward scales.
    height_reward_scale = 2.0
    upright_reward_scale = 2.0
    joint_pose_reward_scale = 0.2
    planar_velocity_reward_scale = 2.0
    yaw_velocity_reward_scale = 0.5
    vertical_velocity_penalty_scale = -1.0
    angular_velocity_penalty_scale = -0.2
    action_rate_penalty_scale = -0.05
    torque_penalty_scale = -2.0e-5
    termination_penalty_scale = -2.0
    feet_air_time_reward_scale = 0.5
    foot_clearance_reward_scale = 0.25
    feet_slide_penalty_scale = -0.2
    gait_reward_scale = 1.0
    gait_reward_std = 0.1
    gait_max_error = 0.3  # [s]
    feet_air_time_threshold = 0.3  # [s]
    target_foot_clearance = 0.08  # [m]

    # Reset randomization for standing recovery.
    initial_joint_position_noise = 0.1  # [rad]
    initial_joint_velocity_noise = 0.1  # [rad/s]
    initial_base_tilt = 0.087  # [rad], approximately 5 degrees
    initial_base_linear_velocity = 0.1  # [m/s]
    initial_base_angular_velocity = 0.1  # [rad/s]

    # Horizontal velocity impulses applied independently during each episode.
    push_interval_range_s = (2.0, 5.0)
    max_push_velocity = 0.5  # [m/s]

    # Planar velocity commands: forward/lateral velocity and yaw rate.
    forward_velocity_range = (-1.0, 2.0)  # [m/s]
    lateral_velocity_range = (-0.5, 0.5)  # [m/s]
    yaw_velocity_range = (-1.0, 1.0)  # [rad/s]
    standing_command_probability = 0.1
    command_resample_range_s = (5.0, 10.0)

    # Interactive command input (disabled during training).
    manual_commands = False
    manual_forward_velocity = 1.0  # [m/s]
    manual_lateral_velocity = 0.5  # [m/s]
    manual_yaw_velocity = 1.0  # [rad/s]

    # Disabled for the flat task and enabled by the derived rough task.
    terrain_curriculum = False
    terrain_move_up_planar_error = 0.35  # [m/s], episode mean
    terrain_move_up_yaw_error = 0.4  # [rad/s], episode mean
    terrain_move_down_planar_error = 0.6  # [m/s], episode mean
    terrain_move_down_yaw_error = 0.7  # [rad/s], episode mean
    terrain_move_up_episode_fraction = 0.8

    # Optional fixed rough-terrain selection for evaluation. Empty/-1 keeps training behavior.
    evaluation_terrain_type = ""
    evaluation_terrain_level = -1


@configclass
class AnymaltrainingRoughSceneCfg(AnymaltrainingSceneCfg):
    """The flat scene with generated terrain and a base-mounted height scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True),
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class AnymaltrainingRoughEnvCfg(AnymaltrainingEnvCfg):
    """Rough-terrain variant sharing the dynamics, commands, rewards, and resets."""

    # The 187 terrain-height samples are appended to the flat task's 48 observations.
    observation_space = 235
    viewer = ViewerCfg(eye=(3.5, 3.5, 2.5), lookat=(0.0, 0.0, 0.5), origin_type="env", env_index=0)
    scene: AnymaltrainingRoughSceneCfg = AnymaltrainingRoughSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    terrain_curriculum = True
