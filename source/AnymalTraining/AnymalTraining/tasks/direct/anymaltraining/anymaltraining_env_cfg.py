# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class AnymaltrainingSceneCfg(InteractiveSceneCfg):
    """Scene configuration using the template-based spawning path."""

    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
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
