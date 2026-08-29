# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .anymaltraining_env_cfg import AnymaltrainingEnvCfg


class AnymaltrainingEnv(DirectRLEnv):
    cfg: AnymaltrainingEnvCfg

    def __init__(self, cfg: AnymaltrainingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self.actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        # Kept at zero for now; these slots make the observation layout compatible
        # with the standard ANYmal locomotion task that we will build next.
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_commands()
        self.push_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_push_intervals()
        self.feet_sensor_ids, _ = self.contact_sensor.find_sensors(".*FOOT")
        self.feet_body_ids, _ = self.robot.find_bodies(".*FOOT")

        self.episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "base_height",
                "upright",
                "joint_pose",
                "forward_velocity",
                "lateral_velocity",
                "vertical_velocity",
                "angular_velocity",
                "action_rate",
                "torque",
                "feet_air_time",
                "foot_clearance",
                "feet_slide",
                "termination",
            )
        }

        if self.sim.visualizers:
            self.set_debug_vis(True)

    def _setup_scene(self):
        # The scene configuration creates and clones the articulation through the
        # template-based path required for effective PhysX joint actuation.
        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene["contact_sensor"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._apply_random_pushes()
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()
        self.processed_actions = self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.processed_actions)

    def _get_observations(self) -> dict:
        self._update_commands()
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b.torch,
                self.robot.data.root_ang_vel_b.torch,
                self.robot.data.projected_gravity_b.torch,
                self.commands,
                self.robot.data.joint_pos.torch - self.robot.data.default_joint_pos.torch,
                self.robot.data.joint_vel.torch,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        base_height_error = torch.square(
            self.robot.data.root_pos_w.torch[:, 2] - self.cfg.target_base_height
        )
        tilt_error = torch.sum(torch.square(self.robot.data.projected_gravity_b.torch[:, :2]), dim=1)
        joint_pose_error = torch.mean(
            torch.square(self.robot.data.joint_pos.torch - self.robot.data.default_joint_pos.torch), dim=1
        )
        forward_velocity_error = torch.square(
            self.commands[:, 0] - self.robot.data.root_lin_vel_b.torch[:, 0]
        )
        lateral_velocity_error = torch.square(self.robot.data.root_lin_vel_b.torch[:, 1])
        vertical_velocity_error = torch.square(self.robot.data.root_lin_vel_b.torch[:, 2])
        angular_velocity_error = torch.sum(torch.square(self.robot.data.root_ang_vel_b.torch), dim=1)
        action_rate_error = torch.mean(torch.square(self.actions - self.previous_actions), dim=1)
        torque_error = torch.mean(torch.square(self.robot.data.applied_torque.torch), dim=1)
        is_moving_command = self.commands[:, 0] > 0.1

        first_contact = self.contact_sensor.compute_first_contact(self.step_dt).torch[:, self.feet_sensor_ids]
        last_air_time = self.contact_sensor.data.last_air_time.torch[:, self.feet_sensor_ids]
        feet_air_time = torch.sum(
            (last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1
        ) * is_moving_command

        feet_height = self.robot.data.body_pos_w.torch[:, self.feet_body_ids, 2]
        feet_horizontal_speed = torch.linalg.vector_norm(
            self.robot.data.body_lin_vel_w.torch[:, self.feet_body_ids, :2], dim=2
        )
        foot_clearance = torch.sum(
            torch.exp(-torch.square(feet_height - self.cfg.target_foot_clearance) / 0.0025)
            * torch.tanh(2.0 * feet_horizontal_speed),
            dim=1,
        ) * is_moving_command

        contact_forces = self.contact_sensor.data.net_forces_w_history.torch[:, :, self.feet_sensor_ids]
        feet_in_contact = torch.max(torch.linalg.vector_norm(contact_forces, dim=-1), dim=1)[0] > 1.0
        feet_slide = torch.sum(feet_horizontal_speed * feet_in_contact, dim=1)

        rewards = {
            "base_height": self.cfg.height_reward_scale * torch.exp(-base_height_error / 0.01) * self.step_dt,
            "upright": self.cfg.upright_reward_scale * torch.exp(-tilt_error / 0.1) * self.step_dt,
            "joint_pose": self.cfg.joint_pose_reward_scale * torch.exp(-joint_pose_error / 0.25) * self.step_dt,
            "forward_velocity": (
                self.cfg.forward_velocity_reward_scale
                * torch.exp(-forward_velocity_error / 0.25)
                * self.step_dt
            ),
            "lateral_velocity": (
                self.cfg.lateral_velocity_penalty_scale * lateral_velocity_error * self.step_dt
            ),
            "vertical_velocity": (
                self.cfg.vertical_velocity_penalty_scale * vertical_velocity_error * self.step_dt
            ),
            "angular_velocity": self.cfg.angular_velocity_penalty_scale * angular_velocity_error * self.step_dt,
            "action_rate": self.cfg.action_rate_penalty_scale * action_rate_error * self.step_dt,
            "torque": self.cfg.torque_penalty_scale * torque_error * self.step_dt,
            "feet_air_time": self.cfg.feet_air_time_reward_scale * feet_air_time * self.step_dt,
            "foot_clearance": self.cfg.foot_clearance_reward_scale * foot_clearance * self.step_dt,
            "feet_slide": self.cfg.feet_slide_penalty_scale * feet_slide * self.step_dt,
            "termination": self.cfg.termination_penalty_scale * self.reset_terminated.float(),
        }

        for name, value in rewards.items():
            self.episode_reward_sums[name] += value
        return torch.stack(tuple(rewards.values())).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self.robot.data.root_pos_w.torch[:, 2] < self.cfg.min_base_height
        base_tilt = torch.linalg.vector_norm(self.robot.data.projected_gravity_b.torch[:, :2], dim=1)
        too_tilted = base_tilt > self.cfg.max_base_tilt
        fell = too_low | too_tilted
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.extras["log"] = {}
        for name, episodic_sum in self.episode_reward_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = torch.mean(episodic_sum[env_ids]).item()
            episodic_sum[env_ids] = 0.0
        self.extras["log"]["Episode_Termination/fall"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._sample_commands(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()

        joint_pos += torch.empty_like(joint_pos).uniform_(
            -self.cfg.initial_joint_position_noise, self.cfg.initial_joint_position_noise
        )
        joint_vel.uniform_(-self.cfg.initial_joint_velocity_noise, self.cfg.initial_joint_velocity_noise)

        num_resets = len(env_ids)
        roll = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.initial_base_tilt, self.cfg.initial_base_tilt
        )
        pitch = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.initial_base_tilt, self.cfg.initial_base_tilt
        )
        yaw = torch.zeros(num_resets, device=self.device)
        orientation_noise = quat_from_euler_xyz(roll, pitch, yaw)
        root_pose[:, 3:7] = quat_mul(orientation_noise, root_pose[:, 3:7])

        root_vel[:, :3].uniform_(
            -self.cfg.initial_base_linear_velocity, self.cfg.initial_base_linear_velocity
        )
        root_vel[:, 3:].uniform_(
            -self.cfg.initial_base_angular_velocity, self.cfg.initial_base_angular_velocity
        )
        root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._sample_push_intervals(env_ids)

    def _apply_random_pushes(self) -> None:
        """Apply independent horizontal velocity impulses at randomized intervals."""
        self.push_countdown -= 1
        push_env_ids = torch.nonzero(self.push_countdown <= 0, as_tuple=False).squeeze(-1)
        if len(push_env_ids) == 0:
            return

        root_velocity = self.robot.data.root_vel_w.torch[push_env_ids].clone()
        root_velocity[:, :2] += torch.empty_like(root_velocity[:, :2]).uniform_(
            -self.cfg.max_push_velocity, self.cfg.max_push_velocity
        )
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=push_env_ids)
        self._sample_push_intervals(push_env_ids)

    def _sample_push_intervals(self, env_ids: Sequence[int] | None = None) -> None:
        """Choose the number of policy steps until each environment's next push."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        min_steps = int(self.cfg.push_interval_range_s[0] / self.step_dt)
        max_steps = int(self.cfg.push_interval_range_s[1] / self.step_dt)
        self.push_countdown[env_ids] = torch.randint(
            min_steps, max_steps + 1, (len(env_ids),), device=self.device
        )

    def _update_commands(self) -> None:
        """Resample expired forward-speed commands."""
        self.command_countdown -= 1
        env_ids = torch.nonzero(self.command_countdown <= 0, as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: Sequence[int] | None = None) -> None:
        """Sample forward commands and their durations; lateral and yaw commands stay zero."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.forward_velocity_range[0], self.cfg.forward_velocity_range[1]
        )
        min_steps = int(self.cfg.command_resample_range_s[0] / self.step_dt)
        max_steps = int(self.cfg.command_resample_range_s[1] / self.step_dt)
        self.command_countdown[env_ids] = torch.randint(
            min_steps, max_steps + 1, (len(env_ids),), device=self.device
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Create and toggle desired/current velocity arrows."""
        if debug_vis:
            if not hasattr(self, "desired_velocity_marker"):
                desired_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                desired_cfg.prim_path = "/Visuals/Velocity/desired"
                desired_cfg.markers["arrow"].scale = (1.0, 0.15, 0.15)
                self.desired_velocity_marker = VisualizationMarkers(desired_cfg)

                current_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                current_cfg.prim_path = "/Visuals/Velocity/current"
                current_cfg.markers["arrow"].scale = (1.0, 0.15, 0.15)
                self.current_velocity_marker = VisualizationMarkers(current_cfg)
            self.desired_velocity_marker.set_visibility(True)
            self.current_velocity_marker.set_visibility(True)
        elif hasattr(self, "desired_velocity_marker"):
            self.desired_velocity_marker.set_visibility(False)
            self.current_velocity_marker.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw velocity arrows above each robot."""
        if not self.robot.is_initialized:
            return
        desired_position = self.robot.data.root_pos_w.torch.clone()
        current_position = desired_position.clone()
        desired_position[:, 2] += 0.7
        current_position[:, 2] += 0.5
        desired_scale, desired_orientation = self._velocity_to_arrow(self.commands[:, :2])
        current_scale, current_orientation = self._velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )
        self.desired_velocity_marker.visualize(desired_position, desired_orientation, desired_scale)
        self.current_velocity_marker.visualize(current_position, current_orientation, current_scale)

    def _velocity_to_arrow(self, velocity_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert body-frame planar velocities to world-frame arrow transforms."""
        default_scale = self.desired_velocity_marker.cfg.markers["arrow"].scale
        scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        scale[:, 0] *= torch.linalg.vector_norm(velocity_xy, dim=1) * 2.0
        heading = torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])
        zeros = torch.zeros_like(heading)
        orientation = quat_from_euler_xyz(zeros, zeros, heading)
        orientation = quat_mul(self.robot.data.root_quat_w.torch, orientation)
        return scale, orientation
