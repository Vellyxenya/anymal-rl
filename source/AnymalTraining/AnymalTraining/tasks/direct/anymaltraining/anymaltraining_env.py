# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv

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

        self.episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "base_height",
                "upright",
                "joint_pose",
                "linear_velocity",
                "angular_velocity",
                "action_rate",
                "torque",
                "termination",
            )
        }

    def _setup_scene(self):
        # The scene configuration creates and clones the articulation through the
        # template-based path required for effective PhysX joint actuation.
        self.robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()
        self.processed_actions = self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.processed_actions)

    def _get_observations(self) -> dict:
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
        linear_velocity_error = torch.sum(torch.square(self.robot.data.root_lin_vel_b.torch), dim=1)
        angular_velocity_error = torch.sum(torch.square(self.robot.data.root_ang_vel_b.torch), dim=1)
        action_rate_error = torch.mean(torch.square(self.actions - self.previous_actions), dim=1)
        torque_error = torch.mean(torch.square(self.robot.data.applied_torque.torch), dim=1)

        rewards = {
            "base_height": self.cfg.height_reward_scale * torch.exp(-base_height_error / 0.01) * self.step_dt,
            "upright": self.cfg.upright_reward_scale * torch.exp(-tilt_error / 0.1) * self.step_dt,
            "joint_pose": self.cfg.joint_pose_reward_scale * torch.exp(-joint_pose_error / 0.25) * self.step_dt,
            "linear_velocity": self.cfg.linear_velocity_penalty_scale * linear_velocity_error * self.step_dt,
            "angular_velocity": self.cfg.angular_velocity_penalty_scale * angular_velocity_error * self.step_dt,
            "action_rate": self.cfg.action_rate_penalty_scale * action_rate_error * self.step_dt,
            "torque": self.cfg.torque_penalty_scale * torque_error * self.step_dt,
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

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
