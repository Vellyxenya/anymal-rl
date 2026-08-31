# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint with fixed commands and reproducible metrics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

import AnymalTraining.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


COMMANDS = {
    "standing": (0.0, 0.0, 0.0),
    "straight": (1.0, 0.0, 0.0),
    "turn_left": (0.0, 0.0, 0.7),
    "combined": (1.0, 0.0, 0.5),
}

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL locomotion checkpoint.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel evaluation environments.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--episodes_per_env", type=int, default=1, help="Completed episodes required per environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory; defaults beside the checkpoint.")
parser.add_argument("--external_callback", default=None, help="Fully qualified environment registration callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.episodes_per_env < 1:
    parser.error("--episodes_per_env must be at least 1.")
if args_cli.num_envs < len(COMMANDS):
    parser.error(f"--num_envs must be at least {len(COMMANDS)} so every command scenario is represented.")

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _terrain_names(base_env) -> list[str]:
    """Convert terrain column indices into the configured terrain-family names."""
    terrain = base_env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        return ["flat"] * base_env.num_envs
    sub_terrains = terrain.cfg.terrain_generator.sub_terrains
    total = sum(cfg.proportion for cfg in sub_terrains.values())
    boundaries = []
    cumulative = 0.0
    for name, cfg in sub_terrains.items():
        cumulative += cfg.proportion / total
        boundaries.append((cumulative, name))
    names = []
    for column in terrain.terrain_types.detach().cpu().tolist():
        fraction = column / terrain.cfg.terrain_generator.num_cols + 0.001
        names.append(next(name for upper, name in boundaries if fraction < upper or upper >= 1.0))
    return names


def _write_reports(records: list[dict], output_dir: Path, metadata_dict: dict) -> None:
    """Write raw episodes, grouped summaries, and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["terrain_type"], record["terrain_level"], record["command_mode"])].append(record)
    summary_rows = []
    for (terrain_type, terrain_level, command_mode), rows in sorted(grouped.items()):
        count = len(rows)
        summary_rows.append(
            {
                "terrain_type": terrain_type,
                "terrain_level": terrain_level,
                "command_mode": command_mode,
                "episodes": count,
                "completion_rate": sum(row["completed"] for row in rows) / count,
                "fall_rate": sum(row["fell"] for row in rows) / count,
                "mean_duration_s": sum(row["duration_s"] for row in rows) / count,
                "mean_planar_error_mps": sum(row["planar_error_mps"] for row in rows) / count,
                "mean_yaw_error_radps": sum(row["yaw_error_radps"] for row in rows) / count,
                "mean_path_length_m": sum(row["path_length_m"] for row in rows) / count,
                "mean_displacement_m": sum(row["displacement_m"] for row in rows) / count,
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    overall = {
        "episodes": len(records),
        "completion_rate": sum(row["completed"] for row in records) / len(records),
        "fall_rate": sum(row["fell"] for row in records) / len(records),
        "mean_planar_error_mps": sum(row["planar_error_mps"] for row in records) / len(records),
        "mean_yaw_error_radps": sum(row["yaw_error_radps"] for row in records) / len(records),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata_dict, "overall": overall, "groups": summary_rows}, file, indent=2)

    print("\n[RESULT] Baseline evaluation")
    print(f"  Episodes:             {overall['episodes']}")
    print(f"  Completion rate:      {overall['completion_rate']:.1%}")
    print(f"  Fall rate:            {overall['fall_rate']:.1%}")
    print(f"  Mean planar error:    {overall['mean_planar_error_mps']:.3f} m/s")
    print(f"  Mean yaw-rate error:  {overall['mean_yaw_error_radps']:.3f} rad/s")
    print(f"  Reports:              {output_dir}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run finite, fixed-command checkpoint evaluation."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.terrain_curriculum = False
        env_cfg.manual_commands = False
        env_cfg.max_push_velocity = 0.0

        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        env_cfg.log_dir = os.path.dirname(resume_path)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped

        print(f"[INFO] Loading checkpoint: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        configure_seed(env_cfg.seed, True)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)
        policy_nn = None if version.parse(installed_version) >= version.parse("4.0.0") else runner.alg.policy

        mode_names = list(COMMANDS)
        env_modes = [mode_names[index % len(mode_names)] for index in range(base_env.num_envs)]
        fixed_commands = torch.tensor(
            [COMMANDS[mode] for mode in env_modes], device=base_env.device, dtype=torch.float32
        )

        def apply_fixed_commands():
            base_env.commands.copy_(fixed_commands)
            base_env.command_countdown.fill_(base_env.max_episode_length + 1)

        apply_fixed_commands()
        obs = env.get_observations()
        terrain_names = _terrain_names(base_env)
        if base_env.scene.terrain.cfg.terrain_type == "generator":
            terrain_levels = base_env.scene.terrain.terrain_levels.detach().cpu().tolist()
        else:
            terrain_levels = [0] * base_env.num_envs

        steps = torch.zeros(base_env.num_envs, device=base_env.device)
        planar_error = torch.zeros_like(steps)
        yaw_error = torch.zeros_like(steps)
        path_length = torch.zeros_like(steps)
        displacement_xy = torch.zeros(base_env.num_envs, 2, device=base_env.device)
        episode_counts = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
        records = []

        try:
            while torch.any(episode_counts < args_cli.episodes_per_env):
                active = episode_counts < args_cli.episodes_per_env
                with torch.inference_mode():
                    current_velocity = base_env.robot.data.root_lin_vel_b.torch[:, :2]
                    current_world_velocity = base_env.robot.data.root_lin_vel_w.torch[:, :2]
                    current_yaw = base_env.robot.data.root_ang_vel_b.torch[:, 2]
                    steps[active] += 1
                    planar_error[active] += torch.linalg.vector_norm(
                        fixed_commands[active, :2] - current_velocity[active], dim=1
                    )
                    yaw_error[active] += torch.abs(fixed_commands[active, 2] - current_yaw[active])
                    path_length[active] += torch.linalg.vector_norm(current_velocity[active], dim=1) * base_env.step_dt
                    displacement_xy[active] += current_world_velocity[active] * base_env.step_dt
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)

                    completed_ids = torch.nonzero(dones & active, as_tuple=False).squeeze(-1)
                    for env_id in completed_ids.detach().cpu().tolist():
                        count = max(steps[env_id].item(), 1.0)
                        displacement = torch.linalg.vector_norm(displacement_xy[env_id]).item()
                        fell = bool(base_env.reset_terminated[env_id].item())
                        records.append(
                            {
                                "env_id": env_id,
                                "episode": int(episode_counts[env_id].item()),
                                "terrain_type": terrain_names[env_id],
                                "terrain_level": int(terrain_levels[env_id]),
                                "command_mode": env_modes[env_id],
                                "command_vx_mps": float(fixed_commands[env_id, 0].item()),
                                "command_vy_mps": float(fixed_commands[env_id, 1].item()),
                                "command_yaw_radps": float(fixed_commands[env_id, 2].item()),
                                "completed": not fell,
                                "fell": fell,
                                "duration_s": count * base_env.step_dt,
                                "planar_error_mps": planar_error[env_id].item() / count,
                                "yaw_error_radps": yaw_error[env_id].item() / count,
                                "path_length_m": path_length[env_id].item(),
                                "displacement_m": displacement,
                            }
                        )
                        episode_counts[env_id] += 1
                    if len(completed_ids) > 0:
                        steps[completed_ids] = 0
                        planar_error[completed_ids] = 0
                        yaw_error[completed_ids] = 0
                        path_length[completed_ids] = 0
                        displacement_xy[completed_ids] = 0
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    elif policy_nn is not None:
                        policy_nn.reset(dones)
                    apply_fixed_commands()
                    obs = env.get_observations()
        finally:
            env.close()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path(args_cli.output_dir).resolve()
            if args_cli.output_dir
            else Path(resume_path).resolve().parent / "evaluations" / timestamp
        )
        _write_reports(
            records,
            output_dir,
            {
                "checkpoint": str(Path(resume_path).resolve()),
                "task": args_cli.task,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "episodes_per_env": args_cli.episodes_per_env,
                "commands": COMMANDS,
            },
        )


if __name__ == "__main__":
    main()
