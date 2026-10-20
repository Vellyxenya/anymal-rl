# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate quadruped footfall timing under a fixed command on a flat plane."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

import AnymalTraining.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


FOOT_NAMES = ("LF", "RF", "LH", "RH")

parser = argparse.ArgumentParser(description="Controlled footfall-timing evaluation for an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel trials on identical flat terrain.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration", type=float, default=10.0, help="Measured duration after warm-up [s].")
parser.add_argument("--warmup", type=float, default=2.0, help="Unmeasured settling duration [s].")
parser.add_argument("--command_vx", type=float, default=1.0)
parser.add_argument("--command_vy", type=float, default=0.0)
parser.add_argument("--command_yaw", type=float, default=0.0)
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Foot contact-force threshold [N].")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.duration <= 0 or args_cli.warmup < 0:
    parser.error("--duration must be positive and --warmup must be non-negative.")

remaining_args = list_intersection(remaining_args, None)
sys.argv = [sys.argv[0]] + remaining_args
installed_version = metadata.version("rsl-rl-lib")


def _touchdowns(contact: np.ndarray) -> np.ndarray:
    """Return sample indices where a foot transitions from swing to stance."""
    return np.flatnonzero(contact[1:] & ~contact[:-1]) + 1


def _circular_error(phase: float, target: float) -> float:
    """Shortest distance between two phases on the unit circle."""
    return abs((phase - target + 0.5) % 1.0 - 0.5)


def _phase_errors(contacts: np.ndarray) -> tuple[list[float], dict[str, list[float]]]:
    """Measure foot touchdown phases relative to successive LF strides."""
    events = [_touchdowns(contacts[:, foot]) for foot in range(4)]
    targets = {"RF": (1, 0.5), "LH": (2, 0.5), "RH": (3, 0.0)}
    all_errors: list[float] = []
    per_pair = {name: [] for name in targets}
    for start, end in zip(events[0][:-1], events[0][1:]):
        period = end - start
        if period <= 1:
            continue
        for name, (foot, target) in targets.items():
            candidates = events[foot][(events[foot] >= start) & (events[foot] < end)]
            if len(candidates) == 0:
                continue
            phase = (candidates[0] - start) / period
            error = _circular_error(float(phase), target)
            per_pair[name].append(error)
            all_errors.append(error)
    return all_errors, per_pair


def _trial_metrics(
    contacts: np.ndarray, dt: float, fell: bool, mean_vx: float, mean_velocity_error: float
) -> dict[str, float | int | bool]:
    duty = contacts.mean(axis=0)
    stride_cvs = []
    stride_periods = []
    touchdown_counts = []
    for foot in range(4):
        events = _touchdowns(contacts[:, foot])
        touchdown_counts.append(int(len(events)))
        intervals = np.diff(events) * dt
        if len(intervals) >= 2 and intervals.mean() > 0:
            stride_cvs.append(float(intervals.std(ddof=1) / intervals.mean()))
            stride_periods.append(float(intervals.mean()))

    phase_errors, pair_errors = _phase_errors(contacts)
    phase_mae = float(np.mean(phase_errors)) if phase_errors else float("nan")
    duty_cv = float(duty.std(ddof=0) / duty.mean()) if duty.mean() > 0 else float("nan")
    stride_cv = float(np.mean(stride_cvs)) if stride_cvs else float("nan")
    # Project-specific aggregate: 0 is an ideal trot. Phase is weighted most heavily;
    # duty imbalance and cycle variability detect one-foot and irregular solutions.
    trot_deviation = phase_mae + 0.25 * duty_cv + 0.25 * stride_cv

    result: dict[str, float | int | bool] = {
        "fell": fell,
        "valid": bool(len(phase_errors) >= 6 and len(stride_cvs) >= 2),
        "mean_vx_mps": mean_vx,
        "mean_velocity_error_mps": mean_velocity_error,
        "trot_deviation_index": trot_deviation,
        "phase_mae_cycles": phase_mae,
        "duty_factor_mean": float(duty.mean()),
        "duty_factor_cv": duty_cv,
        "duty_factor_range": float(duty.max() - duty.min()),
        "stride_period_cv": stride_cv,
        "mean_stride_period_s": float(np.mean(stride_periods)) if stride_periods else float("nan"),
        "diagonal_contact_agreement": float(np.mean(contacts[:, 0] == contacts[:, 3])),
        "other_diagonal_contact_agreement": float(np.mean(contacts[:, 1] == contacts[:, 2])),
        "diagonal_pair_opposition": float(np.mean(contacts[:, 0] != contacts[:, 1])),
        "touchdowns_min": min(touchdown_counts),
    }
    for index, name in enumerate(FOOT_NAMES):
        result[f"duty_{name}"] = float(duty[index])
    for name in ("RF", "LH", "RH"):
        result[f"phase_error_LF_{name}"] = (
            float(np.mean(pair_errors[name])) if pair_errors[name] else float("nan")
        )
    return result


def _summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid"]]
    metric_names = [
        "trot_deviation_index",
        "phase_mae_cycles",
        "duty_factor_mean",
        "duty_factor_cv",
        "duty_factor_range",
        "stride_period_cv",
        "mean_stride_period_s",
        "mean_vx_mps",
        "mean_velocity_error_mps",
        "diagonal_contact_agreement",
        "other_diagonal_contact_agreement",
        "diagonal_pair_opposition",
        "phase_error_LF_RF",
        "phase_error_LF_LH",
        "phase_error_LF_RH",
    ]
    summary = {
        "trials": len(rows),
        "valid_trials": len(valid),
        "fall_rate": sum(bool(row["fell"]) for row in rows) / len(rows),
    }
    for metric in metric_names:
        values = np.asarray([row[metric] for row in valid], dtype=float)
        values = values[np.isfinite(values)]
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.terrain_curriculum = False
        env_cfg.manual_commands = False
        env_cfg.max_push_velocity = 0.0
        # Preserve the rough policy's height-scanner observations while replacing only
        # the generated surface with a deterministic plane.
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1, physics_material=env_cfg.scene.terrain.physics_material
        )

        resume_path = retrieve_file_path(args_cli.checkpoint)
        env_cfg.log_dir = os.path.dirname(resume_path)
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        num_envs = base_env.num_envs
        step_dt = base_env.step_dt

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

        fixed_command = torch.tensor(
            (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw),
            device=base_env.device,
            dtype=torch.float32,
        ).repeat(base_env.num_envs, 1)

        def apply_command():
            base_env.commands.copy_(fixed_command)
            base_env.command_countdown.fill_(base_env.max_episode_length + 1)

        obs = env.get_observations()
        apply_command()
        warmup_steps = round(args_cli.warmup / base_env.step_dt)
        measure_steps = round(args_cli.duration / base_env.step_dt)
        contacts = torch.zeros(
            (measure_steps, num_envs, 4), dtype=torch.bool, device=base_env.device
        )
        alive = torch.ones(num_envs, dtype=torch.bool, device=base_env.device)
        fell = torch.zeros_like(alive)
        velocity_sum = torch.zeros(num_envs, device=base_env.device)
        velocity_error_sum = torch.zeros_like(velocity_sum)
        measured_steps = torch.zeros_like(velocity_sum)
        foot_ids = (
            base_env.left_front_foot_id,
            base_env.right_front_foot_id,
            base_env.left_hind_foot_id,
            base_env.right_hind_foot_id,
        )

        try:
            with torch.inference_mode():
                for step in range(warmup_steps + measure_steps):
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    apply_command()
                    terminated = dones.bool() & alive
                    fell |= terminated
                    alive &= ~dones.bool()
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    elif policy_nn is not None:
                        policy_nn.reset(dones)
                    if step >= warmup_steps:
                        current_vx = base_env.robot.data.root_lin_vel_b.torch[:, 0]
                        velocity_sum[alive] += current_vx[alive]
                        velocity_error_sum[alive] += torch.abs(current_vx[alive] - args_cli.command_vx)
                        measured_steps[alive] += 1
                        forces = base_env.contact_sensor.data.net_forces_w_history.torch[:, :, foot_ids]
                        sample = torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1).values
                        contacts[step - warmup_steps] = (sample > args_cli.contact_threshold) & alive[:, None]
        finally:
            env.close()

        contacts_np = contacts.cpu().numpy()
        fell_np = fell.cpu().numpy()
        sample_count = measured_steps.clamp_min(1)
        mean_vx = (velocity_sum / sample_count).cpu().numpy()
        mean_velocity_error = (velocity_error_sum / sample_count).cpu().numpy()
        rows = [
            {
                "trial": index,
                **_trial_metrics(
                    contacts_np[:, index],
                    step_dt,
                    bool(fell_np[index]),
                    float(mean_vx[index]),
                    float(mean_velocity_error[index]),
                ),
            }
            for index in range(num_envs)
        ]
        summary = _summarize(rows)
        metadata_dict = {
            "checkpoint": str(Path(resume_path).resolve()),
            "task": args_cli.task,
            "terrain": "flat plane",
            "seed": args_cli.seed,
            "duration_s": args_cli.duration,
            "warmup_s": args_cli.warmup,
            "command": [args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw],
            "contact_threshold_n": args_cli.contact_threshold,
            "trot_targets": {"LF-RH": 0.0, "LF-RF": 0.5, "LF-LH": 0.5},
            "trot_deviation_definition": "phase_mae + 0.25*duty_factor_cv + 0.25*stride_period_cv",
        }
        output_dir = Path(args_cli.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "contacts.npz", contacts=contacts_np, dt=step_dt)
        with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump({"metadata": metadata_dict, "summary": summary}, file, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"[INFO] Gait evaluation written to: {output_dir}")


if __name__ == "__main__":
    main()
