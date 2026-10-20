"""Expand a 235-observation rough-locomotion checkpoint for two gait-phase inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def pad_observation_columns(tensor: torch.Tensor, old_size: int, added_size: int) -> torch.Tensor:
    """Append zero columns to first-layer weights or matching Adam moments."""
    if tensor.ndim == 2 and tensor.shape[1] == old_size:
        return torch.nn.functional.pad(tensor, (0, added_size))
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-observations", type=int, default=235)
    parser.add_argument("--added-observations", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    for model_key in ("actor_state_dict", "critic_state_dict"):
        state = checkpoint[model_key]
        state["mlp.0.weight"] = pad_observation_columns(
            state["mlp.0.weight"], args.old_observations, args.added_observations
        )

    # Adam stores first and second moments with the same shapes as parameters.
    for parameter_state in checkpoint["optimizer_state_dict"]["state"].values():
        for state_name, value in tuple(parameter_state.items()):
            if isinstance(value, torch.Tensor):
                parameter_state[state_name] = pad_observation_columns(
                    value, args.old_observations, args.added_observations
                )

    if checkpoint.get("infos") is None:
        checkpoint["infos"] = {}
    infos = checkpoint["infos"]
    infos["phase_clock_migration"] = {
        "source": str(args.input.resolve()),
        "old_observations": args.old_observations,
        "new_observations": args.old_observations + args.added_observations,
        "new_columns_initialized_to_zero": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"[INFO] Migrated checkpoint written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
