"""Compare two controlled gait-evaluation outputs with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def read_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "trials.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


before = read_rows(args.before)
after = read_rows(args.after)
if len(before) != len(after):
    raise ValueError("Paired gait evaluations must contain the same number of trials.")

metrics = {
    "Trot deviation index ↓": "trot_deviation_index",
    "Touchdown phase MAE [cycles] ↓": "phase_mae_cycles",
    "Duty-factor CV ↓": "duty_factor_cv",
    "Duty-factor range ↓": "duty_factor_range",
    "Stride-period CV ↓": "stride_period_cv",
    "LF–RH contact agreement ↑": "diagonal_contact_agreement",
    "RF–LH contact agreement ↑": "other_diagonal_contact_agreement",
    "Diagonal-pair opposition ↑": "diagonal_pair_opposition",
    "Mean forward velocity [m/s]": "mean_vx_mps",
    "Velocity tracking MAE [m/s] ↓": "mean_velocity_error_mps",
}

rng = np.random.default_rng(123)
sample_indices = rng.integers(0, len(before), size=(20_000, len(before)))
results = []
for label, key in metrics.items():
    before_values = np.asarray([float(row[key]) for row in before])
    after_values = np.asarray([float(row[key]) for row in after])
    differences = after_values - before_values
    bootstrap_means = differences[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    results.append(
        {
            "metric": label,
            "before": float(before_values.mean()),
            "after": float(after_values.mean()),
            "change": float(differences.mean()),
            "ci95": [float(low), float(high)],
        }
    )

payload = {"paired_trials": len(before), "bootstrap_resamples": 20_000, "metrics": results}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
