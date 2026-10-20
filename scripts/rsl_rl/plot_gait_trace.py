"""Render one controlled foot-contact trace as a gait diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contacts", type=Path)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = np.load(args.contacts)
    contacts = dataset["contacts"][:, 0, :].T.astype(np.uint8)
    dt = float(dataset["dt"])
    duration = contacts.shape[1] * dt
    foot_names = ("LF", "RF", "LH", "RH")
    duty = contacts.mean(axis=1)

    fig, axis = plt.subplots(figsize=(16, 5.4))
    axis.imshow(
        contacts,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=(0.0, duration, len(foot_names) - 0.5, -0.5),
    )
    axis.set_yticks(range(4), labels=foot_names)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Foot")
    axis.set_title(f"Foot-contact gait — {args.checkpoint_label}", fontsize=19, fontweight="bold", pad=38)
    axis.text(
        1.0,
        1.07,
        "Duty factor  " + "  ·  ".join(f"{name} {value:.1%}" for name, value in zip(foot_names, duty)),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#42526B",
    )
    axis.legend(
        handles=(Patch(facecolor="black", label="Contact"), Patch(facecolor="white", edgecolor="black", label="Swing")),
        loc="upper right",
        ncols=2,
    )
    fig.text(
        0.5,
        0.93,
        "Controlled flat terrain · 1.0 m/s forward command · seed 42 · 2 s warm-up · 10 s measurement",
        ha="center",
        fontsize=11,
        color="#42526B",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"[INFO] Gait plot written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
