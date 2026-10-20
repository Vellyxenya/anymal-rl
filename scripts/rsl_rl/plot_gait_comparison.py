"""Create a two-policy foot-contact comparison from controlled gait evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Before evaluation contacts.npz")
    parser.add_argument("after", type=Path, help="After evaluation contacts.npz")
    parser.add_argument("--before-label", default="Before")
    parser.add_argument("--after-label", default="After")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = [np.load(args.before), np.load(args.after)]
    labels = [args.before_label, args.after_label]
    foot_names = ("LF", "RF", "LH", "RH")
    colors = ("#C94A4A", "#16845B")
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    for axis, dataset, label, color in zip(axes, datasets, labels, colors):
        contacts = dataset["contacts"][:, 0, :].T.astype(np.uint8)
        dt = float(dataset["dt"])
        duration = contacts.shape[1] * dt
        duty = contacts.mean(axis=1)
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
        axis.set_ylabel("Foot")
        axis.set_title(label, loc="left", fontsize=16, fontweight="bold", color=color, pad=14)
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
        axis.grid(False)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(
        handles=(Patch(facecolor="black", label="Contact"), Patch(facecolor="white", edgecolor="black", label="Swing")),
        loc="upper right",
        ncols=2,
    )
    fig.suptitle("ANYmal foot-contact gait: original policy vs phase-clock policy", fontsize=22, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Controlled flat terrain · 1.0 m/s forward command · seed 42 · 2 s warm-up · 10 s measurement",
        ha="center",
        fontsize=12,
        color="#42526B",
    )
    fig.text(0.5, 0.02, "Black = stance/contact   ·   White = swing", ha="center", fontsize=11, color="#42526B")
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.90), h_pad=3.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"[INFO] Comparison written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
