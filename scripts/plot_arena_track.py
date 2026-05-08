from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/a2rl_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from a2rl_drone_training.course import course_by_name


def _gate_segment(center: np.ndarray, right: np.ndarray, width: float) -> np.ndarray:
    half = 0.5 * width * right[:2]
    return np.stack([center[:2] - half, center[:2] + half])


def plot_course(course_name: str, output: Path) -> None:
    course = course_by_name(course_name)
    centers = course.centers
    normals = course.normals
    right_axes = course.right_axes
    line = course.nominal_racing_line

    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    ax.set_title(course.name, fontsize=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(course.bounds_min[0], course.bounds_max[0])
    ax.set_ylim(course.bounds_min[1], course.bounds_max[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.45)

    bounds_x = [
        course.bounds_min[0],
        course.bounds_max[0],
        course.bounds_max[0],
        course.bounds_min[0],
        course.bounds_min[0],
    ]
    bounds_y = [
        course.bounds_min[1],
        course.bounds_min[1],
        course.bounds_max[1],
        course.bounds_max[1],
        course.bounds_min[1],
    ]
    ax.plot(bounds_x, bounds_y, color="#222222", linewidth=1.6, label="arena bounds")

    ax.plot(
        line[:, 0],
        line[:, 1],
        color="#8f8f8f",
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="nominal racing line",
    )
    ax.scatter(line[0, 0], line[0, 1], marker="o", s=55, color="#2ca25f", label="start")
    ax.scatter(line[-1, 0], line[-1, 1], marker="*", s=110, color="#756bb1", label="finish")

    for idx, (center, normal, right, width, logical_id, opening) in enumerate(
        zip(
            centers,
            normals,
            right_axes,
            course.outer_widths,
            course.logical_gate_ids,
            course.openings,
            strict=True,
        ),
        start=1,
    ):
        segment = _gate_segment(center, right, float(width))
        ax.plot(segment[:, 0], segment[:, 1], color="#111111", linewidth=4.0, solid_capstyle="butt")
        ax.scatter(center[0], center[1], s=26, color="#111111", zorder=4)

        arrow_len = 1.1
        ax.arrow(
            center[0],
            center[1],
            normal[0] * arrow_len,
            normal[1] * arrow_len,
            width=0.035,
            head_width=0.32,
            head_length=0.38,
            length_includes_head=True,
            color="#1f77b4",
            alpha=0.95,
            zorder=3,
        )

        label = str(idx)
        if opening:
            label = f"{idx} {opening}"
        label_offset = 0.45 * normal[:2] + np.array([0.12, 0.12], dtype=np.float32)
        ax.text(
            center[0] + label_offset[0],
            center[1] + label_offset[1],
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111111",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            zorder=5,
        )

        if opening:
            ax.text(
                center[0] - 0.55 * normal[0],
                center[1] - 0.55 * normal[1],
                f"logical {logical_id}, z={center[2]:.1f}m",
                fontsize=7,
                ha="center",
                va="center",
                color="#444444",
                zorder=5,
            )

    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a configured A2RL drone racing course.")
    parser.add_argument("--course", default="arena_38m_stacked")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/arena_38m_stacked_track.png"),
    )
    args = parser.parse_args()
    plot_course(args.course, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
