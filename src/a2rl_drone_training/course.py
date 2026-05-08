from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateCourse:
    """Gate centers and plane normals for a finite racing course."""

    name: str
    centers: np.ndarray
    normals: np.ndarray
    radius_m: float
    widths: np.ndarray
    heights: np.ndarray
    outer_widths: np.ndarray
    outer_heights: np.ndarray
    logical_gate_ids: np.ndarray
    openings: tuple[str, ...]
    nominal_racing_line: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    arena_size: tuple[float, float]
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def num_gates(self) -> int:
        return int(self.centers.shape[0])

    @property
    def right_axes(self) -> np.ndarray:
        up = _normalize(np.asarray(self.world_up, dtype=np.float32)[None, :])[0]
        right = np.cross(up[None, :], self.normals)
        return _normalize(right).astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1.0e-6)


def make_gate_course(
    centers: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    radius_m: float = 0.55,
    widths: np.ndarray | None = None,
    heights: np.ndarray | None = None,
    outer_widths: np.ndarray | None = None,
    outer_heights: np.ndarray | None = None,
    logical_gate_ids: np.ndarray | None = None,
    openings: tuple[str, ...] | None = None,
    nominal_racing_line: np.ndarray | None = None,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    arena_size: tuple[float, float] | None = None,
    name: str = "course",
) -> GateCourse:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (num_gates, 3)")
    if centers.shape[0] < 2:
        raise ValueError("a racing course needs at least two gates")

    if normals is None:
        forward = np.empty_like(centers)
        forward[:-1] = centers[1:] - centers[:-1]
        forward[-1] = centers[-1] - centers[-2]
        normals = forward
    normals = _normalize(np.asarray(normals, dtype=np.float32)).astype(np.float32)
    if normals.shape != centers.shape:
        raise ValueError("normals must have shape (num_gates, 3)")

    n_gates = centers.shape[0]
    if widths is None:
        widths = np.full((n_gates,), 2.0 * radius_m, dtype=np.float32)
    if heights is None:
        heights = np.full((n_gates,), 2.0 * radius_m, dtype=np.float32)
    if outer_widths is None:
        outer_widths = widths
    if outer_heights is None:
        outer_heights = heights
    if logical_gate_ids is None:
        logical_gate_ids = np.arange(1, n_gates + 1, dtype=np.int32)
    if openings is None:
        openings = tuple("" for _ in range(n_gates))
    if nominal_racing_line is None:
        nominal_racing_line = centers
    if bounds_min is None:
        bounds_min = np.array([-20.0, -20.0, 0.05], dtype=np.float32)
    if bounds_max is None:
        bounds_max = np.array([20.0, 20.0, 5.0], dtype=np.float32)

    widths = np.asarray(widths, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)
    outer_widths = np.asarray(outer_widths, dtype=np.float32)
    outer_heights = np.asarray(outer_heights, dtype=np.float32)
    logical_gate_ids = np.asarray(logical_gate_ids, dtype=np.int32)
    nominal_racing_line = np.asarray(nominal_racing_line, dtype=np.float32)
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    if arena_size is None:
        arena_size = (float(bounds_max[0] - bounds_min[0]), float(bounds_max[1] - bounds_min[1]))
    if (
        widths.shape != (n_gates,)
        or heights.shape != (n_gates,)
        or outer_widths.shape != (n_gates,)
        or outer_heights.shape != (n_gates,)
    ):
        raise ValueError("gate dimension arrays must have shape (num_gates,)")
    if len(openings) != n_gates:
        raise ValueError("openings must have one entry per gate")
    return GateCourse(
        name=name,
        centers=centers,
        normals=normals,
        radius_m=radius_m,
        widths=widths,
        heights=heights,
        outer_widths=outer_widths,
        outer_heights=outer_heights,
        logical_gate_ids=logical_gate_ids,
        openings=openings,
        nominal_racing_line=nominal_racing_line,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        arena_size=arena_size,
    )


def default_a2rl_course(radius_m: float = 0.55) -> GateCourse:
    """A compact slalom course suitable for fast local training smoke runs."""

    centers = np.array(
        [
            [0.0, 0.0, 1.25],
            [2.0, 0.4, 1.10],
            [4.0, -0.8, 1.35],
            [6.2, 0.8, 1.15],
            [8.4, -0.4, 1.30],
            [10.6, 0.7, 1.05],
            [12.8, 0.0, 1.25],
        ],
        dtype=np.float32,
    )
    return make_gate_course(centers, radius_m=radius_m, name="compact_slalom")


def arena_38m_stacked_course() -> GateCourse:
    """38 m square arena with 12 runtime gates, including two stacked double gates."""

    centers = np.array(
        [
            [18.0, 31.5, 1.0],
            [27.0, 29.0, 1.0],
            [29.0, 24.0, 1.0],
            [34.0, 16.0, 1.0],
            [27.0, 8.0, 1.0],
            [18.0, 18.0, 1.0],
            [11.0, 6.0, 2.5],
            [11.0, 6.0, 1.0],
            [10.0, 14.0, 1.0],
            [5.0, 22.0, 1.0],
            [9.0, 30.0, 1.0],
            [9.0, 30.0, 2.5],
        ],
        dtype=np.float32,
    )
    forwards = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.70710678, -0.70710678, 0.0],
            [0.33035042, -0.94385836, 0.0],
            [0.0, -1.0, 0.0],
            [-0.95358267, 0.30113137, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.19611614, 0.98058068, 0.0],
            [0.44721360, 0.89442719, 0.0],
            [0.64018440, 0.76822128, 0.0],
            [0.64018440, 0.76822128, 0.0],
        ],
        dtype=np.float32,
    )
    waypoints = np.array(
        [
            [16.0, 31.5, 1.0],
            [18.0, 31.5, 1.0],
            [27.0, 29.0, 1.0],
            [29.0, 24.0, 1.0],
            [34.0, 16.0, 1.0],
            [27.0, 8.0, 1.0],
            [18.0, 18.0, 1.0],
            [11.0, 6.0, 2.5],
            [11.0, 6.0, 1.0],
            [10.0, 14.0, 1.0],
            [5.0, 22.0, 1.0],
            [9.0, 30.0, 1.0],
            [9.0, 30.0, 2.5],
            [18.0, 31.5, 2.5],
        ],
        dtype=np.float32,
    )
    return make_gate_course(
        centers,
        normals=forwards,
        radius_m=0.75,
        widths=np.full((12,), 1.5, dtype=np.float32),
        heights=np.full((12,), 1.5, dtype=np.float32),
        outer_widths=np.full((12,), 2.7, dtype=np.float32),
        outer_heights=np.full((12,), 2.7, dtype=np.float32),
        logical_gate_ids=np.array([1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 10], dtype=np.int32),
        openings=("", "", "", "", "", "", "top", "bottom", "", "", "bottom", "top"),
        nominal_racing_line=waypoints,
        bounds_min=np.array([0.0, 0.0, 0.05], dtype=np.float32),
        bounds_max=np.array([38.0, 38.0, 5.0], dtype=np.float32),
        arena_size=(38.0, 38.0),
        name="arena_38m_stacked",
    )


def course_by_name(name: str) -> GateCourse:
    match name:
        case "arena_38m_stacked":
            return arena_38m_stacked_course()
        case "compact_slalom":
            return default_a2rl_course()
        case _:
            raise ValueError(f"Unknown course '{name}'")
