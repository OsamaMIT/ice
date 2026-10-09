"""Physical gate solids shared by planning and swept collision detection."""

from dataclasses import dataclass
import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.course import GateCourse


def fingerprint(value) -> str:
    def convert(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "__dataclass_fields__"):
            return {k: convert(getattr(x, k)) for k in x.__dataclass_fields__}
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [convert(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    return hashlib.sha256(
        json.dumps(convert(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, eq=False)
class FrameBoxes:
    centers: np.ndarray
    axes: np.ndarray  # rows: right, up, normal
    half_sizes: np.ndarray

    @partial(jax.jit, static_argnums=0)
    def clearance(self, positions, radius=0.0):
        """Euclidean signed point-to-solid distance minus a spherical radius."""
        p = jnp.asarray(positions)
        if not len(self.centers):
            return jnp.full(p.shape[:-1], jnp.inf)
        local = jnp.einsum(
            "bij,...bj->...bi",
            jnp.asarray(self.axes),
            p[..., None, :] - jnp.asarray(self.centers),
        )
        d = jnp.abs(local) - jnp.asarray(self.half_sizes)
        distance = jnp.linalg.norm(jnp.maximum(d, 0), axis=-1) + jnp.minimum(
            jnp.max(d, axis=-1), 0
        )
        return jnp.min(distance, axis=-1) - radius

    @partial(jax.jit, static_argnums=0)
    def swept_collision(self, start, end, radius):
        """Conservative swept sphere/OBB test (expanded-box slab intersection).

        Expansion is conservative at box corners; it cannot tunnel through a
        thin frame even when both step endpoints lie outside the solid.
        """
        if not len(self.centers):
            return jnp.zeros(jnp.asarray(start).shape[:-1], dtype=bool)
        axes, centers = jnp.asarray(self.axes), jnp.asarray(self.centers)
        a = jnp.einsum("bij,...bj->...bi", axes, start[..., None, :] - centers)
        b = jnp.einsum("bij,...bj->...bi", axes, end[..., None, :] - centers)
        delta = b - a
        half = jnp.asarray(self.half_sizes) + radius
        parallel = jnp.abs(delta) < 1e-10
        safe_delta = jnp.where(parallel, 1.0, delta)
        t1, t2 = (-half - a) / safe_delta, (half - a) / safe_delta
        lo = jnp.where(parallel, -jnp.inf, jnp.minimum(t1, t2))
        hi = jnp.where(parallel, jnp.inf, jnp.maximum(t1, t2))
        hit = jnp.maximum(jnp.max(lo, axis=-1), 0.0) <= jnp.minimum(
            jnp.min(hi, axis=-1), 1.0
        )
        hit &= ~jnp.any(parallel & (jnp.abs(a) > half), axis=-1)
        return jnp.any(hit, axis=-1)


def frame_boxes(course: GateCourse, depth: float = 0.1) -> FrameBoxes:
    """Union of outer rectangles minus union of all openings per logical gate.

    A 2D cell decomposition handles overlapping stacked frames without putting
    one opening's border across another opening.
    """
    if depth <= 0 or not np.isfinite(depth):
        raise ValueError("Frame depth must be finite and positive")
    if np.any(course.outer_widths < course.widths) or np.any(
        course.outer_heights < course.heights
    ):
        raise ValueError("Outer gate dimensions cannot be smaller than their openings")
    boxes, bases, sizes = [], [], []
    for logical in np.unique(course.logical_gate_ids):
        ids = np.flatnonzero(course.logical_gate_ids == logical)
        first = ids[0]
        n, right = course.normals[first], course.right_axes[first]
        up = np.cross(n, right)
        basis = np.stack((right, up, n))
        origin = course.centers[first]
        local = (course.centers[ids] - origin) @ basis.T
        if np.max(np.abs(local[:, 2])) > 1e-4 or np.any(
            np.abs(course.normals[ids] @ n) < 0.999
        ):
            raise ValueError("Openings sharing a logical gate must be coplanar")
        inner, outer = [], []
        for k, idx in enumerate(ids):
            for rectangles, width, height in (
                (inner, course.widths[idx], course.heights[idx]),
                (outer, course.outer_widths[idx], course.outer_heights[idx]),
            ):
                if width <= 0 or height <= 0:
                    raise ValueError("Gate dimensions must be positive")
                rectangles.append(
                    (
                        local[k, 0] - width / 2,
                        local[k, 0] + width / 2,
                        local[k, 1] - height / 2,
                        local[k, 1] + height / 2,
                    )
                )
        xs = sorted(set(v for r in inner + outer for v in r[:2]))
        ys = sorted(set(v for r in inner + outer for v in r[2:]))
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                x, y = (x0 + x1) / 2, (y0 + y1) / 2

                def inside(rects):
                    return any(a < x < b and c < y < d for a, b, c, d in rects)

                if inside(outer) and not inside(inner):
                    boxes.append(origin + right * x + up * y)
                    bases.append(basis)
                    sizes.append(((x1 - x0) / 2, (y1 - y0) / 2, depth / 2))
    return FrameBoxes(
        np.asarray(boxes, dtype=np.float32).reshape(-1, 3),
        np.asarray(bases, dtype=np.float32).reshape(-1, 3, 3),
        np.asarray(sizes, dtype=np.float32).reshape(-1, 3),
    )


def validate_openings(course, radius, margin):
    if radius < 0 or margin < 0 or not np.isfinite(radius + margin):
        raise ValueError(
            "Vehicle radius and tracking margin must be finite and nonnegative"
        )
    if np.any(np.minimum(course.widths, course.heights) <= 2 * (radius + margin)):
        raise ValueError(
            "A gate opening is infeasible after vehicle and tracking clearance"
        )
