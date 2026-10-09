"""Validated portable reference trajectories and gate-local interpolation."""

from dataclasses import dataclass
import json
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import fingerprint


@dataclass(eq=False)
class Reference:
    time: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    gate_indices: np.ndarray
    metadata: dict

    def __post_init__(self):
        n = len(self.time)
        if n < 2 or self.states.shape != (n, 17) or self.actions.shape != (n, 4):
            raise ValueError("Invalid reference array dimensions")
        if (
            not all(
                np.isfinite(x).all() for x in (self.time, self.states, self.actions)
            )
            or self.time[0] != 0
            or np.any(np.diff(self.time) <= 0)
        ):
            raise ValueError(
                "Reference must have finite states and increasing times starting at zero"
            )
        if np.any(np.abs(self.actions) > 1.00001) or not np.allclose(
            np.linalg.norm(self.states[:, 6:10], axis=1), 1, atol=1e-4
        ):
            raise ValueError(
                "Reference motor bounds or quaternion normalization invalid"
            )
        if (
            self.gate_indices.ndim != 1
            or np.any(np.diff(self.gate_indices) <= 0)
            or np.any(self.gate_indices < 1)
            or np.any(self.gate_indices >= n)
        ):
            raise ValueError("Invalid reference gate events")
        # Stable hemisphere for interpolation.
        self.states = self.states.copy()
        for i in range(1, n):
            if self.states[i - 1, 6:10] @ self.states[i, 6:10] < 0:
                self.states[i, 6:10] *= -1

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "time": self.time,
                "states": self.states,
                "actions": self.actions,
                "gate_indices": self.gate_indices,
                "metadata": self.metadata,
            }
        )

    @property
    def progress(self):
        return np.interp(
            self.time,
            np.r_[0, self.time[self.gate_indices]],
            np.arange(len(self.gate_indices) + 1),
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(
                f,
                time=self.time,
                states=self.states,
                position=self.states[:, :3],
                velocity=self.states[:, 3:6],
                quaternion=self.states[:, 6:10],
                angular_velocity=self.states[:, 10:13],
                motor_commands=self.actions,
                course_progress=self.progress,
                gate_indices=self.gate_indices,
                metadata=json.dumps(self.metadata, sort_keys=True),
                fingerprint=self.fingerprint,
            )

    @classmethod
    def load(cls, path, *, course_fingerprint=None, model_fingerprint=None):
        with np.load(path, allow_pickle=False) as f:
            ref = cls(
                f["time"],
                f["states"],
                f["motor_commands"],
                f["gate_indices"],
                json.loads(str(f["metadata"])),
            )
            if str(f["fingerprint"]) != ref.fingerprint:
                raise ValueError("Reference content fingerprint mismatch")
        if not ref.metadata.get("validated", False):
            raise ValueError("Reference has not passed dense feasibility validation")
        for name, expected in [
            ("course_fingerprint", course_fingerprint),
            ("model_fingerprint", model_fingerprint),
        ]:
            if expected is not None and ref.metadata.get(name) != expected:
                raise ValueError(f"Reference {name} mismatch")
        return ref

    @partial(jax.jit, static_argnums=0)
    def sample(self, t):
        """Cubic Hermite position interpolation; velocity/acceleration are derivatives."""
        times, states = jnp.asarray(self.time), jnp.asarray(self.states)
        requested_t = jnp.asarray(t)
        t = jnp.clip(t, times[0], times[-1])
        i = jnp.clip(
            jnp.searchsorted(times, t, side="right") - 1, 0, len(self.time) - 2
        )
        h = times[i + 1] - times[i]
        z = (t - times[i]) / h
        a, b = states[i], states[i + 1]
        p0, p1, v0, v1 = a[..., :3], b[..., :3], a[..., 3:6], b[..., 3:6]
        z, h = z[..., None], h[..., None]
        position = (
            (2 * z**3 - 3 * z**2 + 1) * p0
            + (z**3 - 2 * z**2 + z) * h * v0
            + (-2 * z**3 + 3 * z**2) * p1
            + (z**3 - z**2) * h * v1
        )
        velocity = (
            (6 * z * z - 6 * z) / h * p0
            + (3 * z * z - 4 * z + 1) * v0
            + (-6 * z * z + 6 * z) / h * p1
            + (3 * z * z - 2 * z) * v1
        )
        acceleration = (
            (12 * z - 6) / h**2 * p0
            + (6 * z - 4) / h * v0
            + (-12 * z + 6) / h**2 * p1
            + (6 * z - 2) / h * v1
        )
        # Extend past the finish at constant velocity so an MPC horizon does
        # not see a stationary position paired with a nonzero velocity target.
        outside = (requested_t - t)[..., None]
        position = position + velocity * outside
        acceleration = jnp.where(outside != 0, 0.0, acceleration)
        state = a * (1 - z) + b * z
        q = state[..., 6:10]
        state = (
            state.at[..., :3]
            .set(position)
            .at[..., 3:6]
            .set(velocity)
            .at[..., 6:10]
            .set(q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8))
        )
        action = jnp.asarray(self.actions)[i]
        return state, action, acceleration

    @partial(jax.jit, static_argnums=0)
    def project(self, position, gate_id):
        """Project onto line segments only within the active ordered gate segment."""
        points = jnp.asarray(self.states[:, :3])
        delta = points[1:] - points[:-1]
        alpha = jnp.clip(
            jnp.sum((position[..., None, :] - points[:-1]) * delta, axis=-1)
            / jnp.maximum(jnp.sum(delta * delta, axis=-1), 1e-10),
            0,
            1,
        )
        projected = points[:-1] + alpha[..., None] * delta
        gates = jnp.asarray(self.gate_indices)
        gate_id = jnp.minimum(gate_id, len(self.gate_indices) - 1)
        start = jnp.where(gate_id == 0, 0, gates[jnp.maximum(gate_id - 1, 0)])
        end = gates[gate_id]
        ids = jnp.arange(len(self.time) - 1)
        distance = jnp.sum((projected - position[..., None, :]) ** 2, axis=-1)
        distance = jnp.where(
            (ids >= start[..., None]) & (ids < end[..., None]), distance, jnp.inf
        )
        index = jnp.argmin(distance, axis=-1)
        fraction = jnp.take_along_axis(alpha, index[..., None], axis=-1)[..., 0]
        time = jnp.asarray(self.time)
        return time[index] + fraction * (time[index + 1] - time[index])
