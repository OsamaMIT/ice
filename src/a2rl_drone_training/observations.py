from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from a2rl_drone_training.config import ObservationConfig


@dataclass(frozen=True)
class ObservationLayout:
    """Index layout for the flattened estimator-style observation vector."""

    config: ObservationConfig

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def core_dim(self) -> int:
        return self.config.core_dim

    @property
    def gate_context(self) -> int:
        return self.config.gate_context

    @property
    def gate_feature_dim(self) -> int:
        return self.config.gate_feature_dim


def quat_conjugate_xyzw(q: Array) -> Array:
    return jnp.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def quat_normalize_xyzw(q: Array) -> Array:
    return q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1.0e-6)


def quat_rotate_xyzw(q: Array, v: Array) -> Array:
    """Rotate vectors by xyzw quaternions."""

    q = quat_normalize_xyzw(q)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * jnp.cross(q_vec, v)
    return v + q_w * t + jnp.cross(q_vec, t)


def world_to_body_xyzw(q_body_to_world: Array, v_world: Array) -> Array:
    return quat_rotate_xyzw(quat_conjugate_xyzw(q_body_to_world), v_world)


def body_to_world_xyzw(q_body_to_world: Array, v_body: Array) -> Array:
    return quat_rotate_xyzw(q_body_to_world, v_body)


def yaw_to_quat_xyzw(yaw: Array) -> Array:
    half = 0.5 * yaw
    zeros = jnp.zeros_like(half)
    return jnp.stack([zeros, zeros, jnp.sin(half), jnp.cos(half)], axis=-1)


def quat_to_yaw_xyzw(q: Array) -> Array:
    q = quat_normalize_xyzw(q)
    x, y, z, w = jnp.moveaxis(q, -1, 0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)


def wrap_pi(angle: Array) -> Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def build_racing_observation(
    *,
    pos_world: Array,
    quat_xyzw: Array,
    vel_world: Array,
    ang_vel_world: Array,
    acc_world: Array,
    gate_centers_world: Array,
    gate_normals_world: Array,
    gate_counter: Array,
    total_gate_passes: int,
    last_action: Array,
    config: ObservationConfig,
    key: Array | None = None,
) -> Array:
    """Build a policy observation from IMU and gate-PNP-compatible state.

    The gate part intentionally uses relative gate pose, image-plane bearing, and a visibility
    channel because those are available from segmented gate masks plus a PNP solve.
    """

    quat_xyzw = quat_normalize_xyzw(quat_xyzw)
    vel_body = world_to_body_xyzw(quat_xyzw, vel_world)
    gyro_body = world_to_body_xyzw(quat_xyzw, ang_vel_world)
    gravity_world = jnp.array([0.0, 0.0, -9.81], dtype=pos_world.dtype)
    accel_body = world_to_body_xyzw(quat_xyzw, acc_world - gravity_world)
    gravity_body = world_to_body_xyzw(
        quat_xyzw, jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), vel_world.shape)
    )

    progress = (gate_counter / max(float(total_gate_passes), 1.0))[..., None]
    core = jnp.concatenate(
        [
            gyro_body / 12.0,
            accel_body / 30.0,
            quat_xyzw,
            vel_body / 10.0,
            gravity_body,
            last_action,
            progress,
        ],
        axis=-1,
    )

    offsets = jnp.arange(config.gate_context, dtype=gate_counter.dtype)[None, :]
    gate_ids = (gate_counter[:, None] + offsets) % gate_centers_world.shape[0]
    centers = gate_centers_world[gate_ids]
    normals = gate_normals_world[gate_ids]
    rel_world = centers - pos_world[:, None, :]
    quat_for_gates = quat_xyzw[:, None, :]
    rel_body = world_to_body_xyzw(quat_for_gates, rel_world)
    normal_body = world_to_body_xyzw(quat_for_gates, normals)

    distance = jnp.linalg.norm(rel_world, axis=-1, keepdims=True)
    rel_scaled = jnp.clip(rel_body / config.max_gate_distance, -1.5, 1.5)
    depth = rel_body[..., 0:1]
    denom = jnp.maximum(jnp.abs(depth), 0.25)
    bearing = jnp.concatenate([rel_body[..., 1:2] / denom, rel_body[..., 2:3] / denom], axis=-1)

    visible = _gate_visibility(depth, bearing, config)
    if key is not None and config.pnp_dropout_prob > 0.0:
        key, dropout_key = jax.random.split(key)
        keep = jax.random.bernoulli(
            dropout_key,
            p=1.0 - config.pnp_dropout_prob,
            shape=visible.shape,
        )
        visible = visible * keep.astype(visible.dtype)

    gate_features = jnp.concatenate(
        [
            rel_scaled,
            normal_body,
            jnp.clip(bearing, -2.0, 2.0),
            visible,
            jnp.clip(distance / config.max_gate_distance, 0.0, 2.0),
        ],
        axis=-1,
    )
    flat = jnp.concatenate([core, gate_features.reshape(pos_world.shape[0], -1)], axis=-1)

    if key is not None and config.additive_noise_std > 0.0:
        _, noise_key = jax.random.split(key)
        flat = flat + config.additive_noise_std * jax.random.normal(noise_key, flat.shape)
    return flat.astype(jnp.float32)


def _gate_visibility(depth: Array, bearing: Array, config: ObservationConfig) -> Array:
    half_x = 0.5 * config.camera_fov_x_rad
    half_y = 0.5 * config.camera_fov_y_rad
    max_x = jnp.tan(half_x)
    max_y = jnp.tan(half_y)
    in_front = depth > 0.05
    in_fov = (jnp.abs(bearing[..., 0:1]) < max_x) & (jnp.abs(bearing[..., 1:2]) < max_y)
    return (in_front & in_fov).astype(jnp.float32)
