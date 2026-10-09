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


def quat_multiply_xyzw(a: Array, b: Array) -> Array:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    vector = aw * bv + bw * av + jnp.cross(av, bv)
    scalar = aw * bw - jnp.sum(av * bv, axis=-1, keepdims=True)
    return quat_normalize_xyzw(jnp.concatenate([vector, scalar], axis=-1))


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
    remaining_time_fraction: Array,
    yaw_error: Array,
    stall_fraction: Array,
    config: ObservationConfig,
    key: Array | None = None,
) -> Array:
    """Build a policy observation from IMU and gate-PNP-compatible state.

    The gate part intentionally uses relative gate pose, image-plane bearing, and a visibility
    channel because those are available from segmented gate masks plus a PNP solve.
    """

    quat_xyzw = quat_normalize_xyzw(quat_xyzw)
    noise_scale = max(float(config.sensor_noise_scale), 0.0)
    if key is not None:
        (
            attitude_axis_key,
            attitude_angle_key,
            gyro_key,
            accel_key,
            velocity_key,
            gate_position_key,
            gate_normal_key,
            bearing_key,
            distance_key,
            dropout_key,
        ) = jax.random.split(key, 10)
        if noise_scale > 0.0 and config.attitude_noise_std_rad > 0.0:
            axis = jax.random.normal(attitude_axis_key, quat_xyzw[..., :3].shape)
            axis = axis / jnp.maximum(jnp.linalg.norm(axis, axis=-1, keepdims=True), 1.0e-6)
            angle = (
                noise_scale
                * config.attitude_noise_std_rad
                * jax.random.normal(attitude_angle_key, quat_xyzw[..., 3:4].shape)
            )
            delta_quat = jnp.concatenate(
                [axis * jnp.sin(0.5 * angle), jnp.cos(0.5 * angle)], axis=-1
            )
            quat_xyzw = quat_multiply_xyzw(quat_xyzw, delta_quat)
    vel_body = world_to_body_xyzw(quat_xyzw, vel_world)
    gyro_body = world_to_body_xyzw(quat_xyzw, ang_vel_world)
    gravity_world = jnp.array([0.0, 0.0, -9.81], dtype=pos_world.dtype)
    accel_body = world_to_body_xyzw(quat_xyzw, acc_world - gravity_world)
    gravity_body = world_to_body_xyzw(
        quat_xyzw, jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), vel_world.shape)
    )
    if key is not None and noise_scale > 0.0:
        gyro_body = gyro_body + noise_scale * config.gyro_noise_std_rad_s * jax.random.normal(
            gyro_key, gyro_body.shape
        )
        accel_body = accel_body + (
            noise_scale
            * config.specific_force_noise_std_m_s2
            * jax.random.normal(accel_key, accel_body.shape)
        )
        vel_body = vel_body + noise_scale * config.velocity_noise_std_m_s * jax.random.normal(
            velocity_key, vel_body.shape
        )

    progress = (gate_counter / max(float(total_gate_passes), 1.0))[..., None]
    core = jnp.concatenate(
        [
            gyro_body,
            accel_body,
            quat_xyzw,
            vel_body,
            gravity_body,
            last_action,
            progress,
            jnp.clip(remaining_time_fraction, 0.0, 1.0)[..., None],
            wrap_pi(yaw_error)[..., None],
            jnp.clip(stall_fraction, 0.0, 1.0)[..., None],
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

    if key is not None and noise_scale > 0.0:
        rel_body = rel_body + (
            noise_scale
            * config.gate_position_noise_std_m
            * jax.random.normal(gate_position_key, rel_body.shape)
        )
        normal_perturbation = jax.random.normal(gate_normal_key, normal_body.shape)
        normal_body = normal_body + (
            noise_scale
            * config.gate_normal_noise_std_rad
            * jnp.cross(normal_perturbation, normal_body)
        )
        normal_body = normal_body / jnp.maximum(
            jnp.linalg.norm(normal_body, axis=-1, keepdims=True), 1.0e-6
        )

    distance = jnp.linalg.norm(rel_body, axis=-1, keepdims=True)
    depth = rel_body[..., 0:1]
    denom = jnp.maximum(jnp.abs(depth), 0.25)
    bearing = jnp.concatenate([rel_body[..., 1:2] / denom, rel_body[..., 2:3] / denom], axis=-1)
    if key is not None and noise_scale > 0.0:
        bearing = bearing + (
            noise_scale
            * config.gate_bearing_noise_std
            * jax.random.normal(bearing_key, bearing.shape)
        )
        distance = distance + (
            noise_scale
            * config.gate_distance_noise_std_m
            * jax.random.normal(distance_key, distance.shape)
        )

    visible = _gate_visibility(depth, bearing, config)
    if key is not None and config.pnp_dropout_prob > 0.0:
        keep = jax.random.bernoulli(
            dropout_key,
            p=1.0 - config.pnp_dropout_prob,
            shape=visible.shape,
        )
        visible = visible * keep.astype(visible.dtype)

    gate_features = jnp.concatenate(
        [
            rel_body * visible,
            normal_body * visible,
            jnp.clip(bearing, -2.0, 2.0) * visible,
            visible,
            jnp.clip(distance, 0.0, 2.0 * config.max_gate_distance) * visible,
        ],
        axis=-1,
    )
    flat = jnp.concatenate([core, gate_features.reshape(pos_world.shape[0], -1)], axis=-1)

    return flat.astype(jnp.float32)


def build_privileged_observation(
    *,
    pos_world: Array,
    quat_xyzw: Array,
    vel_world: Array,
    ang_vel_world: Array,
    acc_world: Array,
    gate_centers_world: Array,
    gate_normals_world: Array,
    gate_right_axes_world: Array,
    gate_counter: Array,
    total_gate_passes: int,
    elapsed_steps: Array,
    max_episode_steps: int,
    last_action: Array,
    yaw_error: Array,
    gate_window_scale: float,
    started_local: Array,
) -> Array:
    gate_id = gate_counter % gate_centers_world.shape[0]
    next_gate_id = (gate_id + 1) % gate_centers_world.shape[0]
    rel_gate = gate_centers_world[gate_id] - pos_world
    rel_next_gate = gate_centers_world[next_gate_id] - pos_world
    position_from_gate = pos_world - gate_centers_world[gate_id]
    plane = jnp.sum(position_from_gate * gate_normals_world[gate_id], axis=-1)
    lateral = jnp.sum(position_from_gate * gate_right_axes_world[gate_id], axis=-1)
    vertical = position_from_gate[..., 2]
    progress = gate_counter / max(float(total_gate_passes), 1.0)
    elapsed = elapsed_steps / max(float(max_episode_steps), 1.0)
    features = jnp.concatenate(
        [
            pos_world / 20.0,
            quat_normalize_xyzw(quat_xyzw),
            vel_world / 10.0,
            ang_vel_world / 12.0,
            acc_world / 30.0,
            rel_gate / 25.0,
            gate_normals_world[gate_id],
            rel_next_gate / 25.0,
            progress[..., None],
            elapsed[..., None],
            last_action,
            (wrap_pi(yaw_error) / jnp.pi)[..., None],
            (plane / 25.0)[..., None],
            (lateral / 5.0)[..., None],
            (vertical / 5.0)[..., None],
            jnp.full_like(progress[..., None], gate_window_scale / 1.8),
            started_local.astype(jnp.float32)[..., None],
        ],
        axis=-1,
    )
    return features.astype(jnp.float32)


def _gate_visibility(depth: Array, bearing: Array, config: ObservationConfig) -> Array:
    half_x = 0.5 * config.camera_fov_x_rad
    half_y = 0.5 * config.camera_fov_y_rad
    max_x = jnp.tan(half_x)
    max_y = jnp.tan(half_y)
    in_front = depth > 0.05
    in_fov = (jnp.abs(bearing[..., 0:1]) < max_x) & (jnp.abs(bearing[..., 1:2]) < max_y)
    return (in_front & in_fov).astype(jnp.float32)
