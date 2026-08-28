from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from a2rl_drone_training.config import RacingEnvConfig


REWARD_COMPONENT_NAMES = (
    "reward_potential_progress",
    "reward_potential_center",
    "reward_time",
    "reward_smoothness",
    "reward_gate",
    "reward_margin",
    "reward_finish",
    "reward_miss",
    "reward_crash",
    "reward_total",
)


def course_coordinate(
    pos: Array,
    gate_counter: Array,
    segment_starts: Array,
    segment_directions: Array,
    segment_lengths: Array,
    *,
    backtrack_limit: float = 0.0,
) -> Array:
    """Continuous, monotonic nominal coordinate measured in gates."""

    gate_id = gate_counter % segment_starts.shape[0]
    rel = pos - segment_starts[gate_id]
    along = jnp.sum(rel * segment_directions[gate_id], axis=-1)
    fraction = jnp.clip(
        along / jnp.maximum(segment_lengths[gate_id], 1.0e-6),
        -max(float(backtrack_limit), 0.0),
        1.0,
    )
    return gate_counter.astype(jnp.float32) + fraction


def progress_potential(
    *,
    pos: Array,
    gate_counter: Array,
    gate_centers: Array,
    gate_normals: Array,
    gate_right_axes: Array,
    segment_starts: Array,
    segment_directions: Array,
    segment_lengths: Array,
    track_scale: float,
    center_scale: float,
    plane_sigma_m: float,
    backtrack_limit: float = 0.0,
) -> tuple[Array, Array, Array]:
    gate_id = gate_counter % gate_centers.shape[0]
    rel_gate = pos - gate_centers[gate_id]
    plane_distance = jnp.sum(rel_gate * gate_normals[gate_id], axis=-1)
    lateral = jnp.sum(rel_gate * gate_right_axes[gate_id], axis=-1)
    vertical = rel_gate[..., 2]
    track = track_scale * course_coordinate(
        pos,
        gate_counter,
        segment_starts,
        segment_directions,
        segment_lengths,
        backtrack_limit=backtrack_limit,
    )
    localization = jnp.exp(-jnp.abs(plane_distance) / max(float(plane_sigma_m), 1.0e-6))
    center = -center_scale * (jnp.square(lateral) + jnp.square(vertical)) * localization
    return track + center, track, center


def rebase_progress_potential(
    potential: Array,
    track_potential: Array,
    reset_gate: Array,
    track_scale: float,
) -> tuple[Array, Array]:
    """Remove the episode's absolute gate-index baseline from a potential."""

    baseline = float(track_scale) * reset_gate.astype(jnp.float32)
    return potential - baseline, track_potential - baseline


def gate_clearance(
    lateral: Array,
    vertical: Array,
    width: Array,
    height: Array,
) -> Array:
    lateral_clearance = 0.5 * width - jnp.abs(lateral)
    vertical_clearance = 0.5 * height - jnp.abs(vertical)
    return jnp.minimum(lateral_clearance, vertical_clearance)


def unsafe_margin_penalty(
    clearance: Array,
    safe_margin: float,
    maximum_penalty: float,
) -> Array:
    safe_margin = max(float(safe_margin), 1.0e-6)
    deficit = jnp.clip((safe_margin - clearance) / safe_margin, 0.0, 1.0)
    return maximum_penalty * jnp.square(deficit)


def reward_v2(
    *,
    track_phi_before: Array,
    track_phi_after: Array,
    center_phi_before: Array,
    center_phi_after: Array,
    action: Array,
    previous_action: Array,
    clearance: Array,
    passed_gate: Array,
    finished: Array,
    missed_gate: Array,
    crashed: Array,
    dt: float,
    time_cost_scale: float,
    position_noise_std_m: float,
    config: RacingEnvConfig,
) -> tuple[Array, dict[str, Array]]:
    gamma = float(config.potential_gamma)
    reward_potential_progress = config.potential_reward_weight * (
        gamma * track_phi_after - track_phi_before
    )
    # Centering is evaluated against the same physical gate on both sides of
    # the transition. It is undiscounted so a stationary offset cannot earn a
    # bonus, and switching the active gate cannot release a stored penalty.
    reward_potential_center = config.potential_reward_weight * (
        center_phi_after - center_phi_before
    )
    reward_time = jnp.full_like(
        reward_potential_progress,
        -float(config.time_penalty) * float(time_cost_scale) * float(dt),
    )
    reward_smoothness = -config.action_delta_penalty * jnp.sum(
        jnp.square(action - previous_action), axis=-1
    )
    safe_margin = config.vehicle_radius_m + config.position_uncertainty_k * position_noise_std_m
    margin_cost = unsafe_margin_penalty(
        clearance,
        safe_margin,
        config.gate_margin_penalty,
    )
    reward_gate = passed_gate.astype(jnp.float32) * config.gate_pass_bonus
    reward_margin = -passed_gate.astype(jnp.float32) * margin_cost
    reward_finish = finished.astype(jnp.float32) * config.lap_finish_bonus
    reward_miss = -missed_gate.astype(jnp.float32) * config.v2_miss_penalty
    reward_crash = -crashed.astype(jnp.float32) * config.v2_crash_penalty
    components = {
        "reward_potential_progress": reward_potential_progress,
        "reward_potential_center": reward_potential_center,
        "reward_time": reward_time,
        "reward_smoothness": reward_smoothness,
        "reward_gate": reward_gate,
        "reward_margin": reward_margin,
        "reward_finish": reward_finish,
        "reward_miss": reward_miss,
        "reward_crash": reward_crash,
    }
    total = sum(components.values())
    return total.astype(jnp.float32), {**components, "reward_total": total}


def reward_v1(
    *,
    plane_before: Array,
    plane_after: Array,
    radial_before: Array,
    radial_after: Array,
    distance_before: Array,
    distance_after: Array,
    lookahead_distance_before: Array,
    lookahead_distance_after: Array,
    cross_radial: Array,
    velocity_alignment: Array,
    stalled: Array,
    action: Array,
    gate_id: Array,
    passed_gate: Array,
    missed_gate: Array,
    crashed: Array,
    finished: Array,
    course_radius_m: float,
    num_gates: int,
    config: RacingEnvConfig,
) -> tuple[Array, dict[str, Array]]:
    progress = jnp.clip(plane_after - plane_before, -0.25, 0.25)
    distance_progress = jnp.clip(distance_before - distance_after, -0.25, 0.25)
    lookahead_progress = jnp.clip(
        lookahead_distance_before - lookahead_distance_after, -0.25, 0.25
    )
    lookahead_active = (plane_after > -2.0) | passed_gate
    radial_progress = jnp.clip(radial_before - radial_after, -0.25, 0.25)
    center = jnp.exp(-2.5 * radial_after / course_radius_m)
    cross_center = jnp.exp(-2.5 * cross_radial / course_radius_m)
    positive_alignment = jnp.maximum(velocity_alignment, 0.0)
    reward_legacy_shaping = (
        config.plane_progress_reward * progress
        + config.distance_progress_reward * distance_progress
        + config.lookahead_progress_reward
        * lookahead_active.astype(jnp.float32)
        * lookahead_progress
        + config.radial_progress_reward * radial_progress
        + config.centerline_reward * center
        + config.crossing_center_reward * cross_center
        + config.path_alignment_reward * positive_alignment
        + config.stall_penalty * stalled.astype(jnp.float32)
        - config.action_penalty * jnp.sum(jnp.square(action), axis=-1)
    )
    gate_rank = (gate_id % num_gates).astype(jnp.float32)
    multiplier = jnp.minimum(
        config.late_gate_reward_base**gate_rank,
        config.late_gate_reward_cap,
    )
    reward_gate = passed_gate.astype(jnp.float32) * (
        config.gate_pass_bonus + config.centered_crossing_bonus * cross_center
    ) * multiplier
    reward_finish = finished.astype(jnp.float32) * config.lap_finish_bonus
    reward_miss = missed_gate.astype(jnp.float32) * config.miss_penalty
    reward_crash = crashed.astype(jnp.float32) * config.crash_penalty
    zero = jnp.zeros_like(reward_legacy_shaping)
    components = {
        "reward_potential_progress": reward_legacy_shaping,
        "reward_potential_center": zero,
        "reward_time": zero,
        "reward_smoothness": zero,
        "reward_gate": reward_gate,
        "reward_margin": zero,
        "reward_finish": reward_finish,
        "reward_miss": reward_miss,
        "reward_crash": reward_crash,
    }
    total = sum(components.values())
    return total.astype(jnp.float32), {**components, "reward_total": total}
