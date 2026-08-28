from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import ObservationConfig


class NormalizationState(NamedTuple):
    count: Array
    mean: Array
    m2: Array


class NormalizationSpec(NamedTuple):
    running_mask: Array
    fixed_mean: Array
    fixed_scale: Array
    minimum_scale: Array


def actor_normalization_spec(config: ObservationConfig) -> NormalizationSpec:
    dim = config.dim
    running = np.zeros((dim,), dtype=bool)
    fixed_mean = np.zeros((dim,), dtype=np.float32)
    fixed_scale = np.ones((dim,), dtype=np.float32)
    minimum_scale = np.ones((dim,), dtype=np.float32)

    running[0:3] = True
    fixed_scale[0:3] = 2.0
    minimum_scale[0:3] = 0.1
    running[3:6] = True
    fixed_scale[3:6] = 9.81
    minimum_scale[3:6] = 0.5
    running[10:13] = True
    fixed_scale[10:13] = 3.0
    minimum_scale[10:13] = 0.1

    fixed_mean[20] = 0.5
    fixed_scale[20] = 0.5
    fixed_mean[21] = 0.5
    fixed_scale[21] = 0.5
    fixed_scale[22] = np.pi
    fixed_mean[23] = 0.5
    fixed_scale[23] = 0.5

    for gate_index in range(config.gate_context):
        start = config.core_dim + gate_index * config.gate_feature_dim
        running[start : start + 3] = True
        fixed_scale[start : start + 3] = 5.0
        minimum_scale[start : start + 3] = 0.1
        fixed_mean[start + 8] = 0.5
        fixed_scale[start + 8] = 0.5
        running[start + 9] = True
        fixed_scale[start + 9] = 10.0
        minimum_scale[start + 9] = 0.1

    return NormalizationSpec(
        running_mask=jnp.asarray(running),
        fixed_mean=jnp.asarray(fixed_mean),
        fixed_scale=jnp.asarray(fixed_scale),
        minimum_scale=jnp.asarray(minimum_scale),
    )


def init_normalization_state(config: ObservationConfig) -> NormalizationState:
    return NormalizationState(
        count=jnp.array(0.0, dtype=jnp.float32),
        mean=jnp.zeros((config.dim,), dtype=jnp.float32),
        m2=jnp.zeros((config.dim,), dtype=jnp.float32),
    )


@jax.jit
def update_normalization_state(
    state: NormalizationState,
    observations: Array,
    spec: NormalizationSpec,
) -> NormalizationState:
    observations = observations.reshape((-1, observations.shape[-1])).astype(jnp.float32)
    batch_count = jnp.asarray(observations.shape[0], dtype=jnp.float32)
    batch_mean = jnp.mean(observations, axis=0)
    batch_m2 = jnp.sum(jnp.square(observations - batch_mean), axis=0)
    total_count = state.count + batch_count
    delta = batch_mean - state.mean
    combined_mean = state.mean + delta * batch_count / jnp.maximum(total_count, 1.0)
    combined_m2 = (
        state.m2
        + batch_m2
        + jnp.square(delta)
        * state.count
        * batch_count
        / jnp.maximum(total_count, 1.0)
    )
    return NormalizationState(
        count=total_count,
        mean=jnp.where(spec.running_mask, combined_mean, state.mean),
        m2=jnp.where(spec.running_mask, combined_m2, state.m2),
    )


@jax.jit
def normalize_actor_observation(
    state: NormalizationState,
    observations: Array,
    spec: NormalizationSpec,
    clip: float,
    epsilon: float,
) -> Array:
    empirical_variance = state.m2 / jnp.maximum(state.count - 1.0, 1.0)
    empirical_scale = jnp.maximum(jnp.sqrt(empirical_variance + epsilon), spec.minimum_scale)
    has_statistics = state.count > 1.0
    running_mean = jnp.where(has_statistics, state.mean, spec.fixed_mean)
    running_scale = jnp.where(has_statistics, empirical_scale, spec.fixed_scale)
    mean = jnp.where(spec.running_mask, running_mean, spec.fixed_mean)
    scale = jnp.where(spec.running_mask, running_scale, spec.fixed_scale)
    normalized = (observations - mean) / jnp.maximum(scale, epsilon)
    return jnp.clip(normalized, -clip, clip).astype(jnp.float32)


def normalization_state_dict(state: NormalizationState) -> dict[str, np.ndarray | float]:
    return {
        "count": float(jax.device_get(state.count)),
        "mean": np.asarray(jax.device_get(state.mean), dtype=np.float32),
        "m2": np.asarray(jax.device_get(state.m2), dtype=np.float32),
    }


def normalization_state_from_dict(
    payload: dict[str, np.ndarray | float],
    config: ObservationConfig,
) -> NormalizationState:
    mean = np.asarray(payload["mean"], dtype=np.float32)
    m2 = np.asarray(payload["m2"], dtype=np.float32)
    if mean.shape != (config.dim,) or m2.shape != (config.dim,):
        raise ValueError(
            "Checkpoint actor-normalization dimension is incompatible with this run: "
            f"checkpoint_dim={mean.shape} current_dim={config.dim}"
        )
    return NormalizationState(
        count=jnp.asarray(payload["count"], dtype=jnp.float32),
        mean=jnp.asarray(mean),
        m2=jnp.asarray(m2),
    )
