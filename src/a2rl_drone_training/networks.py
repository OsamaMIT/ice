from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from a2rl_drone_training.config import NetworkConfig, ObservationConfig

Params = dict[str, Any]


def init_gc_actor(key: Array, obs_config: ObservationConfig, config: NetworkConfig) -> Params:
    """Initialize the gate-conditioned actor."""

    key_state, key_gate, key_fusion = jax.random.split(key, 3)
    state_encoder = init_mlp(
        key_state,
        obs_config.core_dim,
        config.state_hidden,
        config.state_latent_dim,
    )
    gate_encoder = init_mlp(
        key_gate,
        obs_config.gate_feature_dim,
        config.gate_hidden,
        config.gate_latent_dim,
    )
    fusion_in = config.state_latent_dim + obs_config.gate_context * config.gate_latent_dim
    fusion = init_mlp(
        key_fusion,
        fusion_in,
        config.fusion_hidden,
        config.action_dim,
        output_scale=0.01,
    )
    return {
        "state_encoder": state_encoder,
        "gate_encoder": gate_encoder,
        "fusion": fusion,
        "log_std": jnp.full((config.action_dim,), config.initial_log_std, dtype=jnp.float32),
    }


def gc_actor_apply(params: Params, obs: Array, obs_config: ObservationConfig) -> tuple[Array, Array]:
    core = obs[..., : obs_config.core_dim]
    gate_obs = obs[..., obs_config.core_dim :]
    gate_obs = gate_obs.reshape(
        *obs.shape[:-1],
        obs_config.gate_context,
        obs_config.gate_feature_dim,
    )
    state_latent = apply_mlp(params["state_encoder"], core)
    gate_latent = apply_mlp(params["gate_encoder"], gate_obs)
    gate_flat = gate_latent.reshape(*obs.shape[:-1], -1)
    fused = jnp.concatenate([state_latent, gate_flat], axis=-1)
    mean = apply_mlp(params["fusion"], fused, activate_final=False)
    log_std = jnp.clip(params["log_std"], -5.0, 2.0)
    log_std = jnp.broadcast_to(log_std, mean.shape)
    return mean, log_std


def init_critic(key: Array, critic_observation_dim: int, config: NetworkConfig) -> Params:
    return {
        "value": init_mlp(
            key,
            critic_observation_dim,
            config.critic_hidden,
            1,
            output_scale=1.0,
        )
    }


def critic_apply(params: Params, obs: Array) -> Array:
    return apply_mlp(params["value"], obs, activate_final=False).squeeze(-1)


def init_mlp(
    key: Array,
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    *,
    output_scale: float = 1.0,
) -> list[Params]:
    dims = (in_dim, *hidden_dims, out_dim)
    keys = jax.random.split(key, len(dims) - 1)
    layers: list[Params] = []
    for i, (layer_key, fan_in, fan_out) in enumerate(zip(keys, dims[:-1], dims[1:])):
        scale = output_scale if i == len(dims) - 2 else jnp.sqrt(2.0)
        weight = jax.random.normal(layer_key, (fan_in, fan_out), dtype=jnp.float32)
        weight = weight * scale / jnp.sqrt(float(fan_in))
        bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append({"w": weight, "b": bias})
    return layers


def apply_mlp(params: list[Params], x: Array, *, activate_final: bool = True) -> Array:
    y = x
    for i, layer in enumerate(params):
        y = y @ layer["w"] + layer["b"]
        if i < len(params) - 1 or activate_final:
            y = jnp.tanh(y)
    return y


def sample_action(
    actor_params: Params,
    obs: Array,
    key: Array,
    obs_config: ObservationConfig,
) -> tuple[Array, Array]:
    action, log_prob, _ = sample_action_with_mode(
        actor_params,
        obs,
        key,
        obs_config,
    )
    return action, log_prob


def sample_action_with_mode(
    actor_params: Params,
    obs: Array,
    key: Array,
    obs_config: ObservationConfig,
) -> tuple[Array, Array, Array]:
    mean, log_std = gc_actor_apply(actor_params, obs, obs_config)
    raw = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    action = jnp.tanh(raw)
    log_prob = squashed_gaussian_log_prob(raw, action, mean, log_std)
    return action, log_prob, jnp.tanh(mean)


def mode_action(actor_params: Params, obs: Array, obs_config: ObservationConfig) -> Array:
    mean, _ = gc_actor_apply(actor_params, obs, obs_config)
    return jnp.tanh(mean)


def evaluate_action_log_prob(
    actor_params: Params,
    obs: Array,
    action: Array,
    obs_config: ObservationConfig,
) -> tuple[Array, Array]:
    log_prob, entropy, _ = evaluate_action_distribution(
        actor_params,
        obs,
        action,
        obs_config,
    )
    return log_prob, entropy


def evaluate_action_distribution(
    actor_params: Params,
    obs: Array,
    action: Array,
    obs_config: ObservationConfig,
) -> tuple[Array, Array, Array]:
    mean, log_std = gc_actor_apply(actor_params, obs, obs_config)
    action = jnp.clip(action, -0.999999, 0.999999)
    raw = atanh(action)
    log_prob = squashed_gaussian_log_prob(raw, action, mean, log_std)
    entropy = gaussian_entropy(log_std)
    return log_prob, entropy, jnp.tanh(mean)


def squashed_gaussian_log_prob(raw: Array, action: Array, mean: Array, log_std: Array) -> Array:
    std = jnp.exp(log_std)
    normal_logp = -0.5 * jnp.square((raw - mean) / std) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    correction = jnp.log(jnp.maximum(1.0 - jnp.square(action), 1.0e-6))
    return jnp.sum(normal_logp - correction, axis=-1)


def gaussian_entropy(log_std: Array) -> Array:
    entropy = log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi))
    return jnp.sum(entropy, axis=-1)


def atanh(x: Array) -> Array:
    x = jnp.clip(x, -0.999999, 0.999999)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))
