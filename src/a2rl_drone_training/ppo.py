from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from a2rl_drone_training.config import ObservationConfig, PPOConfig
from a2rl_drone_training.networks import critic_apply, evaluate_action_log_prob
from a2rl_drone_training.optim import adam_update


class Rollout(NamedTuple):
    obs: Array
    actions: Array
    log_probs: Array
    values: Array
    rewards: Array
    dones: Array


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    last_value: Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Generalized advantage estimation over [time, env] arrays."""

    def scan_step(
        carry: tuple[Array, Array],
        transition: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], Array]:
        next_advantage, next_value = carry
        reward, value, done = transition
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        return (advantage, value), advantage

    _, advantages_rev = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    *,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    rollout: Rollout,
    advantages: Array,
    returns: Array,
    key: Array,
    obs_config: ObservationConfig,
    config: PPOConfig,
):
    batch = _flatten_rollout(rollout, advantages, returns)
    advantages_flat = batch["advantages"]
    advantages_flat = (advantages_flat - jnp.mean(advantages_flat)) / (
        jnp.std(advantages_flat) + 1.0e-8
    )
    batch["advantages"] = advantages_flat
    n_samples = batch["obs"].shape[0]
    minibatch_size = max(n_samples // config.minibatches, 1)
    metrics_accum = []

    for _ in range(config.update_epochs):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, n_samples)
        for start in range(0, n_samples, minibatch_size):
            mb_idx = permutation[start : start + minibatch_size]
            minibatch = {name: value[mb_idx] for name, value in batch.items()}
            (loss, metrics), grads = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(
                actor_params,
                critic_params,
                minibatch,
                obs_config,
                config,
            )
            actor_grads, critic_grads = grads
            actor_params, actor_opt_state, actor_grad_norm = adam_update(
                actor_params,
                actor_grads,
                actor_opt_state,
                lr=config.actor_lr,
                max_grad_norm=config.max_grad_norm,
            )
            critic_params, critic_opt_state, critic_grad_norm = adam_update(
                critic_params,
                critic_grads,
                critic_opt_state,
                lr=config.critic_lr,
                max_grad_norm=config.max_grad_norm,
            )
            metrics = {
                **metrics,
                "loss": loss,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
            }
            metrics_accum.append(metrics)

    metrics_mean = _mean_metrics(metrics_accum)
    return actor_params, critic_params, actor_opt_state, critic_opt_state, key, metrics_mean


@partial(jax.jit, static_argnames=("obs_config", "config"))
def _loss(actor_params, critic_params, batch, obs_config: ObservationConfig, config: PPOConfig):
    log_probs, entropy = evaluate_action_log_prob(
        actor_params,
        batch["obs"],
        batch["actions"],
        obs_config,
    )
    values = critic_apply(critic_params, batch["obs"])
    log_ratio = log_probs - batch["old_log_probs"]
    ratio = jnp.exp(log_ratio)
    unclipped_policy = ratio * batch["advantages"]
    clipped_policy = jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch[
        "advantages"
    ]
    policy_loss = -jnp.mean(jnp.minimum(unclipped_policy, clipped_policy))

    value_unclipped = jnp.square(values - batch["returns"])
    value_clipped = batch["old_values"] + jnp.clip(
        values - batch["old_values"],
        -config.value_clip_coef,
        config.value_clip_coef,
    )
    value_loss = 0.5 * jnp.mean(
        jnp.maximum(value_unclipped, jnp.square(value_clipped - batch["returns"]))
    )
    entropy_loss = jnp.mean(entropy)
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_coef).astype(jnp.float32))
    total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return total_loss, metrics


def _flatten_rollout(rollout: Rollout, advantages: Array, returns: Array) -> dict[str, Array]:
    return {
        "obs": _flatten_time_env(rollout.obs),
        "actions": _flatten_time_env(rollout.actions),
        "old_log_probs": _flatten_time_env(rollout.log_probs),
        "old_values": _flatten_time_env(rollout.values),
        "advantages": _flatten_time_env(advantages),
        "returns": _flatten_time_env(returns),
    }


def _flatten_time_env(x: Array) -> Array:
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))


def _mean_metrics(metrics: list[dict[str, Array]]) -> dict[str, Array]:
    if not metrics:
        return {}
    names = metrics[0].keys()
    return {name: jnp.mean(jnp.stack([m[name] for m in metrics])) for name in names}
