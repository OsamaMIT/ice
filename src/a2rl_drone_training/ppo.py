from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from a2rl_drone_training.config import ObservationConfig, PPOConfig
from a2rl_drone_training.networks import critic_apply, evaluate_action_distribution
from a2rl_drone_training.optim import adam_update


class Rollout(NamedTuple):
    obs: Array
    critic_obs: Array
    actions: Array
    log_probs: Array
    values: Array
    rewards: Array
    next_values: Array
    terminated: Array
    truncated: Array


@partial(jax.jit, static_argnames=("gamma", "gae_lambda"))
def compute_gae(
    rewards: Array,
    values: Array,
    next_values: Array,
    terminated: Array,
    truncated: Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """GAE with distinct terminal and time-limit semantics.

    ``next_values`` must be evaluated on the final pre-reset state. True terminals
    suppress that bootstrap; truncations retain it but end the recursive trace.
    """

    def scan_step(
        next_advantage: Array,
        transition: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array]:
        reward, value, next_value, terminal, truncation = transition
        bootstrap = 1.0 - terminal.astype(jnp.float32)
        episode_continues = 1.0 - (terminal | truncation).astype(jnp.float32)
        delta = reward + gamma * next_value * bootstrap - value
        advantage = delta + gamma * gae_lambda * episode_continues * next_advantage
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_step,
        jnp.zeros_like(values[-1]),
        (
            rewards[::-1],
            values[::-1],
            next_values[::-1],
            terminated[::-1],
            truncated[::-1],
        ),
    )
    advantages = advantages_rev[::-1]
    return advantages, advantages + values


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
    schedule_progress: float = 0.0,
):
    if config.minibatches < 1:
        raise ValueError("PPOConfig.minibatches must be at least 1")
    if config.update_epochs < 1:
        raise ValueError("PPOConfig.update_epochs must be at least 1")

    progress = float(jnp.clip(schedule_progress, 0.0, 1.0))
    actor_lr = _linear_schedule(config.actor_lr, config.actor_lr_end, progress)
    critic_lr = _linear_schedule(config.critic_lr, config.critic_lr_end, progress)
    entropy_progress = min(progress / max(config.entropy_decay_fraction, 1.0e-8), 1.0)
    entropy_coef = _linear_schedule(
        config.entropy_coef,
        config.entropy_coef_end,
        entropy_progress,
    )
    batch = _normalize_batch_advantages(_flatten_rollout(rollout, advantages, returns))
    n_samples = batch["obs"].shape[0]
    if n_samples % config.minibatches == 0:
        result = _ppo_update_scanned(
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            batch,
            key,
            jnp.asarray(actor_lr, dtype=jnp.float32),
            jnp.asarray(critic_lr, dtype=jnp.float32),
            jnp.asarray(entropy_coef, dtype=jnp.float32),
            obs_config,
            config,
        )
    else:
        result = _ppo_update_python_minibatches(
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            batch,
            key,
            actor_lr,
            critic_lr,
            entropy_coef,
            obs_config,
            config,
        )

    (
        actor_params,
        critic_params,
        actor_opt_state,
        critic_opt_state,
        key,
        metrics,
    ) = result
    actor_params, actor_opt_state, exploration_clipped_fraction = (
        constrain_actor_exploration(
            actor_params,
            actor_opt_state,
            config,
            progress,
        )
    )
    predictions = critic_apply(critic_params, batch["critic_obs"])
    return_variance = jnp.var(batch["returns"])
    explained_variance = jnp.where(
        return_variance > 1.0e-8,
        1.0 - jnp.var(batch["returns"] - predictions) / return_variance,
        0.0,
    )
    metrics = {
        **metrics,
        "explained_variance": explained_variance,
        "actor_lr": jnp.asarray(actor_lr),
        "critic_lr": jnp.asarray(critic_lr),
        "entropy_coef": jnp.asarray(entropy_coef),
        "exploration_std_ceiling": jnp.asarray(
            exploration_std_ceiling(config, progress)
        ),
        "exploration_clipped_fraction": exploration_clipped_fraction,
    }
    return actor_params, critic_params, actor_opt_state, critic_opt_state, key, metrics


def _linear_schedule(start: float, end: float, progress: float) -> float:
    return float(start + (end - start) * progress)


def exploration_std_ceiling(config: PPOConfig, schedule_progress: float) -> float:
    if config.exploration_std_start <= 0.0 or config.exploration_std_end <= 0.0:
        raise ValueError("Exploration standard-deviation endpoints must be positive")
    if config.exploration_std_floor <= 0.0:
        raise ValueError("exploration_std_floor must be positive")
    progress = float(jnp.clip(schedule_progress, 0.0, 1.0))
    decay_progress = min(
        progress / max(config.exploration_decay_fraction, 1.0e-8),
        1.0,
    )
    scheduled = _linear_schedule(
        config.exploration_std_start,
        config.exploration_std_end,
        decay_progress,
    )
    return max(float(config.exploration_std_floor), scheduled)


def constrain_actor_exploration(
    actor_params,
    actor_opt_state,
    config: PPOConfig,
    schedule_progress: float,
):
    """Keep trainable per-action exploration inside a cooling envelope."""

    ceiling = exploration_std_ceiling(config, schedule_progress)
    floor = min(float(config.exploration_std_floor), ceiling)
    log_std = actor_params["log_std"]
    constrained = jnp.clip(log_std, jnp.log(floor), jnp.log(ceiling))
    clipped = constrained != log_std
    actor_params = {**actor_params, "log_std": constrained}
    if actor_opt_state is not None:
        actor_opt_state = {
            **actor_opt_state,
            "m": {
                **actor_opt_state["m"],
                "log_std": jnp.where(
                    clipped,
                    jnp.zeros_like(actor_opt_state["m"]["log_std"]),
                    actor_opt_state["m"]["log_std"],
                ),
            },
            "v": {
                **actor_opt_state["v"],
                "log_std": jnp.where(
                    clipped,
                    jnp.zeros_like(actor_opt_state["v"]["log_std"]),
                    actor_opt_state["v"]["log_std"],
                ),
            },
        }
    return actor_params, actor_opt_state, jnp.mean(clipped.astype(jnp.float32))


def _normalize_batch_advantages(batch: dict[str, Array]) -> dict[str, Array]:
    advantages = batch["advantages"]
    return {
        **batch,
        "advantages": (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1.0e-8),
    }


def _ppo_update_python_minibatches(
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    batch: dict[str, Array],
    key: Array,
    actor_lr: float,
    critic_lr: float,
    entropy_coef: float,
    obs_config: ObservationConfig,
    config: PPOConfig,
):
    n_samples = batch["obs"].shape[0]
    minibatch_size = max(n_samples // config.minibatches, 1)
    metrics_accum = []
    epochs_completed = 0

    for _ in range(config.update_epochs):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, n_samples)
        epoch_metrics = []
        for start in range(0, n_samples, minibatch_size):
            mb_idx = permutation[start : start + minibatch_size]
            minibatch = {name: value[mb_idx] for name, value in batch.items()}
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                metrics,
            ) = _minibatch_update(
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                minibatch,
                jnp.asarray(actor_lr),
                jnp.asarray(critic_lr),
                jnp.asarray(entropy_coef),
                obs_config,
                config,
            )
            metrics_accum.append(metrics)
            epoch_metrics.append(metrics)
        epochs_completed += 1
        epoch_kl = float(jax.device_get(_mean_metrics(epoch_metrics)["approx_kl"]))
        if config.target_kl > 0.0 and epoch_kl > config.target_kl:
            break

    metrics_mean = _mean_metrics(metrics_accum)
    metrics_mean["epochs_completed"] = jnp.asarray(epochs_completed, dtype=jnp.float32)
    return actor_params, critic_params, actor_opt_state, critic_opt_state, key, metrics_mean


@partial(jax.jit, static_argnames=("obs_config", "config"))
def _ppo_update_scanned(
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    batch: dict[str, Array],
    key: Array,
    actor_lr: Array,
    critic_lr: Array,
    entropy_coef: Array,
    obs_config: ObservationConfig,
    config: PPOConfig,
):
    n_samples = batch["obs"].shape[0]
    minibatch_size = n_samples // config.minibatches
    zero_metrics = _empty_metrics()

    def epoch_step(carry, _: Array):
        actor_params, critic_params, actor_opt_state, critic_opt_state, key, active = carry
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, n_samples)
        minibatch_indices = permutation.reshape((config.minibatches, minibatch_size))

        def run_epoch(values):
            actor_params, critic_params, actor_opt_state, critic_opt_state = values

            def minibatch_step(minibatch_carry, mb_idx: Array):
                actor_params, critic_params, actor_opt_state, critic_opt_state = minibatch_carry
                minibatch = {name: value[mb_idx] for name, value in batch.items()}
                (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    metrics,
                ) = _minibatch_update(
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    minibatch,
                    actor_lr,
                    critic_lr,
                    entropy_coef,
                    obs_config,
                    config,
                )
                return (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                ), metrics

            updated, minibatch_metrics = jax.lax.scan(
                minibatch_step,
                (actor_params, critic_params, actor_opt_state, critic_opt_state),
                minibatch_indices,
            )
            epoch_metrics = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0),
                minibatch_metrics,
            )
            return (*updated, epoch_metrics)

        def skip_epoch(values):
            return (*values, zero_metrics)

        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            epoch_metrics,
        ) = jax.lax.cond(
            active,
            run_epoch,
            skip_epoch,
            (actor_params, critic_params, actor_opt_state, critic_opt_state),
        )
        executed = active.astype(jnp.float32)
        epoch_metrics = {**epoch_metrics, "epoch_executed": executed}
        below_kl = (config.target_kl <= 0.0) | (epoch_metrics["approx_kl"] <= config.target_kl)
        active = active & below_kl
        return (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            key,
            active,
        ), epoch_metrics

    (
        actor_params,
        critic_params,
        actor_opt_state,
        critic_opt_state,
        key,
        _,
    ), epoch_metrics = jax.lax.scan(
        epoch_step,
        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            key,
            jnp.asarray(True),
        ),
        jnp.arange(config.update_epochs),
    )
    epochs_completed = jnp.sum(epoch_metrics["epoch_executed"])
    denominator = jnp.maximum(epochs_completed, 1.0)
    metrics_mean = {
        name: jnp.sum(value, axis=0) / denominator
        for name, value in epoch_metrics.items()
        if name != "epoch_executed"
    }
    metrics_mean["epochs_completed"] = epochs_completed
    return actor_params, critic_params, actor_opt_state, critic_opt_state, key, metrics_mean


@partial(jax.jit, static_argnames=("obs_config", "config"))
def _minibatch_update(
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    minibatch: dict[str, Array],
    actor_lr: Array,
    critic_lr: Array,
    entropy_coef: Array,
    obs_config: ObservationConfig,
    config: PPOConfig,
):
    (loss, metrics), grads = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(
        actor_params,
        critic_params,
        minibatch,
        entropy_coef,
        obs_config,
        config,
    )
    actor_grads, critic_grads = grads
    actor_params, actor_opt_state, actor_grad_norm = adam_update(
        actor_params,
        actor_grads,
        actor_opt_state,
        lr=actor_lr,
        max_grad_norm=config.max_grad_norm,
    )
    critic_params, critic_opt_state, critic_grad_norm = adam_update(
        critic_params,
        critic_grads,
        critic_opt_state,
        lr=critic_lr,
        max_grad_norm=config.max_grad_norm,
    )
    return actor_params, critic_params, actor_opt_state, critic_opt_state, {
        **metrics,
        "loss": loss,
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
    }


@partial(jax.jit, static_argnames=("obs_config", "config"))
def _loss(
    actor_params,
    critic_params,
    batch,
    entropy_coef: Array,
    obs_config: ObservationConfig,
    config: PPOConfig,
):
    log_probs, entropy, mean_actions = evaluate_action_distribution(
        actor_params,
        batch["obs"],
        batch["actions"],
        obs_config,
    )
    values = critic_apply(critic_params, batch["critic_obs"])
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
    entropy_mean = jnp.mean(entropy)
    positive_advantages = jax.lax.stop_gradient(jnp.maximum(batch["advantages"], 0.0))
    alignment_error = jnp.mean(jnp.square(mean_actions - batch["actions"]), axis=-1)
    mean_action_alignment_loss = jnp.sum(
        positive_advantages * alignment_error
    ) / jnp.maximum(jnp.sum(positive_advantages), 1.0e-6)
    actor_loss = (
        policy_loss
        - entropy_coef * entropy_mean
        + config.mean_action_alignment_coef * mean_action_alignment_loss
    )
    critic_loss = config.value_coef * value_loss
    total_loss = actor_loss + critic_loss
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_coef).astype(jnp.float32))
    return total_loss, {
        "policy_loss": policy_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "critic_loss": critic_loss,
        "entropy": entropy_mean,
        "mean_action_alignment_loss": mean_action_alignment_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def _empty_metrics() -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        "policy_loss": zero,
        "actor_loss": zero,
        "value_loss": zero,
        "critic_loss": zero,
        "entropy": zero,
        "mean_action_alignment_loss": zero,
        "approx_kl": zero,
        "clip_fraction": zero,
        "loss": zero,
        "actor_grad_norm": zero,
        "critic_grad_norm": zero,
    }


def _flatten_rollout(rollout: Rollout, advantages: Array, returns: Array) -> dict[str, Array]:
    return {
        "obs": _flatten_time_env(rollout.obs),
        "critic_obs": _flatten_time_env(rollout.critic_obs),
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
    return {
        name: jnp.mean(jnp.stack([metric[name] for metric in metrics]))
        for name in metrics[0]
    }
