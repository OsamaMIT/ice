from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import TrainingConfig
from a2rl_drone_training.course import GateCourse
from a2rl_drone_training.env import CrazyflowRacingEnv
from a2rl_drone_training.networks import critic_apply, init_critic, init_gc_actor, sample_action
from a2rl_drone_training.optim import adam_init
from a2rl_drone_training.ppo import Rollout, compute_gae, ppo_update


@dataclass
class TrainerState:
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    obs: Array
    key: Array
    env_steps: int = 0
    updates: int = 0
    episodes_completed: int = 0


class PPOTrainer:
    def __init__(
        self,
        config: TrainingConfig = TrainingConfig(),
        course: GateCourse | None = None,
        restore_checkpoint: Path | None = None,
    ):
        self.config = config
        self.env = CrazyflowRacingEnv(config.env, config.obs, course=course)
        key = jax.random.key(config.ppo.seed)
        key, actor_key, critic_key = jax.random.split(key, 3)
        actor_params = init_gc_actor(actor_key, config.obs, config.net)
        critic_params = init_critic(critic_key, config.obs, config.net)
        obs = self.env.reset(seed=config.ppo.seed)
        self.state = TrainerState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=adam_init(actor_params),
            critic_opt_state=adam_init(critic_params),
            obs=obs,
            key=key,
        )
        if restore_checkpoint is not None:
            self.load_checkpoint(restore_checkpoint)

    def train(self) -> None:
        cfg = self.config
        steps_per_update = cfg.env.num_envs * cfg.ppo.horizon
        target_env_steps = max(cfg.ppo.total_env_steps, steps_per_update)
        if self.state.env_steps >= target_env_steps:
            print(
                "checkpoint already satisfies target "
                f"env_steps={self.state.env_steps} target_env_steps={target_env_steps}"
            )
            return
        total_updates = int(np.ceil(target_env_steps / steps_per_update))
        start_update = self.state.env_steps // steps_per_update + 1
        checkpoint_dir = cfg.checkpoint_dir
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(self._device_log_line(total_updates, steps_per_update, target_env_steps))
        for update in range(start_update, total_updates + 1):
            progress_start = (update - 1) / max(total_updates - 1, 1)
            self.env.set_training_progress(progress_start)
            if update == start_update and self.state.env_steps > 0:
                self.state.obs = self.env.reset(seed=self.config.ppo.seed + update)
            rollout, rollout_metrics = self.collect_rollout()
            self.state.episodes_completed += int(rollout_metrics["episodes_completed"])
            last_value = critic_apply(self.state.critic_params, self.state.obs)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
            )
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                key,
                update_metrics,
            ) = ppo_update(
                actor_params=self.state.actor_params,
                critic_params=self.state.critic_params,
                actor_opt_state=self.state.actor_opt_state,
                critic_opt_state=self.state.critic_opt_state,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                key=self.state.key,
                obs_config=cfg.obs,
                config=cfg.ppo,
            )
            self.state.actor_params = actor_params
            self.state.critic_params = critic_params
            self.state.actor_opt_state = actor_opt_state
            self.state.critic_opt_state = critic_opt_state
            self.state.key = key
            self.state.updates = update
            self.state.env_steps += steps_per_update

            if update == 1 or update % cfg.ppo.log_interval == 0:
                self._log(update, total_updates, rollout_metrics, update_metrics)
            if checkpoint_dir is not None and (
                update == total_updates or update % cfg.ppo.log_interval == 0
            ):
                self.save_checkpoint(Path(checkpoint_dir) / f"checkpoint_{update:06d}.pkl")

    def collect_rollout(self) -> tuple[Rollout, dict[str, Any]]:
        obs_buf = []
        action_buf = []
        log_prob_buf = []
        value_buf = []
        reward_buf = []
        done_buf = []
        final_returns = []
        final_lengths = []
        miss_lengths = []
        crash_lengths = []
        bounds_lengths = []
        truncate_lengths = []
        finish_lengths = []
        pass_rates = []
        miss_rates = []
        crash_rates = []
        bounds_rates = []
        truncate_rates = []
        finish_rates = []
        gate_counters = []
        episode_gate_progress = []
        reset_gates = []
        random_start_rates = []
        focused_start_rates = []
        max_reset_gates = []
        window_scales = []
        speeds = []
        forward_speeds = []
        time_to_gates = []
        stall_rates = []
        reset_event_count = jnp.array(0.0, dtype=jnp.float32)
        reset_random_event_count = jnp.array(0.0, dtype=jnp.float32)
        reset_focused_event_count = jnp.array(0.0, dtype=jnp.float32)
        num_gates = self.env.course.num_gates
        gate_target_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_pass_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_miss_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_stop_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_stall_counts = jnp.zeros((num_gates,), dtype=jnp.float32)

        obs = self.state.obs
        for _ in range(self.config.ppo.horizon):
            self.state.key, action_key = jax.random.split(self.state.key)
            action, log_prob = sample_action(
                self.state.actor_params,
                obs,
                action_key,
                self.config.obs,
            )
            value = critic_apply(self.state.critic_params, obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated

            obs_buf.append(obs)
            action_buf.append(action)
            log_prob_buf.append(log_prob)
            value_buf.append(value)
            reward_buf.append(reward)
            done_buf.append(done)
            pass_rates.append(jnp.mean(info["passed_gate"].astype(jnp.float32)))
            miss_rates.append(jnp.mean(info["missed_gate"].astype(jnp.float32)))
            crash_rates.append(jnp.mean(info["crashed"].astype(jnp.float32)))
            bounds_rates.append(jnp.mean(info["out_of_bounds"].astype(jnp.float32)))
            truncate_rates.append(jnp.mean(truncated.astype(jnp.float32)))
            finish_rates.append(jnp.mean(info["finished"].astype(jnp.float32)))
            gate_counters.append(jnp.mean(info["gate_counter"].astype(jnp.float32)))
            episode_gate_progress.append(
                jnp.mean(info["episode_gate_progress"].astype(jnp.float32))
            )
            reset_gates.append(jnp.mean(info["reset_gate"].astype(jnp.float32)))
            random_start_rates.append(jnp.mean(info["random_start"].astype(jnp.float32)))
            focused_start_rates.append(jnp.mean(info["focused_start"].astype(jnp.float32)))
            max_reset_gates.append(jnp.mean(info["max_reset_gate"]))
            window_scales.append(jnp.mean(info["gate_window_scale"]))
            speeds.append(jnp.mean(info["speed"]))
            forward_speeds.append(jnp.mean(info["forward_speed"]))
            time_to_gates.append(jnp.mean(jnp.clip(info["time_to_gate"], 0.0, 30.0)))
            stall_rates.append(jnp.mean(info["stalled"].astype(jnp.float32)))
            reset_event_count = reset_event_count + jnp.sum(
                info["reset_occurred"].astype(jnp.float32)
            )
            reset_random_event_count = reset_random_event_count + jnp.sum(
                info["reset_random_start"].astype(jnp.float32)
            )
            reset_focused_event_count = reset_focused_event_count + jnp.sum(
                info["reset_focused_start"].astype(jnp.float32)
            )
            gate_one_hot = jax.nn.one_hot(info["gate_id"], num_gates, dtype=jnp.float32)
            gate_target_counts = gate_target_counts + jnp.sum(gate_one_hot, axis=0)
            gate_pass_counts = gate_pass_counts + jnp.sum(
                gate_one_hot * info["passed_gate"].astype(jnp.float32)[:, None],
                axis=0,
            )
            gate_miss_counts = gate_miss_counts + jnp.sum(
                gate_one_hot * info["missed_gate"].astype(jnp.float32)[:, None],
                axis=0,
            )
            stopped_on_gate = (
                info["missed_gate"]
                | info["crashed"]
                | info["out_of_bounds"]
                | truncated
            )
            gate_stop_counts = gate_stop_counts + jnp.sum(
                gate_one_hot * stopped_on_gate.astype(jnp.float32)[:, None],
                axis=0,
            )
            gate_stall_counts = gate_stall_counts + jnp.sum(
                gate_one_hot * info["stalled"].astype(jnp.float32)[:, None],
                axis=0,
            )
            final_returns.append(info["final_return"])
            final_lengths.append(info["final_length"])
            miss_lengths.append(jnp.where(info["missed_gate"], info["final_length"], jnp.nan))
            crash_lengths.append(jnp.where(info["crashed"], info["final_length"], jnp.nan))
            bounds_lengths.append(jnp.where(info["out_of_bounds"], info["final_length"], jnp.nan))
            truncate_lengths.append(jnp.where(truncated, info["final_length"], jnp.nan))
            finish_lengths.append(jnp.where(info["finished"], info["final_length"], jnp.nan))
            obs = next_obs

        self.state.obs = obs
        rollout = Rollout(
            obs=jnp.stack(obs_buf),
            actions=jnp.stack(action_buf),
            log_probs=jnp.stack(log_prob_buf),
            values=jnp.stack(value_buf),
            rewards=jnp.stack(reward_buf),
            dones=jnp.stack(done_buf),
        )
        mean_episode_steps = _nanmean_or_zero(jnp.concatenate(final_lengths))
        control_hz = max(float(self.config.env.control_hz), 1.0)
        metrics = {
            "mean_reward": float(jnp.mean(rollout.rewards)),
            "pass_rate": float(jnp.mean(jnp.stack(pass_rates))),
            "miss_rate": float(jnp.mean(jnp.stack(miss_rates))),
            "crash_rate": float(jnp.mean(jnp.stack(crash_rates))),
            "bounds_rate": float(jnp.mean(jnp.stack(bounds_rates))),
            "truncate_rate": float(jnp.mean(jnp.stack(truncate_rates))),
            "finish_rate": float(jnp.mean(jnp.stack(finish_rates))),
            "avg_gate_counter": float(jnp.mean(jnp.stack(gate_counters))),
            "avg_episode_gate_progress": float(jnp.mean(jnp.stack(episode_gate_progress))),
            "avg_reset_gate": float(jnp.mean(jnp.stack(reset_gates))),
            "random_start_rate": float(jnp.mean(jnp.stack(random_start_rates))),
            "focused_start_rate": float(jnp.mean(jnp.stack(focused_start_rates))),
            "max_reset_gate": float(jnp.mean(jnp.stack(max_reset_gates))),
            "gate_window_scale": float(jnp.mean(jnp.stack(window_scales))),
            "avg_speed": float(jnp.mean(jnp.stack(speeds))),
            "avg_forward_speed": float(jnp.mean(jnp.stack(forward_speeds))),
            "avg_time_to_gate": float(jnp.mean(jnp.stack(time_to_gates))),
            "stall_rate": float(jnp.mean(jnp.stack(stall_rates))),
            "reset_random_rate": _safe_ratio(reset_random_event_count, reset_event_count),
            "reset_focused_rate": _safe_ratio(reset_focused_event_count, reset_event_count),
            "reset_event_count": float(jax.device_get(reset_event_count)),
            "episodic_return": _nanmean_or_zero(jnp.concatenate(final_returns)),
            "episodic_length": mean_episode_steps,
            "episodic_time_s": mean_episode_steps / control_hz,
            "miss_time_s": _nanmean_or_zero(jnp.concatenate(miss_lengths)) / control_hz,
            "crash_time_s": _nanmean_or_zero(jnp.concatenate(crash_lengths)) / control_hz,
            "bounds_time_s": _nanmean_or_zero(jnp.concatenate(bounds_lengths)) / control_hz,
            "truncate_time_s": _nanmean_or_zero(jnp.concatenate(truncate_lengths)) / control_hz,
            "finish_time_s": _nanmean_or_zero(jnp.concatenate(finish_lengths)) / control_hz,
            "episodes_completed": float(_finite_count(jnp.concatenate(final_returns))),
            "gate_target_counts": np.asarray(
                jax.device_get(gate_target_counts),
                dtype=np.float32,
            ),
            "gate_pass_counts": np.asarray(
                jax.device_get(gate_pass_counts),
                dtype=np.float32,
            ),
            "gate_miss_counts": np.asarray(
                jax.device_get(gate_miss_counts),
                dtype=np.float32,
            ),
            "gate_stop_counts": np.asarray(
                jax.device_get(gate_stop_counts),
                dtype=np.float32,
            ),
            "gate_stall_counts": np.asarray(
                jax.device_get(gate_stall_counts),
                dtype=np.float32,
            ),
        }
        return rollout, metrics

    def save_checkpoint(self, path: Path) -> None:
        payload = {
            "actor_params": jax.device_get(self.state.actor_params),
            "critic_params": jax.device_get(self.state.critic_params),
            "actor_opt_state": jax.device_get(self.state.actor_opt_state),
            "critic_opt_state": jax.device_get(self.state.critic_opt_state),
            "key": jax.device_get(self.state.key),
            "env_steps": self.state.env_steps,
            "updates": self.state.updates,
            "episodes_completed": self.state.episodes_completed,
            "config": self.config,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        required = ("actor_params", "critic_params", "actor_opt_state", "critic_opt_state")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Checkpoint {path} is missing required fields: {missing}")

        _validate_checkpoint_observation_compatibility(path, payload, self.config)
        self.state.actor_params = jax.device_put(payload["actor_params"])
        self.state.critic_params = jax.device_put(payload["critic_params"])
        self.state.actor_opt_state = jax.device_put(payload["actor_opt_state"])
        self.state.critic_opt_state = jax.device_put(payload["critic_opt_state"])
        self.state.env_steps = int(payload.get("env_steps", 0))
        self.state.updates = int(payload.get("updates", 0))
        self.state.episodes_completed = int(payload.get("episodes_completed", 0))
        if "key" in payload:
            self.state.key = jax.device_put(payload["key"])
        print(
            "restored checkpoint "
            f"path={path} "
            f"env_steps={self.state.env_steps} "
            f"updates={self.state.updates} "
            f"episodes_completed={self.state.episodes_completed}"
        )

    def _log(
        self,
        update: int,
        total_updates: int,
        rollout_metrics: dict[str, Any],
        update_metrics: dict[str, Array],
    ) -> None:
        scalars = {name: float(value) for name, value in update_metrics.items()}
        steps_per_update = self.config.env.num_envs * self.config.ppo.horizon
        target_steps = max(self.config.ppo.total_env_steps, steps_per_update)
        progress = 100.0 * min(self.state.env_steps / max(target_steps, 1), 1.0)
        steps_left = max(target_steps - self.state.env_steps, 0)
        updates_left = max(total_updates - update, 0)
        episodes_completed = int(rollout_metrics["episodes_completed"])
        if rollout_metrics["episodic_length"] > 0.0:
            estimated_episodes_left = steps_left / rollout_metrics["episodic_length"]
            episodes_left_text = f"{estimated_episodes_left:.1f}"
        else:
            episodes_left_text = "n/a"
        print(
            "update "
            f"{update}/{total_updates} "
            f"progress={progress:.1f}% "
            f"steps={self.state.env_steps} "
            f"steps_left={steps_left} "
            f"updates_left={updates_left} "
            f"episodes_rollout={episodes_completed} "
            f"episodes_total={self.state.episodes_completed} "
            f"est_episodes_left={episodes_left_text} "
            f"gate_window={rollout_metrics['gate_window_scale']:.2f} "
            f"max_reset_gate={rollout_metrics['max_reset_gate']:.1f} "
            f"reward={rollout_metrics['mean_reward']:.3f} "
            f"return={rollout_metrics['episodic_return']:.3f} "
            f"pass={rollout_metrics['pass_rate']:.3f} "
            f"miss={rollout_metrics['miss_rate']:.3f} "
            f"crash={rollout_metrics['crash_rate']:.3f} "
            f"bounds={rollout_metrics['bounds_rate']:.3f} "
            f"trunc={rollout_metrics['truncate_rate']:.3f} "
            f"ep_time={rollout_metrics['episodic_time_s']:.2f}s "
            f"miss_time={rollout_metrics['miss_time_s']:.2f}s "
            f"crash_time={rollout_metrics['crash_time_s']:.2f}s "
            f"bounds_time={rollout_metrics['bounds_time_s']:.2f}s "
            f"trunc_time={rollout_metrics['truncate_time_s']:.2f}s "
            f"finish_time={rollout_metrics['finish_time_s']:.2f}s "
            f"gate={rollout_metrics['avg_gate_counter']:.2f} "
            f"reset_gate={rollout_metrics['avg_reset_gate']:.2f} "
            f"start_rand={rollout_metrics['random_start_rate']:.2f} "
            f"focus_rand={rollout_metrics['focused_start_rate']:.2f} "
            f"reset_rand={rollout_metrics['reset_random_rate']:.2f} "
            f"reset_focus={rollout_metrics['reset_focused_rate']:.2f} "
            f"resets={rollout_metrics['reset_event_count']:.0f} "
            f"gate_delta={rollout_metrics['avg_episode_gate_progress']:.2f} "
            f"speed={rollout_metrics['avg_speed']:.2f} "
            f"fwd_speed={rollout_metrics['avg_forward_speed']:.2f} "
            f"ttg={rollout_metrics['avg_time_to_gate']:.2f} "
            f"stall={rollout_metrics['stall_rate']:.3f} "
            f"finish={rollout_metrics['finish_rate']:.3f} "
            f"pi_loss={scalars.get('policy_loss', 0.0):.3f} "
            f"v_loss={scalars.get('value_loss', 0.0):.3f} "
            f"kl={scalars.get('approx_kl', 0.0):.4f}"
        )
        print(_gate_diagnostic_line(rollout_metrics))

    def _device_log_line(
        self,
        total_updates: int,
        steps_per_update: int,
        target_env_steps: int,
    ) -> str:
        actor_leaf = jax.tree_util.tree_leaves(self.state.actor_params)[0]
        devices = sorted(str(device) for device in actor_leaf.devices())
        return (
            "training "
            f"device_config={self.config.env.device} "
            f"jax_backend={jax.default_backend()} "
            f"param_devices={devices} "
            f"num_envs={self.config.env.num_envs} "
            f"start_env_steps={self.state.env_steps} "
            f"updates={total_updates} "
            f"steps_per_update={steps_per_update} "
            f"target_env_steps={target_env_steps}"
        )


def _nanmean_or_zero(x: Array) -> float:
    arr = np.asarray(jax.device_get(x), dtype=np.float32)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _finite_count(x: Array) -> int:
    arr = np.asarray(jax.device_get(x), dtype=np.float32)
    return int(np.isfinite(arr).sum())


def _validate_checkpoint_observation_compatibility(
    path: Path,
    payload: dict[str, Any],
    config: TrainingConfig,
) -> None:
    checkpoint_config = payload.get("config")
    if checkpoint_config is None:
        return

    checkpoint_obs = getattr(checkpoint_config, "obs", None)
    checkpoint_obs_dim = getattr(checkpoint_obs, "dim", None)
    if checkpoint_obs_dim is None or checkpoint_obs_dim == config.obs.dim:
        return

    raise ValueError(
        "Checkpoint observation dimension is incompatible with this run: "
        f"path={path} "
        f"checkpoint_obs_dim={checkpoint_obs_dim} "
        f"current_obs_dim={config.obs.dim}. "
        "Start a fresh time-independent run or restore a checkpoint created "
        "with the same observation architecture."
    )


def _safe_ratio(numerator: Array, denominator: Array) -> float:
    num = float(jax.device_get(numerator))
    den = float(jax.device_get(denominator))
    if den <= 0.0:
        return 0.0
    return num / den


def _format_gate_counts(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float32)
    return ",".join(f"{index + 1}:{int(round(value))}" for index, value in enumerate(arr))


def _gate_diagnostic_line(rollout_metrics: dict[str, Any]) -> str:
    target = np.asarray(rollout_metrics.get("gate_target_counts", []), dtype=np.float32)
    passed = np.asarray(rollout_metrics.get("gate_pass_counts", []), dtype=np.float32)
    missed = np.asarray(rollout_metrics.get("gate_miss_counts", []), dtype=np.float32)
    stopped = np.asarray(rollout_metrics.get("gate_stop_counts", []), dtype=np.float32)
    stalled = np.asarray(rollout_metrics.get("gate_stall_counts", []), dtype=np.float32)
    if target.size == 0:
        return "gate_diag unavailable"

    if np.max(missed) > 0.0:
        weak_index = int(np.argmax(missed))
        weak_reason = "miss"
    elif np.max(stopped) > 0.0:
        weak_index = int(np.argmax(stopped))
        weak_reason = "stop"
    else:
        unresolved_steps = np.maximum(target - passed * 50.0, 0.0)
        weak_index = int(np.argmax(unresolved_steps))
        weak_reason = "time"

    return (
        "gate_diag "
        f"weak=G{weak_index + 1}:{weak_reason} "
        f"target={_format_gate_counts(target)} "
        f"pass={_format_gate_counts(passed)} "
        f"miss={_format_gate_counts(missed)} "
        f"stop={_format_gate_counts(stopped)} "
        f"stall={_format_gate_counts(stalled)}"
    )
