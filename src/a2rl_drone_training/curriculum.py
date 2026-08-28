from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import CurriculumConfig


PHASES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class CurriculumParameters:
    phase: str
    gate1_fraction: float
    gate_window_scale: float
    time_cost_scale: float
    prioritized_local_starts: bool
    local_gate_probabilities: np.ndarray


@dataclass
class CurriculumState:
    phase_index: int
    gate_window_scale: float
    time_cost_scale: float
    qualification_streak: int
    updates: int
    training_passes: np.ndarray
    training_failures: np.ndarray
    evaluation_attempts: np.ndarray
    evaluation_passes: np.ndarray
    latest_evaluation_attempts: np.ndarray
    latest_evaluation_passes: np.ndarray
    evaluation_episodes: int
    evaluation_finishes: int
    evaluation_runs: int
    skill_attempts: np.ndarray
    skill_passes: np.ndarray
    skill_strict_passes: np.ndarray
    skill_audits: int
    skill_recent_attempts: np.ndarray
    skill_recent_passes: np.ndarray
    skill_recent_strict_passes: np.ndarray
    skill_recent_index: int
    skill_recent_count: int
    phase_a_priority_ready: bool


def _reset_local_start_mask(
    key: Array,
    reset_mask: Array,
    local_fraction: float,
) -> Array:
    count_key, rank_key = jax.random.split(key)
    reset_count = jnp.sum(reset_mask.astype(jnp.int32))
    expected = jnp.clip(float(local_fraction), 0.0, 1.0) * reset_count
    base_count = jnp.floor(expected).astype(jnp.int32)
    stochastic_remainder = jax.random.bernoulli(count_key, expected - base_count)
    local_count = base_count + stochastic_remainder.astype(jnp.int32)
    priority = jnp.where(
        reset_mask,
        jax.random.uniform(rank_key, reset_mask.shape),
        2.0,
    )
    ranks = jnp.argsort(jnp.argsort(priority))
    return reset_mask & (ranks < local_count)


def sample_reset_gates(
    *,
    key: Array,
    reset_mask: Array,
    current_reset_gate: Array,
    gate1_fraction: float,
    local_gate_probabilities: Array,
    prioritized: bool,
    evaluation: bool,
) -> tuple[Array, Array]:
    """Single reset sampler used by both training and strict evaluation."""

    reset_mask = jnp.asarray(reset_mask, dtype=jnp.bool_)
    current_reset_gate = jnp.asarray(current_reset_gate, dtype=jnp.int32)
    if evaluation:
        reset_gate = jnp.where(reset_mask, 0, current_reset_gate)
        return reset_gate, jnp.zeros_like(reset_mask)

    branch_key, gate_key, offset_key = jax.random.split(key, 3)
    local_mask = _reset_local_start_mask(
        branch_key,
        reset_mask,
        1.0 - float(gate1_fraction),
    )
    num_local_gates = int(local_gate_probabilities.shape[0])
    if num_local_gates < 1:
        reset_gate = jnp.where(reset_mask, 0, current_reset_gate)
        return reset_gate, jnp.zeros_like(reset_mask)

    if prioritized:
        probabilities = local_gate_probabilities / jnp.maximum(
            jnp.sum(local_gate_probabilities), 1.0e-8
        )
        sampled_local = jax.random.categorical(
            gate_key,
            jnp.log(jnp.maximum(probabilities, 1.0e-8)),
            shape=reset_mask.shape,
        ).astype(jnp.int32) + 1
    else:
        local_priority = jnp.where(
            local_mask,
            jax.random.uniform(gate_key, reset_mask.shape),
            2.0,
        )
        local_ranks = jnp.argsort(jnp.argsort(local_priority))
        offset = jax.random.randint(offset_key, (), 0, num_local_gates)
        sampled_local = ((local_ranks + offset) % num_local_gates).astype(jnp.int32) + 1

    sampled_gate = jnp.where(local_mask, sampled_local, 0)
    reset_gate = jnp.where(reset_mask, sampled_gate, current_reset_gate)
    return reset_gate, local_mask


class CurriculumController:
    def __init__(self, config: CurriculumConfig, num_gates: int):
        self.config = config
        self.num_gates = int(num_gates)
        if config.skill_qualification_window < 1:
            raise ValueError("skill_qualification_window must be at least 1")
        window_shape = (config.skill_qualification_window, num_gates)
        self.state = CurriculumState(
            phase_index=0,
            gate_window_scale=float(config.phase_a_window_scale),
            time_cost_scale=0.0,
            qualification_streak=0,
            updates=0,
            training_passes=np.zeros((num_gates,), dtype=np.float64),
            training_failures=np.zeros((num_gates,), dtype=np.float64),
            evaluation_attempts=np.zeros((num_gates,), dtype=np.float64),
            evaluation_passes=np.zeros((num_gates,), dtype=np.float64),
            latest_evaluation_attempts=np.zeros((num_gates,), dtype=np.float64),
            latest_evaluation_passes=np.zeros((num_gates,), dtype=np.float64),
            evaluation_episodes=0,
            evaluation_finishes=0,
            evaluation_runs=0,
            skill_attempts=np.zeros((num_gates,), dtype=np.float64),
            skill_passes=np.zeros((num_gates,), dtype=np.float64),
            skill_strict_passes=np.zeros((num_gates,), dtype=np.float64),
            skill_audits=0,
            skill_recent_attempts=np.zeros(window_shape, dtype=np.float64),
            skill_recent_passes=np.zeros(window_shape, dtype=np.float64),
            skill_recent_strict_passes=np.zeros(window_shape, dtype=np.float64),
            skill_recent_index=0,
            skill_recent_count=0,
            phase_a_priority_ready=False,
        )

    @property
    def phase(self) -> str:
        return PHASES[self.state.phase_index]

    def parameters(self) -> CurriculumParameters:
        if not self.config.enabled:
            return CurriculumParameters(
                phase="D",
                gate1_fraction=1.0,
                gate_window_scale=1.0,
                time_cost_scale=1.0,
                prioritized_local_starts=False,
                local_gate_probabilities=self._uniform_local_probabilities(),
            )
        gate1_fraction = (
            self.config.phase_a_gate1_fraction,
            self.config.phase_b_gate1_fraction,
            self.config.phase_c_gate1_fraction,
            self.config.phase_d_gate1_fraction,
        )[self.state.phase_index]
        probabilities = self.local_gate_probabilities()
        prioritized = self.state.phase_index >= 2 or (
            self.state.phase_index == 0 and self._phase_a_priority_ready()
        )
        return CurriculumParameters(
            phase=self.phase,
            gate1_fraction=float(gate1_fraction),
            gate_window_scale=float(self.state.gate_window_scale),
            time_cost_scale=float(self.state.time_cost_scale),
            prioritized_local_starts=prioritized,
            local_gate_probabilities=probabilities,
        )

    def step(self) -> None:
        self.state.updates += 1
        target_window = (
            self.config.phase_a_window_scale,
            self.config.phase_b_window_scale,
            self.config.phase_c_window_scale,
            self.config.phase_d_window_scale,
        )[self.state.phase_index]
        target_time = (
            0.0,
            self.config.phase_b_time_cost_scale,
            self.config.phase_c_time_cost_scale,
            1.0,
        )[self.state.phase_index]
        self.state.gate_window_scale = _move_toward(
            self.state.gate_window_scale,
            target_window,
            self.config.gate_window_rate_limit,
        )
        self.state.time_cost_scale = _move_toward(
            self.state.time_cost_scale,
            target_time,
            self.config.time_cost_rate_limit,
        )

    def record_training(self, passes: np.ndarray, failures: np.ndarray) -> None:
        self.state.training_passes += np.asarray(passes, dtype=np.float64)
        self.state.training_failures += np.asarray(failures, dtype=np.float64)

    def record_evaluation(
        self,
        *,
        attempts: np.ndarray,
        passes: np.ndarray,
        episodes: int,
        finishes: int,
    ) -> bool:
        attempts = self._validate_gate_array(attempts, "Evaluation attempts")
        passes = self._validate_gate_array(passes, "Evaluation passes")
        self.state.evaluation_attempts += attempts
        self.state.evaluation_passes += passes
        self.state.latest_evaluation_attempts = attempts.copy()
        self.state.latest_evaluation_passes = passes.copy()
        self.state.evaluation_episodes += int(episodes)
        self.state.evaluation_finishes += int(finishes)
        self.state.evaluation_runs += 1
        if not self.config.enabled or self.state.phase_index >= len(PHASES) - 1:
            return False
        if self.state.phase_index == 0:
            return False
        return self._update_qualification(self._official_evaluation_qualified())

    def record_skill_evaluation(
        self,
        *,
        attempts: np.ndarray,
        passes: np.ndarray,
        strict_passes: np.ndarray | None = None,
    ) -> bool:
        attempts = self._validate_gate_array(attempts, "Skill-audit attempts")
        passes = self._validate_gate_array(passes, "Skill-audit passes")
        if strict_passes is None:
            strict_passes = passes
        strict_passes = self._validate_gate_array(
            strict_passes,
            "Skill-audit strict passes",
        )
        self.state.skill_attempts += attempts
        self.state.skill_passes += passes
        self.state.skill_strict_passes += strict_passes
        self.state.skill_audits += 1
        index = self.state.skill_recent_index
        self.state.skill_recent_attempts[index] = attempts
        self.state.skill_recent_passes[index] = passes
        self.state.skill_recent_strict_passes[index] = strict_passes
        self.state.skill_recent_index = (index + 1) % self.config.skill_qualification_window
        self.state.skill_recent_count = min(
            self.state.skill_recent_count + 1,
            self.config.skill_qualification_window,
        )
        self.state.phase_a_priority_ready = self.state.phase_a_priority_ready or bool(
            np.all(self.state.skill_attempts >= self.config.min_eval_samples_per_gate)
        )
        if not self.config.enabled or self.state.phase_index != 0:
            return False
        return self._update_qualification(self._skill_evaluation_qualified())

    def _update_qualification(self, qualified: bool) -> bool:
        self.state.qualification_streak = (
            self.state.qualification_streak + 1 if qualified else 0
        )
        if self.state.qualification_streak < self.config.hysteresis_evaluations:
            return False
        self.state.phase_index += 1
        self.state.qualification_streak = 0
        self.state.evaluation_attempts.fill(0.0)
        self.state.evaluation_passes.fill(0.0)
        self.state.latest_evaluation_attempts.fill(0.0)
        self.state.latest_evaluation_passes.fill(0.0)
        self.state.evaluation_episodes = 0
        self.state.evaluation_finishes = 0
        self.state.evaluation_runs = 0
        return True

    def evaluation_pass_rates(self) -> np.ndarray:
        return np.divide(
            self.state.evaluation_passes,
            self.state.evaluation_attempts,
            out=np.zeros_like(self.state.evaluation_passes),
            where=self.state.evaluation_attempts > 0,
        )

    def training_pass_rates(self) -> np.ndarray:
        outcomes = self.state.training_passes + self.state.training_failures
        return np.divide(
            self.state.training_passes,
            outcomes,
            out=np.zeros_like(self.state.training_passes),
            where=outcomes > 0,
        )

    def skill_pass_rates(self) -> np.ndarray:
        return np.divide(
            self.state.skill_passes,
            self.state.skill_attempts,
            out=np.zeros_like(self.state.skill_passes),
            where=self.state.skill_attempts > 0,
        )

    def skill_strict_pass_rates(self) -> np.ndarray:
        return np.divide(
            self.state.skill_strict_passes,
            self.state.skill_attempts,
            out=np.zeros_like(self.state.skill_strict_passes),
            where=self.state.skill_attempts > 0,
        )

    def recent_skill_totals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = self.state.skill_recent_count
        return (
            np.sum(self.state.skill_recent_attempts[:count], axis=0),
            np.sum(self.state.skill_recent_passes[:count], axis=0),
            np.sum(self.state.skill_recent_strict_passes[:count], axis=0),
        )

    def recent_skill_pass_rates(self) -> np.ndarray:
        attempts, passes, _ = self.recent_skill_totals()
        return np.divide(
            passes,
            attempts,
            out=np.zeros_like(passes),
            where=attempts > 0,
        )

    def recent_skill_strict_pass_rates(self) -> np.ndarray:
        attempts, _, strict_passes = self.recent_skill_totals()
        return np.divide(
            strict_passes,
            attempts,
            out=np.zeros_like(strict_passes),
            where=attempts > 0,
        )

    def qualification_statistics(self) -> dict[str, Any]:
        if self.state.phase_index == 0:
            attempts, _, _ = self.recent_skill_totals()
            rates = self.recent_skill_pass_rates()
            source = "skill_audit"
        else:
            attempts = self.state.evaluation_attempts
            rates = self.evaluation_pass_rates()
            source = "strict_evaluation"
        return {
            "source": source,
            "attempts": attempts.copy(),
            "pass_rates": rates.copy(),
            "minimum_samples": float(np.min(attempts)) if attempts.size else 0.0,
            "minimum_pass_rate": float(np.min(rates)) if rates.size else 0.0,
            "qualification_streak": self.state.qualification_streak,
            "recent_audits": self.state.skill_recent_count,
            "phase_a_priority_ready": self.state.phase_a_priority_ready,
            "lifetime_attempts": self.state.skill_attempts.copy(),
            "lifetime_pass_rates": self.skill_pass_rates(),
            "strict_pass_rates": self.recent_skill_strict_pass_rates(),
            "lifetime_strict_pass_rates": self.skill_strict_pass_rates(),
            "latest_evaluation_pass_rates": self.latest_evaluation_pass_rates(),
        }

    def local_gate_probabilities(self) -> np.ndarray:
        return self.local_gate_probability_components()["combined"]

    def local_gate_probability_components(self) -> dict[str, np.ndarray]:
        if self.num_gates <= 1:
            empty = np.zeros((0,), dtype=np.float32)
            return {"direct": empty, "link": empty, "combined": empty}
        uniform = self._uniform_local_probabilities()
        if self.state.phase_index == 0:
            if not self._phase_a_priority_ready():
                return {"direct": uniform, "link": uniform, "combined": uniform}
            rates = self.recent_skill_pass_rates()[1:]
            priority = self._priority_probabilities(rates)
            mix = float(np.clip(self.config.phase_a_priority_mix, 0.0, 1.0))
            direct = ((1.0 - mix) * uniform + mix * priority).astype(np.float32)
            link_targets = self._priority_probabilities(
                self._phase_a_link_target_pass_rates()[1:]
            )
        elif self.state.phase_index == 1:
            return {"direct": uniform, "link": uniform, "combined": uniform}
        else:
            direct = self._priority_probabilities(self.training_pass_rates()[1:])
            link_targets = direct
        link = self._predecessor_start_probabilities(link_targets)
        link_mix = float(np.clip(self.config.local_link_start_mix, 0.0, 1.0))
        combined = ((1.0 - link_mix) * direct + link_mix * link).astype(np.float32)
        return {"direct": direct, "link": link, "combined": combined}

    def latest_evaluation_pass_rates(self) -> np.ndarray:
        return np.divide(
            self.state.latest_evaluation_passes,
            self.state.latest_evaluation_attempts,
            out=np.zeros_like(self.state.latest_evaluation_passes),
            where=self.state.latest_evaluation_attempts > 0,
        )

    def _phase_a_link_target_pass_rates(self) -> np.ndarray:
        skill_rates = self.recent_skill_pass_rates()
        evaluation_rates = self.latest_evaluation_pass_rates()
        return np.where(
            self.state.latest_evaluation_attempts > 0,
            evaluation_rates,
            skill_rates,
        )

    @staticmethod
    def _predecessor_start_probabilities(
        target_probabilities: np.ndarray,
    ) -> np.ndarray:
        target_probabilities = np.asarray(target_probabilities, dtype=np.float64)
        if target_probabilities.size == 0:
            return target_probabilities.astype(np.float32)
        starts = np.zeros_like(target_probabilities)
        starts[0] += target_probabilities[0]
        starts[:-1] += target_probabilities[1:]
        starts /= max(float(np.sum(starts)), 1.0e-8)
        return starts.astype(np.float32)

    def _priority_probabilities(self, rates: np.ndarray) -> np.ndarray:
        raw = self.config.priority_epsilon + np.power(
            np.maximum(1.0 - rates, 0.0),
            self.config.priority_alpha,
        )
        normalized = raw / max(float(np.sum(raw)), 1.0e-8)
        floor = min(
            float(self.config.priority_probability_floor),
            1.0 / len(normalized),
        )
        probabilities = floor + (1.0 - floor * len(normalized)) * normalized
        return probabilities.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase_index": self.state.phase_index,
            "gate_window_scale": self.state.gate_window_scale,
            "time_cost_scale": self.state.time_cost_scale,
            "qualification_streak": self.state.qualification_streak,
            "updates": self.state.updates,
            "training_passes": self.state.training_passes.copy(),
            "training_failures": self.state.training_failures.copy(),
            "evaluation_attempts": self.state.evaluation_attempts.copy(),
            "evaluation_passes": self.state.evaluation_passes.copy(),
            "latest_evaluation_attempts": self.state.latest_evaluation_attempts.copy(),
            "latest_evaluation_passes": self.state.latest_evaluation_passes.copy(),
            "evaluation_episodes": self.state.evaluation_episodes,
            "evaluation_finishes": self.state.evaluation_finishes,
            "evaluation_runs": self.state.evaluation_runs,
            "skill_attempts": self.state.skill_attempts.copy(),
            "skill_passes": self.state.skill_passes.copy(),
            "skill_strict_passes": self.state.skill_strict_passes.copy(),
            "skill_audits": self.state.skill_audits,
            "skill_recent_attempts": self.state.skill_recent_attempts.copy(),
            "skill_recent_passes": self.state.skill_recent_passes.copy(),
            "skill_recent_strict_passes": self.state.skill_recent_strict_passes.copy(),
            "skill_recent_index": self.state.skill_recent_index,
            "skill_recent_count": self.state.skill_recent_count,
            "phase_a_priority_ready": self.state.phase_a_priority_ready,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        training_passes = self._validate_gate_array(
            payload["training_passes"], "Training passes"
        )
        skill_passes = self._validate_gate_array(
            payload.get("skill_passes", np.zeros((self.num_gates,))),
            "Skill-audit passes",
        )
        self.state = CurriculumState(
            phase_index=int(payload["phase_index"]),
            gate_window_scale=float(payload["gate_window_scale"]),
            time_cost_scale=float(payload["time_cost_scale"]),
            qualification_streak=int(payload["qualification_streak"]),
            updates=int(payload["updates"]),
            training_passes=training_passes,
            training_failures=self._validate_gate_array(
                payload["training_failures"], "Training failures"
            ),
            evaluation_attempts=self._validate_gate_array(
                payload["evaluation_attempts"], "Evaluation attempts"
            ),
            evaluation_passes=self._validate_gate_array(
                payload["evaluation_passes"], "Evaluation passes"
            ),
            latest_evaluation_attempts=self._validate_gate_array(
                payload.get("latest_evaluation_attempts", np.zeros((self.num_gates,))),
                "Latest evaluation attempts",
            ),
            latest_evaluation_passes=self._validate_gate_array(
                payload.get("latest_evaluation_passes", np.zeros((self.num_gates,))),
                "Latest evaluation passes",
            ),
            evaluation_episodes=int(payload["evaluation_episodes"]),
            evaluation_finishes=int(payload["evaluation_finishes"]),
            evaluation_runs=int(payload.get("evaluation_runs", 0)),
            skill_attempts=self._validate_gate_array(
                payload.get("skill_attempts", np.zeros((self.num_gates,))),
                "Skill-audit attempts",
            ),
            skill_passes=skill_passes,
            skill_strict_passes=self._validate_gate_array(
                payload.get("skill_strict_passes", skill_passes),
                "Skill-audit strict passes",
            ),
            skill_audits=int(payload.get("skill_audits", 0)),
            skill_recent_attempts=np.zeros(
                (self.config.skill_qualification_window, self.num_gates),
                dtype=np.float64,
            ),
            skill_recent_passes=np.zeros(
                (self.config.skill_qualification_window, self.num_gates),
                dtype=np.float64,
            ),
            skill_recent_strict_passes=np.zeros(
                (self.config.skill_qualification_window, self.num_gates),
                dtype=np.float64,
            ),
            skill_recent_index=0,
            skill_recent_count=0,
            phase_a_priority_ready=bool(payload.get("phase_a_priority_ready", False)),
        )
        self._load_recent_history(payload)

    def _validate_gate_array(self, values: Any, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.num_gates,):
            raise ValueError(f"{label} do not match the course gate count")
        return array

    def _uniform_local_probabilities(self) -> np.ndarray:
        if self.num_gates <= 1:
            return np.zeros((0,), dtype=np.float32)
        return np.full((self.num_gates - 1,), 1.0 / (self.num_gates - 1), dtype=np.float32)

    def _phase_a_priority_ready(self) -> bool:
        return self.state.phase_a_priority_ready

    def _load_recent_history(self, payload: dict[str, Any]) -> None:
        attempts = np.asarray(
            payload.get("skill_recent_attempts", np.zeros((0, self.num_gates))),
            dtype=np.float64,
        )
        passes = np.asarray(
            payload.get("skill_recent_passes", np.zeros_like(attempts)),
            dtype=np.float64,
        )
        strict = np.asarray(
            payload.get("skill_recent_strict_passes", passes),
            dtype=np.float64,
        )
        if attempts.ndim != 2 or attempts.shape[1:] != (self.num_gates,):
            raise ValueError("Skill-audit recent attempts do not match the course")
        if passes.shape != attempts.shape or strict.shape != attempts.shape:
            raise ValueError("Skill-audit recent histories have inconsistent shapes")
        old_count = min(int(payload.get("skill_recent_count", attempts.shape[0])), attempts.shape[0])
        old_index = int(payload.get("skill_recent_index", old_count))
        if old_count == attempts.shape[0] and old_count > 0:
            order = np.concatenate(
                [np.arange(old_index, old_count), np.arange(0, old_index)]
            )
            attempts = attempts[order]
            passes = passes[order]
            strict = strict[order]
        else:
            attempts = attempts[:old_count]
            passes = passes[:old_count]
            strict = strict[:old_count]
        keep = min(old_count, self.config.skill_qualification_window)
        if keep:
            self.state.skill_recent_attempts[:keep] = attempts[-keep:]
            self.state.skill_recent_passes[:keep] = passes[-keep:]
            self.state.skill_recent_strict_passes[:keep] = strict[-keep:]
        self.state.skill_recent_count = keep
        self.state.skill_recent_index = keep % self.config.skill_qualification_window

    def _skill_evaluation_qualified(self) -> bool:
        attempts, _, _ = self.recent_skill_totals()
        if np.any(attempts < self.config.min_eval_samples_per_gate):
            return False
        return (
            float(np.min(self.recent_skill_pass_rates()))
            >= self.config.phase_a_min_pass_rate
        )

    def _official_evaluation_qualified(self) -> bool:
        attempts = self.state.evaluation_attempts
        if np.any(attempts < self.config.min_eval_samples_per_gate):
            return False
        minimum_pass_rate = float(np.min(self.evaluation_pass_rates()))
        completion_rate = self.state.evaluation_finishes / max(
            self.state.evaluation_episodes, 1
        )
        if self.state.phase_index == 1:
            return (
                minimum_pass_rate >= self.config.phase_b_min_pass_rate
                and completion_rate >= self.config.phase_b_completion_rate
            )
        return (
            minimum_pass_rate >= self.config.phase_c_min_pass_rate
            and completion_rate >= self.config.phase_c_completion_rate
        )


def _move_toward(value: float, target: float, maximum_delta: float) -> float:
    delta = float(np.clip(target - value, -maximum_delta, maximum_delta))
    return float(value + delta)
