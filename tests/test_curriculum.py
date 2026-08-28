import unittest

import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import CurriculumConfig
from a2rl_drone_training.curriculum import CurriculumController, sample_reset_gates


class ResetSamplerTests(unittest.TestCase):
    def test_phase_a_training_mixture_is_exact_and_stratified(self):
        gates, local = sample_reset_gates(
            key=jax.random.key(3),
            reset_mask=jnp.ones((100,), dtype=jnp.bool_),
            current_reset_gate=jnp.zeros((100,), dtype=jnp.int32),
            gate1_fraction=0.2,
            local_gate_probabilities=jnp.ones((4,), dtype=jnp.float32) / 4.0,
            prioritized=False,
            evaluation=False,
        )
        gates = np.asarray(gates)
        self.assertEqual(int(np.sum(gates == 0)), 20)
        self.assertEqual(int(jnp.sum(local)), 80)
        self.assertEqual([int(np.sum(gates == i)) for i in range(1, 5)], [20, 20, 20, 20])

    def test_evaluation_sampler_always_targets_gate_one(self):
        current = jnp.arange(16, dtype=jnp.int32) % 5
        gates, local = sample_reset_gates(
            key=jax.random.key(7),
            reset_mask=jnp.ones((16,), dtype=jnp.bool_),
            current_reset_gate=current,
            gate1_fraction=0.2,
            local_gate_probabilities=jnp.ones((4,), dtype=jnp.float32) / 4.0,
            prioritized=True,
            evaluation=True,
        )
        np.testing.assert_array_equal(np.asarray(gates), np.zeros((16,), dtype=np.int32))
        self.assertFalse(bool(jnp.any(local)))

    def test_single_environment_resets_are_unbiased_and_deterministic(self):
        keys = jax.random.split(jax.random.key(19), 4000)
        reset_mask = jnp.ones((1,), dtype=jnp.bool_)
        current = jnp.zeros((1,), dtype=jnp.int32)
        probabilities = jnp.ones((4,), dtype=jnp.float32) / 4.0

        def sample_one(key):
            _, local = sample_reset_gates(
                key=key,
                reset_mask=reset_mask,
                current_reset_gate=current,
                gate1_fraction=0.2,
                local_gate_probabilities=probabilities,
                prioritized=False,
                evaluation=False,
            )
            return local[0]

        first = np.asarray(jax.vmap(sample_one)(keys))
        second = np.asarray(jax.vmap(sample_one)(keys))
        np.testing.assert_array_equal(first, second)
        self.assertLess(abs(float(np.mean(first)) - 0.8), 0.03)

    def test_partial_reset_branch_does_not_depend_on_current_occupancy(self):
        reset_mask = jnp.array([False, True, False, True, True], dtype=jnp.bool_)
        probabilities = jnp.ones((3,), dtype=jnp.float32) / 3.0
        kwargs = {
            "key": jax.random.key(41),
            "reset_mask": reset_mask,
            "gate1_fraction": 0.5,
            "local_gate_probabilities": probabilities,
            "prioritized": False,
            "evaluation": False,
        }
        first_gates, first_local = sample_reset_gates(
            current_reset_gate=jnp.array([0, 0, 0, 0, 0]),
            **kwargs,
        )
        second_gates, second_local = sample_reset_gates(
            current_reset_gate=jnp.array([3, 2, 1, 3, 1]),
            **kwargs,
        )
        np.testing.assert_array_equal(np.asarray(first_local), np.asarray(second_local))
        np.testing.assert_array_equal(
            np.asarray(first_gates)[np.asarray(reset_mask)],
            np.asarray(second_gates)[np.asarray(reset_mask)],
        )


class CurriculumControllerTests(unittest.TestCase):
    def test_advancement_uses_evaluation_thresholds_and_hysteresis(self):
        config = CurriculumConfig(
            min_eval_samples_per_gate=2,
            hysteresis_evaluations=2,
            phase_a_min_pass_rate=0.8,
            phase_b_min_pass_rate=0.85,
            phase_c_min_pass_rate=0.9,
            phase_b_completion_rate=0.5,
            phase_c_completion_rate=0.8,
        )
        controller = CurriculumController(config, num_gates=3)
        attempts = np.full((3,), 2.0)
        passes = np.full((3,), 2.0)
        self.assertFalse(
            controller.record_evaluation(attempts=attempts, passes=passes, episodes=2, finishes=2)
        )
        self.assertEqual(controller.phase, "A")
        self.assertFalse(
            controller.record_skill_evaluation(attempts=attempts, passes=passes)
        )
        self.assertTrue(
            controller.record_skill_evaluation(attempts=attempts, passes=passes)
        )
        self.assertEqual(controller.phase, "B")

        for expected_advance in (False, True):
            advanced = controller.record_evaluation(
                attempts=attempts,
                passes=passes,
                episodes=2,
                finishes=1,
            )
            self.assertEqual(advanced, expected_advance)
        self.assertEqual(controller.phase, "C")

        for expected_advance in (False, True):
            advanced = controller.record_evaluation(
                attempts=attempts,
                passes=passes,
                episodes=5,
                finishes=4,
            )
            self.assertEqual(advanced, expected_advance)
        self.assertEqual(controller.phase, "D")

    def test_gate_window_changes_are_rate_limited(self):
        controller = CurriculumController(
            CurriculumConfig(gate_window_rate_limit=0.02), num_gates=3
        )
        controller.state.phase_index = 1
        controller.step()
        self.assertAlmostEqual(controller.state.gate_window_scale, 1.78)

    def test_prioritized_local_starts_keep_probability_floor(self):
        controller = CurriculumController(
            CurriculumConfig(priority_probability_floor=0.05), num_gates=4
        )
        controller.state.phase_index = 2
        controller.record_training(
            passes=np.array([0.0, 100.0, 50.0, 10.0]),
            failures=np.array([0.0, 0.0, 50.0, 90.0]),
        )
        components = controller.local_gate_probability_components()
        probabilities = components["combined"]
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=6)
        self.assertTrue(np.all(probabilities >= 0.05))
        self.assertGreater(components["direct"][2], components["direct"][1])
        self.assertGreater(components["direct"][1], components["direct"][0])
        self.assertGreater(probabilities[1], components["direct"][1])

    def test_phase_a_priority_waits_for_coverage_then_mixes_recent_failures(self):
        config = CurriculumConfig(
            min_eval_samples_per_gate=32,
            phase_a_priority_mix=0.5,
        )
        controller = CurriculumController(config, num_gates=4)
        uniform = np.full((3,), 1.0 / 3.0, dtype=np.float32)
        np.testing.assert_allclose(controller.local_gate_probabilities(), uniform)

        controller.record_skill_evaluation(
            attempts=np.full((4,), 32.0),
            passes=np.array([32.0, 32.0, 16.0, 0.0]),
            strict_passes=np.array([30.0, 28.0, 12.0, 0.0]),
        )
        priority = controller._priority_probabilities(np.array([1.0, 0.5, 0.0]))
        direct = 0.5 * uniform + 0.5 * priority
        link = controller._predecessor_start_probabilities(priority)
        expected = 0.5 * direct + 0.5 * link
        np.testing.assert_allclose(
            controller.local_gate_probabilities(),
            expected,
            atol=1.0e-7,
        )
        components = controller.local_gate_probability_components()
        np.testing.assert_allclose(components["direct"], direct, atol=1.0e-7)
        np.testing.assert_allclose(components["link"], link, atol=1.0e-7)
        self.assertTrue(controller.parameters().prioritized_local_starts)

    def test_official_weak_gate_redirects_link_replay_to_its_predecessor(self):
        controller = CurriculumController(
            CurriculumConfig(
                min_eval_samples_per_gate=4,
                phase_a_priority_mix=0.0,
                local_link_start_mix=1.0,
            ),
            num_gates=4,
        )
        attempts = np.full((4,), 4.0)
        controller.record_skill_evaluation(attempts=attempts, passes=attempts)
        controller.record_evaluation(
            attempts=attempts,
            passes=np.array([4.0, 4.0, 0.0, 4.0]),
            episodes=4,
            finishes=0,
        )
        components = controller.local_gate_probability_components()
        # Weak G3 should primarily create starts at G2, its predecessor.
        self.assertGreater(components["link"][0], components["link"][1])
        self.assertGreater(components["link"][0], components["link"][2])
        np.testing.assert_allclose(
            controller.local_gate_probabilities(),
            components["link"],
        )

    def test_rolling_phase_a_qualification_recovers_from_old_failures(self):
        controller = CurriculumController(
            CurriculumConfig(
                skill_qualification_window=2,
                min_eval_samples_per_gate=8,
                hysteresis_evaluations=2,
                phase_a_min_pass_rate=0.8,
            ),
            num_gates=3,
        )
        attempts = np.full((3,), 8.0)
        failed = np.array([8.0, 0.0, 8.0])
        passed = np.full((3,), 8.0)
        self.assertFalse(controller.record_skill_evaluation(attempts=attempts, passes=failed))
        self.assertFalse(controller.record_skill_evaluation(attempts=attempts, passes=failed))
        self.assertFalse(controller.record_skill_evaluation(attempts=attempts, passes=passed))
        self.assertFalse(controller.record_skill_evaluation(attempts=attempts, passes=passed))
        self.assertTrue(controller.record_skill_evaluation(attempts=attempts, passes=passed))
        self.assertEqual(controller.phase, "B")
        self.assertLess(controller.skill_pass_rates()[1], 0.8)
        self.assertEqual(controller.recent_skill_pass_rates()[1], 1.0)

    def test_curriculum_state_round_trips(self):
        first = CurriculumController(CurriculumConfig(), num_gates=4)
        first.state.phase_index = 2
        first.state.gate_window_scale = 1.17
        first.state.time_cost_scale = 0.41
        first.record_training(np.arange(4), np.arange(4)[::-1])
        first.record_skill_evaluation(
            attempts=np.full((4,), 8.0),
            passes=np.arange(4, dtype=np.float64),
            strict_passes=np.arange(4, dtype=np.float64) / 2.0,
        )
        first.record_evaluation(
            attempts=np.full((4,), 6.0),
            passes=np.array([6.0, 5.0, 1.0, 4.0]),
            episodes=6,
            finishes=1,
        )
        payload = first.state_dict()

        second = CurriculumController(CurriculumConfig(), num_gates=4)
        second.load_state_dict(payload)
        self.assertEqual(second.phase, "C")
        self.assertEqual(second.state.gate_window_scale, 1.17)
        self.assertEqual(second.state.time_cost_scale, 0.41)
        np.testing.assert_array_equal(second.state.training_passes, first.state.training_passes)
        np.testing.assert_array_equal(second.state.skill_attempts, first.state.skill_attempts)
        np.testing.assert_array_equal(second.state.skill_passes, first.state.skill_passes)
        np.testing.assert_array_equal(
            second.state.skill_strict_passes,
            first.state.skill_strict_passes,
        )
        np.testing.assert_array_equal(
            second.state.skill_recent_attempts,
            first.state.skill_recent_attempts,
        )
        np.testing.assert_array_equal(
            second.state.latest_evaluation_attempts,
            first.state.latest_evaluation_attempts,
        )
        np.testing.assert_array_equal(
            second.state.latest_evaluation_passes,
            first.state.latest_evaluation_passes,
        )

    def test_old_curriculum_state_initializes_empty_rolling_history(self):
        first = CurriculumController(CurriculumConfig(), num_gates=3)
        payload = first.state_dict()
        for key in tuple(payload):
            if key.startswith("skill_recent_") or key == "skill_strict_passes":
                payload.pop(key)
        payload["skill_attempts"] = np.full((3,), 100.0)
        payload["skill_passes"] = np.full((3,), 100.0)

        restored = CurriculumController(CurriculumConfig(), num_gates=3)
        restored.load_state_dict(payload)
        self.assertEqual(restored.state.skill_recent_count, 0)
        self.assertEqual(float(np.sum(restored.recent_skill_totals()[0])), 0.0)
        np.testing.assert_array_equal(
            restored.state.skill_strict_passes,
            restored.state.skill_passes,
        )
        np.testing.assert_array_equal(
            restored.state.latest_evaluation_attempts,
            np.zeros((3,)),
        )


if __name__ == "__main__":
    unittest.main()
