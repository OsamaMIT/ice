import unittest

import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import RacingEnvConfig
from a2rl_drone_training.rewards import (
    course_coordinate,
    gate_clearance,
    progress_potential,
    rebase_progress_potential,
    reward_v1,
    reward_v2,
)


class RewardV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = RacingEnvConfig(
            reward_version="v2",
            potential_gamma=1.0,
            time_penalty=0.2,
            action_delta_penalty=0.02,
        )

    def _reward(self, **overrides):
        zero = jnp.zeros((1,), dtype=jnp.float32)
        false = jnp.zeros((1,), dtype=jnp.bool_)
        kwargs = {
            "track_phi_before": zero,
            "track_phi_after": zero,
            "center_phi_before": zero,
            "center_phi_after": zero,
            "action": jnp.zeros((1, 4), dtype=jnp.float32),
            "previous_action": jnp.zeros((1, 4), dtype=jnp.float32),
            "clearance": jnp.ones((1,), dtype=jnp.float32),
            "passed_gate": false,
            "finished": false,
            "missed_gate": false,
            "crashed": false,
            "dt": 0.01,
            "time_cost_scale": 0.0,
            "position_noise_std_m": 0.03,
            "config": self.config,
        }
        kwargs.update(overrides)
        return reward_v2(**kwargs)

    def test_course_progress_is_positive_and_continuous_at_gate_handoff(self):
        starts = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        directions = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        lengths = jnp.ones((2,), dtype=jnp.float32)
        before = course_coordinate(
            jnp.array([[0.999, 0.0, 0.0]], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
            starts,
            directions,
            lengths,
        )
        after = course_coordinate(
            jnp.array([[1.001, 0.0, 0.0]], dtype=jnp.float32),
            jnp.array([1], dtype=jnp.int32),
            starts,
            directions,
            lengths,
        )
        self.assertGreater(float(after[0]), float(before[0]))
        self.assertLess(abs(float(after[0] - before[0])), 0.01)

    def test_centering_potential_is_localized_near_gate_plane(self):
        centers = jnp.array([[1.0, 0.0, 1.0]], dtype=jnp.float32)
        normals = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
        right = jnp.array([[0.0, 1.0, 0.0]], dtype=jnp.float32)
        starts = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
        lengths = jnp.ones((1,), dtype=jnp.float32)
        kwargs = {
            "gate_counter": jnp.array([0], dtype=jnp.int32),
            "gate_centers": centers,
            "gate_normals": normals,
            "gate_right_axes": right,
            "segment_starts": starts,
            "segment_directions": normals,
            "segment_lengths": lengths,
            "track_scale": 1.0,
            "center_scale": 0.25,
            "plane_sigma_m": 0.75,
        }
        _, _, center_near = progress_potential(
            pos=jnp.array([[1.0, 0.4, 1.0]], dtype=jnp.float32), **kwargs
        )
        _, _, center_far = progress_potential(
            pos=jnp.array([[-4.0, 0.4, 1.0]], dtype=jnp.float32), **kwargs
        )
        self.assertGreater(abs(float(center_near[0])), 100.0 * abs(float(center_far[0])))

    def test_time_cost_uses_physical_dt(self):
        _, short = self._reward(dt=0.01, time_cost_scale=1.0)
        _, long = self._reward(dt=0.02, time_cost_scale=1.0)
        self.assertAlmostEqual(float(short["reward_time"][0]), -0.002, places=7)
        self.assertAlmostEqual(float(long["reward_time"][0]), -0.004, places=7)

    def test_potential_progress_has_correct_sign(self):
        _, forward = self._reward(
            track_phi_before=jnp.array([0.2]),
            track_phi_after=jnp.array([0.3]),
        )
        _, backward = self._reward(
            track_phi_before=jnp.array([0.3]),
            track_phi_after=jnp.array([0.2]),
        )
        self.assertGreater(float(forward["reward_potential_progress"][0]), 0.0)
        self.assertLess(float(backward["reward_potential_progress"][0]), 0.0)

    def test_default_progress_weight_strengthens_but_keeps_per_gate_scale_small(self):
        config = RacingEnvConfig(potential_gamma=1.0)
        _, components = self._reward(
            track_phi_before=jnp.array([0.0]),
            track_phi_after=jnp.array([1.0]),
            config=config,
        )
        self.assertAlmostEqual(
            float(components["reward_potential_progress"][0]),
            1.25,
            places=6,
        )

    def test_course_coordinate_penalizes_bounded_backtracking(self):
        starts = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        directions = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
        lengths = jnp.ones((1,), dtype=jnp.float32)
        position = jnp.array([[-0.25, 0.0, 0.0]], dtype=jnp.float32)
        legacy = course_coordinate(
            position,
            jnp.array([0], dtype=jnp.int32),
            starts,
            directions,
            lengths,
        )
        corrected = course_coordinate(
            position,
            jnp.array([0], dtype=jnp.int32),
            starts,
            directions,
            lengths,
            backtrack_limit=0.5,
        )
        self.assertEqual(float(legacy[0]), 0.0)
        self.assertAlmostEqual(float(corrected[0]), -0.25, places=6)

    def test_rebased_potential_removes_local_gate_index_offset(self):
        absolute = jnp.array([0.4, 7.4], dtype=jnp.float32)
        track = jnp.array([0.4, 7.4], dtype=jnp.float32)
        rebased, rebased_track = rebase_progress_potential(
            absolute,
            track,
            jnp.array([0, 7], dtype=jnp.int32),
            track_scale=1.0,
        )
        np.testing.assert_allclose(np.asarray(rebased), [0.4, 0.4], atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(rebased_track), [0.4, 0.4], atol=1.0e-6)

        gamma = RacingEnvConfig().potential_gamma
        stationary_shaping = gamma * np.asarray(rebased_track) - np.asarray(rebased_track)
        self.assertAlmostEqual(stationary_shaping[0], stationary_shaping[1], places=7)

    def test_action_smoothing_penalizes_changes_not_magnitude(self):
        constant = jnp.full((1, 4), 0.8, dtype=jnp.float32)
        _, unchanged = self._reward(action=constant, previous_action=constant)
        _, changed = self._reward(
            action=constant,
            previous_action=jnp.zeros((1, 4), dtype=jnp.float32),
        )
        self.assertAlmostEqual(float(unchanged["reward_smoothness"][0]), 0.0)
        self.assertLess(float(changed["reward_smoothness"][0]), 0.0)

    def test_centering_improvement_is_undiscounted_and_has_no_switch_bonus(self):
        _, improved = self._reward(
            center_phi_before=jnp.array([-0.10]),
            center_phi_after=jnp.array([-0.04]),
        )
        _, worsened = self._reward(
            center_phi_before=jnp.array([-0.04]),
            center_phi_after=jnp.array([-0.10]),
        )
        _, unchanged = self._reward(
            center_phi_before=jnp.array([-0.10]),
            center_phi_after=jnp.array([-0.10]),
        )
        self.assertGreater(float(improved["reward_potential_center"][0]), 0.0)
        self.assertLess(float(worsened["reward_potential_center"][0]), 0.0)
        self.assertEqual(float(unchanged["reward_potential_center"][0]), 0.0)

    def test_default_smoothness_budget_is_below_five_percent(self):
        config = RacingEnvConfig()
        observed_sum_squared_delta = 1.5
        ten_second_cost = (
            config.action_delta_penalty * observed_sum_squared_delta * 10.0 / config.dt
        )
        nominal_success_return = 12 * config.gate_pass_bonus + config.lap_finish_bonus
        self.assertLess(ten_second_cost, 0.05 * nominal_success_return)

    def test_clearance_threshold_replaces_exact_center_bonus(self):
        passed = jnp.ones((1,), dtype=jnp.bool_)
        _, center = self._reward(clearance=jnp.array([0.60]), passed_gate=passed)
        _, safely_off_center = self._reward(clearance=jnp.array([0.30]), passed_gate=passed)
        _, unsafe = self._reward(clearance=jnp.array([0.05]), passed_gate=passed)
        self.assertEqual(float(center["reward_margin"][0]), 0.0)
        self.assertEqual(float(safely_off_center["reward_margin"][0]), 0.0)
        self.assertLess(float(unsafe["reward_margin"][0]), 0.0)
        self.assertEqual(float(center["reward_gate"][0]), float(safely_off_center["reward_gate"][0]))

    def test_physical_edge_margin_cancels_relaxed_gate_bonus(self):
        passed = jnp.ones((1,), dtype=jnp.bool_)
        _, edge = self._reward(clearance=jnp.array([0.0]), passed_gate=passed)
        _, centered = self._reward(clearance=jnp.array([0.75]), passed_gate=passed)
        self.assertAlmostEqual(
            float(edge["reward_gate"][0] + edge["reward_margin"][0]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(float(centered["reward_gate"][0]), 6.0, places=6)
        self.assertAlmostEqual(float(centered["reward_margin"][0]), 0.0, places=6)

    def test_finish_bonus_requires_true_course_finish_flag(self):
        _, local_segment = self._reward(finished=jnp.array([False]))
        _, full_course = self._reward(finished=jnp.array([True]))
        self.assertEqual(float(local_segment["reward_finish"][0]), 0.0)
        self.assertEqual(
            float(full_course["reward_finish"][0]),
            self.config.lap_finish_bonus,
        )

    def test_rectangular_gate_clearance_uses_nearest_inner_edge(self):
        clearance = gate_clearance(
            jnp.array([0.4]),
            jnp.array([0.1]),
            jnp.array([1.0]),
            jnp.array([2.0]),
        )
        self.assertAlmostEqual(float(clearance[0]), 0.1, places=6)


class RewardV1RegressionTests(unittest.TestCase):
    def test_v1_matches_legacy_formula(self):
        cfg = RacingEnvConfig(reward_version="v1")
        action = jnp.array([[0.1, -0.2, 0.3, -0.4]], dtype=jnp.float32)
        reward, _ = reward_v1(
            plane_before=jnp.array([-1.0]),
            plane_after=jnp.array([-0.9]),
            radial_before=jnp.array([0.5]),
            radial_after=jnp.array([0.4]),
            distance_before=jnp.array([2.0]),
            distance_after=jnp.array([1.9]),
            lookahead_distance_before=jnp.array([3.0]),
            lookahead_distance_after=jnp.array([2.95]),
            cross_radial=jnp.array([0.2]),
            velocity_alignment=jnp.array([0.7]),
            stalled=jnp.array([True]),
            action=action,
            gate_id=jnp.array([2], dtype=jnp.int32),
            passed_gate=jnp.array([True]),
            missed_gate=jnp.array([False]),
            crashed=jnp.array([False]),
            finished=jnp.array([False]),
            course_radius_m=0.55,
            num_gates=12,
            config=cfg,
        )
        center = np.exp(-2.5 * 0.4 / 0.55)
        cross_center = np.exp(-2.5 * 0.2 / 0.55)
        expected = (
            cfg.plane_progress_reward * 0.1
            + cfg.distance_progress_reward * 0.1
            + cfg.lookahead_progress_reward * 0.05
            + cfg.radial_progress_reward * 0.1
            + cfg.centerline_reward * center
            + cfg.crossing_center_reward * cross_center
            + cfg.path_alignment_reward * 0.7
            + cfg.stall_penalty
            - cfg.action_penalty * float(jnp.sum(jnp.square(action)))
            + cfg.gate_pass_bonus
            + cfg.centered_crossing_bonus * cross_center
        )
        self.assertAlmostEqual(float(reward[0]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
