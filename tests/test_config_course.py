import unittest
from pathlib import Path

import numpy as np

from a2rl_drone_training.config import ObservationConfig, RacingEnvConfig, TrainingConfig
from a2rl_drone_training.course import arena_38m_stacked_course, course_by_name, default_a2rl_course


class ConfigCourseTests(unittest.TestCase):
    def test_observation_dimension_tracks_gate_context(self):
        cfg = ObservationConfig(gate_context=4)
        self.assertEqual(cfg.core_dim, 21)
        self.assertEqual(cfg.dim, cfg.core_dim + 4 * cfg.gate_feature_dim)

    def test_default_course_has_unit_normals(self):
        course = default_a2rl_course()
        self.assertEqual(course.centers.shape[1], 3)
        self.assertEqual(course.normals.shape, course.centers.shape)
        norms = (course.normals**2).sum(axis=-1) ** 0.5
        self.assertTrue(((norms > 0.999) & (norms < 1.001)).all())

    def test_env_dt_matches_control_rate(self):
        cfg = RacingEnvConfig(control_hz=200)
        self.assertEqual(cfg.dt, 0.005)

    def test_default_episode_horizon_is_24_seconds(self):
        cfg = RacingEnvConfig()
        self.assertEqual(cfg.max_episode_time_s, 24.0)
        self.assertEqual(cfg.max_episode_steps, 2400)

    def test_late_gate_reward_is_opt_in(self):
        cfg = RacingEnvConfig()
        self.assertEqual(cfg.late_gate_reward_base, 1.0)
        self.assertEqual(cfg.late_gate_reward_cap, 4.0)
        self.assertEqual(cfg.time_penalty, 0.0)
        self.assertEqual(cfg.early_gate_bonus, 0.0)

    def test_default_mixed_start_is_60_40(self):
        cfg = RacingEnvConfig()
        self.assertAlmostEqual(cfg.mixed_start_random_fraction, 0.4)
        self.assertEqual(cfg.mixed_start_random_min_gate, 2)
        self.assertIsNone(cfg.mixed_start_random_max_gate)
        self.assertIsNone(cfg.mixed_start_focus_gate)
        self.assertAlmostEqual(cfg.mixed_start_focus_fraction, 0.0)
        self.assertAlmostEqual(cfg.racing_line_spawn_min_distance_m, 1.0)
        self.assertAlmostEqual(cfg.racing_line_spawn_max_distance_m, 3.0)
        self.assertAlmostEqual(cfg.racing_line_spawn_speed_m_s, 2.0)

    def test_arena_38m_stacked_course_metadata(self):
        course = arena_38m_stacked_course()
        self.assertEqual(course.name, "arena_38m_stacked")
        self.assertEqual(course.num_gates, 12)
        self.assertEqual(course.logical_gate_ids.tolist().count(7), 2)
        self.assertEqual(course.logical_gate_ids.tolist().count(10), 2)
        self.assertEqual(course.openings[6], "top")
        self.assertEqual(course.openings[7], "bottom")
        self.assertEqual(course.openings[10], "bottom")
        self.assertEqual(course.openings[11], "top")
        self.assertEqual(course.bounds_min[:2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(course.bounds_min[2]), 0.05)
        self.assertEqual(course.bounds_max.tolist(), [38.0, 38.0, 5.0])
        self.assertEqual(course.arena_size, (38.0, 38.0))
        self.assertAlmostEqual(float(course.widths[0]), 1.5)
        self.assertAlmostEqual(float(course.heights[0]), 1.5)
        self.assertAlmostEqual(float(course.outer_widths[0]), 2.7)
        self.assertAlmostEqual(float(course.outer_heights[0]), 2.7)
        self.assertEqual(course.nominal_racing_line.shape, (14, 3))

    def test_course_by_name(self):
        self.assertEqual(course_by_name("arena_38m_stacked").num_gates, 12)
        self.assertEqual(course_by_name("compact_slalom").name, "compact_slalom")

    def test_gate_diagnostic_line_marks_weak_gate(self):
        try:
            from a2rl_drone_training.trainer import _gate_diagnostic_line
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("trainer diagnostics require jax")
            raise

        line = _gate_diagnostic_line(
            {
                "gate_target_counts": np.array([100, 20, 15], dtype=np.float32),
                "gate_pass_counts": np.array([4, 1, 0], dtype=np.float32),
                "gate_miss_counts": np.array([0, 3, 1], dtype=np.float32),
                "gate_stop_counts": np.array([0, 3, 1], dtype=np.float32),
                "gate_stall_counts": np.array([0, 5, 0], dtype=np.float32),
            }
        )
        self.assertIn("weak=G2:miss", line)
        self.assertIn("target=1:100,2:20,3:15", line)
        self.assertIn("pass=1:4,2:1,3:0", line)
        self.assertIn("miss=1:0,2:3,3:1", line)

    def test_observation_builder_is_time_independent(self):
        try:
            import jax.numpy as jnp

            from a2rl_drone_training.observations import build_racing_observation
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("observation builder requires jax")
            raise

        cfg = ObservationConfig(additive_noise_std=0.0, pnp_dropout_prob=0.0)
        course = arena_38m_stacked_course()
        obs = build_racing_observation(
            pos_world=jnp.zeros((2, 3), dtype=jnp.float32),
            quat_xyzw=jnp.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=jnp.float32,
            ),
            vel_world=jnp.zeros((2, 3), dtype=jnp.float32),
            ang_vel_world=jnp.zeros((2, 3), dtype=jnp.float32),
            acc_world=jnp.zeros((2, 3), dtype=jnp.float32),
            gate_centers_world=jnp.asarray(course.centers, dtype=jnp.float32),
            gate_normals_world=jnp.asarray(course.normals, dtype=jnp.float32),
            gate_counter=jnp.array([0, 5], dtype=jnp.int32),
            total_gate_passes=course.num_gates,
            last_action=jnp.zeros((2, 4), dtype=jnp.float32),
            config=cfg,
        )
        self.assertEqual(obs.shape, (2, cfg.dim))

    def test_checkpoint_observation_dimension_guard(self):
        try:
            from a2rl_drone_training.trainer import (
                _validate_checkpoint_observation_compatibility,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("checkpoint compatibility guard requires jax")
            raise

        payload = {"config": TrainingConfig(obs=ObservationConfig(core_dim=22))}
        with self.assertRaisesRegex(ValueError, "checkpoint_obs_dim=52"):
            _validate_checkpoint_observation_compatibility(
                Path("old_checkpoint.pkl"),
                payload,
                TrainingConfig(),
            )

    def test_balanced_random_start_mask_targets_active_ratio(self):
        try:
            import jax
            import jax.numpy as jnp

            from a2rl_drone_training.env import _balanced_random_start_mask
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("balanced reset sampler requires jax")
            raise

        key = jax.random.key(0)
        full_reset = jnp.ones((10,), dtype=jnp.bool_)
        no_random = jnp.zeros((10,), dtype=jnp.bool_)
        selected = _balanced_random_start_mask(key, full_reset, no_random, 0.4)
        self.assertEqual(int(jnp.sum(selected)), 4)

        partial_reset = jnp.array(
            [False, False, False, False, False, False, True, True, True, True],
            dtype=jnp.bool_,
        )
        current = jnp.array(
            [True, True, False, False, False, False, False, False, False, False],
            dtype=jnp.bool_,
        )
        selected = _balanced_random_start_mask(key, partial_reset, current, 0.4)
        self.assertEqual(int(jnp.sum(selected)), 2)


if __name__ == "__main__":
    unittest.main()
