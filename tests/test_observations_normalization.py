import unittest

import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import ObservationConfig
from a2rl_drone_training.normalization import (
    actor_normalization_spec,
    init_normalization_state,
    normalization_state_dict,
    normalization_state_from_dict,
    normalize_actor_observation,
    update_normalization_state,
)
from a2rl_drone_training.observations import build_racing_observation


def _observation(config: ObservationConfig, key=None):
    return build_racing_observation(
        pos_world=jnp.zeros((2, 3), dtype=jnp.float32),
        quat_xyzw=jnp.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=jnp.float32),
        vel_world=jnp.array([[1.0, 2.0, 3.0]] * 2, dtype=jnp.float32),
        ang_vel_world=jnp.zeros((2, 3), dtype=jnp.float32),
        acc_world=jnp.zeros((2, 3), dtype=jnp.float32),
        gate_centers_world=jnp.array([[2.0, 0.0, 0.0]], dtype=jnp.float32),
        gate_normals_world=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        gate_counter=jnp.zeros((2,), dtype=jnp.int32),
        total_gate_passes=1,
        last_action=jnp.zeros((2, 4), dtype=jnp.float32),
        remaining_time_fraction=jnp.array([1.0, 0.5], dtype=jnp.float32),
        yaw_error=jnp.array([0.4, -0.2], dtype=jnp.float32),
        stall_fraction=jnp.array([0.0, 0.5], dtype=jnp.float32),
        config=config,
        key=key,
    )


class ObservationTests(unittest.TestCase):
    def test_yaw_controller_and_deadline_state_are_observable(self):
        config = ObservationConfig(gate_context=1, sensor_noise_scale=0.0, pnp_dropout_prob=0.0)
        obs = _observation(config)
        np.testing.assert_allclose(np.asarray(obs[:, 21]), [1.0, 0.5])
        np.testing.assert_allclose(np.asarray(obs[:, 22]), [0.4, -0.2], atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(obs[:, 23]), [0.0, 0.5])

    def test_pnp_dropout_masks_all_pnp_derived_geometry(self):
        config = ObservationConfig(gate_context=1, sensor_noise_scale=0.0, pnp_dropout_prob=1.0)
        obs = _observation(config, key=jax.random.key(0))
        gate_features = np.asarray(obs[:, config.core_dim :])
        np.testing.assert_array_equal(gate_features, np.zeros_like(gate_features))

    def test_feature_specific_noise_does_not_touch_unconfigured_channels(self):
        config = ObservationConfig(
            gate_context=1,
            sensor_noise_scale=1.0,
            pnp_dropout_prob=0.0,
            gyro_noise_std_rad_s=0.0,
            specific_force_noise_std_m_s2=0.0,
            attitude_noise_std_rad=0.0,
            velocity_noise_std_m_s=0.5,
            gate_position_noise_std_m=0.0,
            gate_normal_noise_std_rad=0.0,
            gate_bearing_noise_std=0.0,
            gate_distance_noise_std_m=0.0,
        )
        clean = np.asarray(_observation(config))
        noisy = np.asarray(_observation(config, key=jax.random.key(1)))
        changed = np.any(np.abs(noisy - clean) > 1.0e-6, axis=0)
        expected = np.zeros((config.dim,), dtype=bool)
        expected[10:13] = True
        np.testing.assert_array_equal(changed, expected)


class NormalizationTests(unittest.TestCase):
    def test_normalization_freezes_when_only_applied(self):
        config = ObservationConfig(gate_context=1, sensor_noise_scale=0.0)
        spec = actor_normalization_spec(config)
        state = update_normalization_state(
            init_normalization_state(config), _observation(config), spec
        )
        before = normalization_state_dict(state)
        _ = normalize_actor_observation(
            state,
            _observation(config),
            spec,
            config.normalization_clip,
            config.normalization_epsilon,
        )
        after = normalization_state_dict(state)
        self.assertEqual(before["count"], after["count"])
        np.testing.assert_array_equal(before["mean"], after["mean"])
        np.testing.assert_array_equal(before["m2"], after["m2"])

    def test_normalization_state_survives_checkpoint_round_trip(self):
        config = ObservationConfig(gate_context=1, sensor_noise_scale=0.0)
        spec = actor_normalization_spec(config)
        state = update_normalization_state(
            init_normalization_state(config), _observation(config), spec
        )
        restored = normalization_state_from_dict(normalization_state_dict(state), config)
        self.assertEqual(float(restored.count), float(state.count))
        np.testing.assert_array_equal(np.asarray(restored.mean), np.asarray(state.mean))
        np.testing.assert_array_equal(np.asarray(restored.m2), np.asarray(state.m2))

    def test_bounded_quaternion_uses_fixed_scaling(self):
        config = ObservationConfig(gate_context=1, sensor_noise_scale=0.0)
        spec = actor_normalization_spec(config)
        state = update_normalization_state(
            init_normalization_state(config),
            jnp.full((4, config.dim), 100.0),
            spec,
        )
        obs = _observation(config)
        normalized = normalize_actor_observation(
            state,
            obs,
            spec,
            config.normalization_clip,
            config.normalization_epsilon,
        )
        np.testing.assert_allclose(np.asarray(normalized[:, 6:10]), np.asarray(obs[:, 6:10]))


if __name__ == "__main__":
    unittest.main()
