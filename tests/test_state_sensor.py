import unittest
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
from a2rl_drone_training.config import ResidualConfig
from a2rl_drone_training.hierarchical.sensors import StateSensor


class StateSensorTests(unittest.TestCase):
    def sensor(self, **kwargs):
        sensor = StateSensor(replace(ResidualConfig(), **kwargs), 100, 500)
        sensor.reset(jnp.zeros((2, 17)), jnp.array([True, True]), seed=42)
        return sensor

    def test_clean_and_shared_sample(self):
        state = jnp.ones((2, 17))
        sensor = self.sensor()
        np.testing.assert_array_equal(sensor.read(state, 0), state)
        sensor = self.sensor(position_noise_std_m=0.03, velocity_noise_std_m_s=0.)
        first = sensor.read(state, 0)
        np.testing.assert_array_equal(first, sensor.read(state * 2, 0))
        np.testing.assert_array_equal(first, sensor.read(state * 2, 1))
        np.testing.assert_array_equal(first[:, 3:], state[:, 3:])
        self.assertFalse(np.array_equal(first[:, :3], state[:, :3]))

    def test_velocity_noise_does_not_change_position(self):
        sensor = self.sensor(position_noise_std_m=0., velocity_noise_std_m_s=.1)
        state = jnp.ones((2, 17))
        estimate = sensor.read(state, 0)
        np.testing.assert_array_equal(estimate[:, :3], state[:, :3])
        self.assertFalse(np.array_equal(estimate[:, 3:6], state[:, 3:6]))

    def test_delay_only_affects_position_velocity(self):
        sensor = self.sensor(sensor_delay_s=0.02)
        for tick in range(4):
            state = jnp.full((2, 17), float(tick + 1))
            estimate = sensor.read(state, tick * 5)
            np.testing.assert_array_equal(estimate[:, :6], max(0, tick - 1))
            np.testing.assert_array_equal(estimate[:, 6:], tick + 1)

    def test_bias_dropout_and_partial_reset(self):
        sensor = self.sensor(position_bias_m=(0.2, 0., 0.))
        np.testing.assert_allclose(sensor.read(jnp.zeros((2, 17)), 0)[:, 0], .2)
        sensor = self.sensor(sensor_dropout_prob=1.)
        value = sensor.read(jnp.ones((2, 17)), 0)
        np.testing.assert_array_equal(value[:, :6], 0)
        np.testing.assert_array_equal(value[:, 6:], 1)
        sensor.reset(jnp.full((2, 17), 7.), jnp.array([True, False]))
        value = sensor.read(jnp.ones((2, 17)), 5)
        np.testing.assert_array_equal(value[0, :6], 7)
        np.testing.assert_array_equal(value[1, :6], 0)

    def test_reset_between_control_ticks_does_not_freeze_estimator(self):
        sensor = self.sensor()
        sensor.read(jnp.ones((2, 17)), 0)
        sensor.reset(jnp.zeros((2, 17)), jnp.array([True, True]))
        sensor.read(jnp.ones((2, 17)), 3)
        np.testing.assert_array_equal(sensor.read(jnp.full((2, 17), 2.), 8), 2.)

    def test_training_cli_preserves_separate_noise_controls(self):
        from a2rl_drone_training.train import build_parser, _build_training_config
        args = build_parser().parse_args(['--estimator-position-noise-std', '0.01',
            '--estimator-velocity-noise-std', '0.05', '--sensor-delay', '0.02'])
        cfg = _build_training_config(args)
        self.assertEqual(cfg.residual.position_noise_std_m, .01)
        self.assertEqual(cfg.residual.velocity_noise_std_m_s, .05)
        self.assertEqual(cfg.residual.sensor_delay_s, .02)
        self.assertEqual(args.velocity_noise_std, .03)

    def test_same_seed_and_invalid_settings(self):
        a = self.sensor(position_noise_std_m=.01)
        b = self.sensor(position_noise_std_m=.01)
        np.testing.assert_array_equal(a.read(jnp.ones((2, 17)), 0), b.read(jnp.ones((2, 17)), 0))
        for config in [dict(sensor_delay_s=.001), dict(sensor_dropout_prob=1.1), dict(position_noise_std_m=-1), dict(velocity_noise_std_m_s=float('nan'))]:
            with self.assertRaises(ValueError):
                self.sensor(**config)


if __name__ == '__main__':
    unittest.main()
