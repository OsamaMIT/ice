import unittest

import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.actions import ACTION_SPACE, validate_checkpoint_actions
from a2rl_drone_training.config import RacingEnvConfig
from a2rl_drone_training.env import CrazyflowRacingEnv


class MotorControlTests(unittest.TestCase):
    def test_legacy_or_different_action_semantics_are_rejected(self):
        for payload in ({}, {"action_space": "attitude"}):
            with self.assertRaisesRegex(ValueError, "action space"):
                validate_checkpoint_actions(payload)
        validate_checkpoint_actions({"action_space": ACTION_SPACE})

    def test_attitude_physics_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first_principles"):
            CrazyflowRacingEnv(RacingEnvConfig(physics="so_rpy_rotor"))

    def test_hover_thrust_matches_mass_and_partial_reset_preserves_other_motors(self):
        env = CrazyflowRacingEnv(RacingEnvConfig(num_envs=2, auto_reset=False))
        try:
            a, b, c = np.asarray(env.sim.data.params.rpm2thrust)
            rpm = float(env.motor_rpm_hover)
            total_force = 4 * (a + b * rpm + c * rpm**2)
            mass = float(jnp.mean(env.sim.data.params.mass))
            self.assertAlmostEqual(total_force, mass * 9.81, places=5)
            states = env.sim.data.states.replace(rotor_vel=jnp.full((2, 1, 4), 10000.0))
            env.sim.data = env.sim.data.replace(states=states)
            env.reset(mask=jnp.array([True, False]))
            np.testing.assert_allclose(env.sim.data.states.rotor_vel[0], rpm)
            np.testing.assert_allclose(env.sim.data.states.rotor_vel[1], 10000.0)
        finally:
            env.close()

    def test_oblique_gate_resets_stay_on_approach_side(self):
        env = CrazyflowRacingEnv(RacingEnvConfig(num_envs=96, auto_reset=False,
                                               reset_distribution="evaluation"))
        try:
            env.set_skill_audit_window_scale(1.8)
            gates = jnp.repeat(jnp.arange(12), 8)
            for seed in (10123, 10124, 7):
                env.reset(seed=seed, forced_reset_gates=gates)
                plane, _, _ = env._gate_offsets(env.sim.data.states.pos[:, 0], gates)
                self.assertTrue(np.all(np.asarray(plane) < 0),
                                f"Spawn beyond the gate plane with seed {seed}")
        finally:
            env.close()
