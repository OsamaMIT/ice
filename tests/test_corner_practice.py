import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import CurriculumConfig, RacingEnvConfig
from a2rl_drone_training.curriculum import CurriculumController
from a2rl_drone_training.env import CrazyflowRacingEnv
from a2rl_drone_training.train import parse_args, _build_training_config


class CornerPracticeTests(unittest.TestCase):
    def test_profile_resume_and_reset_mixture(self):
        args = parse_args(["--profile", "rtx-5050-corner"])
        cfg = _build_training_config(args)
        self.assertFalse(cfg.curriculum.strict_course_training)
        self.assertTrue(cfg.curriculum.corner_practice)
        old = CurriculumController(CurriculumConfig(), 12)
        old.record_skill_evaluation(attempts=np.full(12, 8.), passes=np.full(12, 8.))
        controller = CurriculumController(cfg.curriculum, 12)
        controller.load_state_dict(old.state_dict())
        self.assertEqual(controller.state.skill_recent_count, 0)
        for phase in range(4):
            controller.state.phase_index = phase
            params = controller.parameters()
            self.assertEqual(params.gate_window_scale, 1.)
            self.assertEqual(params.gate1_fraction, .4)
            self.assertTrue(params.prioritized_local_starts)
            self.assertAlmostEqual(.6 * params.local_gate_probabilities[1], .4)
            self.assertAlmostEqual(float(params.local_gate_probabilities.sum()), 1., places=6)
            controller.step()
            self.assertEqual(controller.state.gate_window_scale, 1.)
        with self.assertRaises(ValueError):
            _build_training_config(parse_args(["--profile", "rtx-5050-corner", "--strict-course-training"]))

    def test_recorded_state_partial_reset_and_three_gate_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            pos = np.array([[28., 27., 1.2]], dtype=np.float32)
            vel = np.array([[3., -4., 0.1]], dtype=np.float32)
            angular = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            np.savez(path, pos=pos, vel=vel, quat=[[0., 0., 0., 1.]],
                     angular=angular, rpm=[[18000.] * 4], action=[[.1] * 4],
                     gate_center=[29., 24., 1.], gate_normal=[0.33035042, -0.94385836, 0.])
            env = CrazyflowRacingEnv(RacingEnvConfig(
                num_envs=2, sim_hz=200, control_hz=100, auto_reset=False,
                corner_reset_bank=str(path)))
            try:
                env.reset(seed=7, forced_reset_gates=jnp.array([2, 0]))
                np.testing.assert_allclose(env.sim.data.states.pos[0, 0], pos[0])
                np.testing.assert_allclose(env.sim.data.states.vel[0, 0], vel[0])
                np.testing.assert_allclose(env.sim.data.states.ang_vel[0, 0], angular[0])
                np.testing.assert_allclose(env.sim.data.states.rotor_vel[0, 0], 18000.)
                np.testing.assert_allclose(env.last_action[0], .1)
                other = np.asarray(env.sim.data.states.pos[1]).copy()
                env.reset(mask=jnp.array([True, False]), forced_reset_gates=jnp.array([2, 0]))
                np.testing.assert_array_equal(env.sim.data.states.pos[1], other)
                for gate in (2, 3, 4):
                    state = env.sim.data.states
                    normal = env.gate_normals[gate]
                    state = state.replace(
                        pos=state.pos.at[0, 0].set(env.gate_centers[gate] - .02 * normal),
                        vel=state.vel.at[0, 0].set(5. * normal),
                        quat=state.quat.at[0, 0].set(jnp.array([0., 0., 0., 1.])),
                        ang_vel=state.ang_vel.at[0, 0].set(0.),
                        rotor_vel=state.rotor_vel.at[0, 0].set(env.motor_rpm_hover),
                    )
                    env.sim.data = env.sim.data.replace(states=state)
                    _, _, terminated, _, info = env.step(jnp.zeros((2, 4)))
                    self.assertTrue(bool(info["passed_gate"][0]))
                    self.assertEqual(bool(terminated[0]), gate == 4)
                    self.assertEqual(bool(info["local_segment_complete"][0]), gate == 4)
                    self.assertFalse(bool(info["course_finished"][0]))
                    self.assertEqual(float(info["reward_finish"][0]), 0.)
            finally:
                env.close()
            evaluation = CrazyflowRacingEnv(RacingEnvConfig(
                num_envs=2, corner_reset_bank=str(path), reset_distribution="evaluation"))
            try:
                self.assertIsNone(evaluation.corner_bank)
                self.assertTrue(np.all(np.asarray(evaluation.reset_gate) == 0))
            finally:
                evaluation.close()
