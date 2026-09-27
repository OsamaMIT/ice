"""Contracts for gate geometry, shared dynamics, MPC, and multi-rate RL."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import (
    RacingEnvConfig,
    TrainingConfig,
    ResidualConfig,
    MPCConfig,
)
from a2rl_drone_training.course import make_gate_course, arena_38m_stacked_course
from a2rl_drone_training.hierarchical.geometry import (
    frame_boxes,
    validate_openings,
    fingerprint,
)
from a2rl_drone_training.hierarchical.reference import Reference
from a2rl_drone_training.hierarchical.dynamics import DroneModel, attitude_error
from a2rl_drone_training.hierarchical.environment import (
    smooth_offset,
    residual_training_config,
    ResidualMPCEnv,
)
from a2rl_drone_training.hierarchical.mpc import QuaternionMPC
from a2rl_drone_training.actions import validate_checkpoint_actions, ACTION_SPACE
from a2rl_drone_training.ppo import compute_gae


def straight_course():
    return make_gate_course(
        np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        normals=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        widths=np.ones(2) * 1.5,
        heights=np.ones(2) * 1.5,
        outer_widths=np.ones(2) * 2.7,
        outer_heights=np.ones(2) * 2.7,
        nominal_racing_line=np.array(
            [[-1.5, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
        ),
    )


def stacked_course():
    return make_gate_course(
        np.array([[0.0, 0.0, 2.5], [0.0, 0.0, 1.0]]),
        normals=np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        widths=np.full(2, 1.5),
        heights=np.full(2, 1.5),
        outer_widths=np.full(2, 2.7),
        outer_heights=np.full(2, 2.7),
        logical_gate_ids=np.array([1, 1]),
        openings=("top", "bottom"),
        nominal_racing_line=np.array(
            [[0.0, 1.5, 2.5], [0.0, 0.0, 2.5], [0.0, 0.0, 1.0]]
        ),
    )


class GeometryTests(unittest.TestCase):
    def test_swept_edges_and_center(self):
        boxes = frame_boxes(straight_course())
        start = jnp.array([[-1.0, 0.0, 1.0], [-1.0, 0.7, 1.0], [-1.0, 3.0, 1.0]])
        end = start.at[:, 0].set(1.0)
        np.testing.assert_array_equal(
            boxes.swept_collision(start, end, 0.15), [False, True, False]
        )
        self.assertGreater(
            float(boxes.clearance(jnp.array([[0.0, 0.0, 1.0]]), 0.15)[0]), 0
        )

    def test_stacked_openings_do_not_have_phantom_barriers(self):
        course = arena_38m_stacked_course()
        boxes = frame_boxes(course)
        for idx in (6, 7, 10, 11):
            center, normal = course.centers[idx], course.normals[idx]
            self.assertFalse(
                bool(
                    boxes.swept_collision(
                        jnp.asarray(center - normal), jnp.asarray(center + normal), 0.15
                    )
                )
            )

    def test_infeasible_opening_rejected(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            validate_openings(straight_course(), 0.7, 0.1)

    def test_parallel_segment_outside_slab(self):
        boxes = frame_boxes(straight_course())
        self.assertFalse(
            bool(
                boxes.swept_collision(
                    jnp.array([-0.5, -2.0, 1.0]), jnp.array([-0.5, 2.0, 1.0]), 0.15
                )
            )
        )


class ReferenceTests(unittest.TestCase):
    def reference(self):
        x = np.zeros((5, 17))
        x[:, 0] = np.arange(5)
        x[:, 3] = 1
        x[:, 9] = 1
        x[:, 13:] = 1
        return Reference(
            np.arange(5, dtype=float),
            x,
            np.zeros((5, 4)),
            np.array([2, 4]),
            {"validated": True},
        )

    def test_round_trip_and_tamper(self):
        ref = self.reference()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ref.npz"
            ref.save(path)
            restored = Reference.load(path)
            self.assertEqual(restored.fingerprint, ref.fingerprint)
            with self.assertRaisesRegex(ValueError, "course_fingerprint"):
                Reference.load(path, course_fingerprint="different")

    def test_derivative_consistency_and_local_projection(self):
        ref = self.reference()
        x, _, a = ref.sample(jnp.array([0.5, 2.5]))
        np.testing.assert_allclose(x[:, 0], [0.5, 2.5])
        np.testing.assert_allclose(x[:, 3], 1)
        np.testing.assert_allclose(a, 0)
        times = ref.project(jnp.array([[3.9, 0, 0], [0.1, 0, 0]]), jnp.array([0, 1]))
        np.testing.assert_allclose(times, [2.0, 2.0])
        tail, _, acceleration = ref.sample(jnp.array([4.5]))
        np.testing.assert_allclose(tail[:, 0], [4.5])
        np.testing.assert_allclose(tail[:, 3], [1.0])
        np.testing.assert_allclose(acceleration, 0.0)

    def test_quaternion_sign_invariance(self):
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q = q / jnp.linalg.norm(q)
        np.testing.assert_allclose(
            attitude_error(q, q), attitude_error(-q, q), atol=1e-6
        )

    def test_filter_derivatives_and_zero(self):
        value = jnp.zeros(4)
        velocity = jnp.zeros(4)
        target = jnp.array([0.5, 0, 0, 0.3])
        o, v, a, integral = smooth_offset(value, velocity, target, 0.1, 0.2)
        derivative = jax.jacfwd(
            lambda t: smooth_offset(value, velocity, target, t, 0.2)[0]
        )(0.1)
        np.testing.assert_allclose(v, derivative, atol=1e-6)
        np.testing.assert_allclose(
            a,
            jax.jacfwd(lambda t: smooth_offset(value, velocity, target, t, 0.2)[1])(
                0.1
            ),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            o,
            jax.jacfwd(lambda t: smooth_offset(value, velocity, target, t, 0.2)[3])(
                0.1
            ),
            atol=1e-6,
        )
        self.assertTrue(
            all(
                np.allclose(x, 0)
                for x in smooth_offset(value, velocity, value, 0.1, 0.2)
            )
        )

    def test_physical_discounts_and_mode_validation(self):
        cfg = TrainingConfig(
            controller="residual_mpc",
            residual=ResidualConfig(reference_path=Path("reference.npz")),
        )
        cfg = residual_training_config(cfg)
        self.assertEqual(cfg.obs.dim, 82)
        self.assertAlmostEqual(cfg.ppo.gamma**20, 0.98)
        self.assertEqual(residual_training_config(cfg), cfg)
        with self.assertRaises(ValueError):
            residual_training_config(replace(cfg, mpc=replace(cfg.mpc, update_hz=99)))
        validate_checkpoint_actions({"action_space": ACTION_SPACE})
        with self.assertRaises(ValueError):
            validate_checkpoint_actions(
                {"action_space": ACTION_SPACE}, "residual_mpc", "reference"
            )

    def test_partial_step_truncation_bootstrap(self):
        advantage, _ = compute_gae(
            jnp.array([[1.0]]),
            jnp.array([[0.0]]),
            jnp.array([[2.0]]),
            jnp.array([[False]]),
            jnp.array([[True]]),
            gamma=0.81,
            gae_lambda=0.9,
            transition_fraction=jnp.array([[0.5]]),
        )
        np.testing.assert_allclose(advantage, [[2.8]])

    def test_acceptance_requires_reliability_and_paired_improvement(self):
        from a2rl_drone_training.hierarchical.evaluation import compare_trials

        baseline = dict(
            success=np.ones(100, dtype=bool),
            lap_time=np.full(100, 10.0),
            collision=np.zeros(100, dtype=bool),
            minimum_clearance=np.full(100, 0.3),
            tracking_error=np.full(100, 0.02),
            fallbacks=np.zeros(100),
            latency_s=np.array([0.001, 0.002]),
            physics_steps=np.array([1000]),
            wall_seconds=2.0,
        )
        learned = {**baseline, "lap_time": np.full(100, 9.0)}
        self.assertTrue(compare_trials(baseline, learned)["accepted"])
        self.assertFalse(compare_trials(baseline, baseline)["accepted"])
        unreliable = {**learned, "success": np.arange(100) >= 6}
        report = compare_trials(baseline, unreliable)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["failures"]["learned"], list(range(6)))


class ControlIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from a2rl_drone_training.env import CrazyflowRacingEnv

        cls.env = CrazyflowRacingEnv(
            RacingEnvConfig(num_envs=2, auto_reset=False), course=straight_course()
        )
        cls.model = DroneModel.from_env(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def hover(self):
        x = (
            jnp.zeros(17)
            .at[:3]
            .set(jnp.array([-1.0, 0.0, 1.0]))
            .at[9]
            .set(1.0)
            .at[13:]
            .set(1.0)
        )
        return x

    def test_prediction_matches_crazyflow_and_symbolic(self):
        env = self.env
        env.reset(seed=42)
        x = self.model.pack(env.sim.data.states)
        u = jnp.array([[0.03, -0.02, 0.01, -0.03], [0.02, 0.01, -0.01, 0.03]])
        expected = jax.vmap(
            lambda state, action: self.model.step(state, action, 0.002)
        )(x, u)
        env.step(u)
        np.testing.assert_allclose(
            expected, self.model.pack(env.sim.data.states), atol=1e-5
        )
        symbolic = self.model.symbolic_step(1)
        np.testing.assert_allclose(
            np.asarray(symbolic(np.asarray(x[0]), np.asarray(u[0]), 0.002)).ravel(),
            expected[0],
            atol=1e-5,
        )

    def test_optimizer_produces_densely_validated_reference(self):
        from a2rl_drone_training.config import PlannerConfig
        from a2rl_drone_training.hierarchical.planner import (
            optimize_reference,
            validate_reference,
        )

        config = PlannerConfig(
            nodes_per_segment=12, max_iterations=200, max_refinements=2
        )
        ref = optimize_reference(self.env.course, self.model, config)
        self.assertTrue(ref.metadata["validated"])
        self.assertTrue(
            validate_reference(ref, self.env.course, self.model, config, 0.15)[
                "validated"
            ]
        )
        self.assertLess(ref.time[-1], 2.0)
        np.testing.assert_allclose(ref.states[0, 3:6], 0.0, atol=1e-6)
        np.testing.assert_array_equal(ref.gate_indices.shape, (2,))

    def test_stacked_seed_has_far_side_turnaround(self):
        from a2rl_drone_training.hierarchical.planner import initial_guess

        course = arena_38m_stacked_course()
        _, states, commands, events = initial_guess(course, self.model, 24)
        np.testing.assert_allclose(states[0, 3:6], 0.0, atol=1e-6)
        for first, second in ((6, 7), (10, 11)):
            position = states[events[first] : events[second], :3]
            distance = (position - course.centers[first]) @ course.normals[first]
            self.assertGreater(float(distance.max()), 1.4)
        self.assertTrue(np.isfinite(commands).all())

    def test_stacked_optimization_can_depart_the_shared_gate_plane(self):
        from a2rl_drone_training.config import PlannerConfig
        from a2rl_drone_training.hierarchical.planner import optimize_reference

        course = stacked_course()
        reference = optimize_reference(
            course,
            self.model,
            PlannerConfig(
                nodes_per_segment=12,
                max_iterations=400,
                max_refinements=2,
                max_speed_m_s=6,
            ),
        )
        self.assertTrue(reference.metadata["validated"])
        crossing = reference.states[reference.gate_indices, :3]
        np.testing.assert_allclose(crossing[:, 1], 0.0, atol=1e-3)
        self.assertGreater(crossing[0, 2], crossing[1, 2])

    def test_hover_mpc_and_failure_escalation(self):
        mpc = QuaternionMPC(self.model, MPCConfig(horizon_s=0.1, iterations=1), 2)
        x = jnp.tile(self.hover(), (2, 1))
        refs = jnp.tile(x[:, None], (1, 6, 1))
        feed = jnp.zeros((2, 5, 4))
        command, info = mpc.command(x, refs, feed)
        self.assertTrue(np.asarray(info["mpc_success"]).all())
        np.testing.assert_allclose(command, 0, atol=1e-5)
        with patch.object(
            mpc,
            "_batch_solve",
            return_value=(
                jnp.zeros((2, 5, 4)),
                jnp.zeros(2, dtype=bool),
                jnp.full(2, jnp.inf),
            ),
        ):
            _, info = mpc.command(x, refs, feed)
            self.assertTrue(np.asarray(info["mpc_reused"]).all())
            command, info = mpc.command(x, refs, feed)
            self.assertFalse(np.asarray(info["mpc_reused"]).any())
            self.assertTrue(np.isfinite(command).all())
            self.assertLessEqual(float(jnp.max(jnp.abs(command))), 1)
        mpc.reset(jnp.array([True, False]))
        np.testing.assert_array_equal(mpc.failures, [0, 2])

    def test_residual_partial_terminal_and_independent_reset(self):
        x = np.tile(np.asarray(self.hover()), (4, 1))
        x[:, 0] = [-1.5, -0.75, 0, 2]
        x[:, 3] = 0
        ref = Reference(
            np.array([0.0, 1.0, 2.0, 4.0]),
            x,
            np.zeros((4, 4)),
            np.array([2, 3]),
            {
                "validated": True,
                "course_fingerprint": fingerprint(self.env.course),
                "model_fingerprint": self.model.fingerprint,
                "vehicle_radius_m": 0.15,
                "planner": {"frame_depth_m": 0.1},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            ref.save(path)
            cfg = TrainingConfig(
                controller="residual_mpc",
                env=RacingEnvConfig(
                    num_envs=2, auto_reset=False, artificial_time_limit_s=0.006
                ),
                residual=ResidualConfig(reference_path=path),
                mpc=MPCConfig(horizon_s=0.1, iterations=1),
            )
            env = ResidualMPCEnv(cfg, course=self.env.course)
            try:
                refs, ff = env.references(env.estimated_state())
                nominal, _, _ = ref.sample(jnp.arange(6) * 0.02)
                np.testing.assert_allclose(refs[0], nominal, atol=1e-6)
                obs, reward, term, trunc, info = env.step(jnp.zeros((2, 4)))
                self.assertEqual(obs.shape, (2, 82))
                self.assertTrue(np.asarray(trunc).all())
                np.testing.assert_array_equal(info["physics_steps"], [3, 3])
                np.testing.assert_allclose(info["transition_seconds"], 0.006)
                np.testing.assert_allclose(
                    info["final_critic_observation"], env.privileged_observe()
                )
                env.reset(mask=jnp.array([True, False]))
                np.testing.assert_array_equal(env.done, [False, True])
                np.testing.assert_array_equal(env.mpc.valid, [False, True])
            finally:
                env.close()

    def test_frame_collision_overrides_gate_completion(self):
        from a2rl_drone_training.env import CrazyflowRacingEnv

        env = CrazyflowRacingEnv(
            RacingEnvConfig(num_envs=1, auto_reset=False, gate_frame_collisions=True),
            course=straight_course(),
        )
        try:
            env.reset(forced_reset_gates=jnp.zeros(1, dtype=jnp.int32))
            states = env.sim.data.states
            env.sim.data = env.sim.data.replace(
                states=states.replace(
                    pos=jnp.array([[[-0.001, 0.7, 1.0]]]),
                    vel=jnp.array([[[2.0, 0.0, 0.0]]]),
                )
            )
            _, _, terminated, _, info = env.step(jnp.zeros((1, 4)))
            self.assertTrue(bool(terminated[0]))
            self.assertTrue(bool(info["frame_collision"][0]))
            self.assertFalse(bool(info["passed_gate"][0]))
            self.assertEqual(float(info["reward_gate"][0]), 0.0)
            # An arena-boundary collision uses the same vehicle volume.
            env.reset(forced_reset_gates=jnp.zeros(1, dtype=jnp.int32))
            states = env.sim.data.states
            env.sim.data = env.sim.data.replace(
                states=states.replace(
                    pos=jnp.array([[[-19.9, 0.0, 1.0]]]), vel=jnp.zeros((1, 1, 3))
                )
            )
            _, _, terminated, _, info = env.step(jnp.zeros((1, 4)))
            self.assertTrue(bool(terminated[0]))
            self.assertTrue(bool(info["out_of_bounds"][0]))
        finally:
            env.close()

    def test_residual_trainer_rollout_and_checkpoint(self):
        from a2rl_drone_training.config import PPOConfig, NetworkConfig
        from a2rl_drone_training.trainer import PPOTrainer

        states = np.tile(np.asarray(self.hover()), (4, 1))
        states[:, 0] = [-1.5, -0.75, 0, 2]
        ref = Reference(
            np.array([0.0, 1.0, 2.0, 4.0]),
            states,
            np.zeros((4, 4)),
            np.array([2, 3]),
            {
                "validated": True,
                "course_fingerprint": fingerprint(self.env.course),
                "model_fingerprint": self.model.fingerprint,
                "vehicle_radius_m": 0.15,
                "planner": {"frame_depth_m": 0.1},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            ref.save(path)
            config = TrainingConfig(
                controller="residual_mpc",
                env=RacingEnvConfig(num_envs=2, artificial_time_limit_s=0.004),
                residual=ResidualConfig(reference_path=path),
                mpc=MPCConfig(horizon_s=0.1, iterations=1),
                ppo=PPOConfig(horizon=1, minibatches=1, update_epochs=1),
                net=NetworkConfig(
                    state_hidden=(8,),
                    state_latent_dim=8,
                    gate_hidden=(8,),
                    gate_latent_dim=8,
                    fusion_hidden=(8,),
                    critic_hidden=(8,),
                ),
            )
            trainer = PPOTrainer(config, course=self.env.course)
            try:
                rollout, metrics = trainer.collect_rollout()
                self.assertEqual(rollout.obs.shape, (1, 2, 82))
                self.assertEqual(rollout.critic_obs.shape, (1, 2, 50))
                np.testing.assert_allclose(rollout.transition_fraction, 0.08)
                self.assertEqual(metrics["physics_steps"], 4)
                checkpoint = Path(directory) / "checkpoint.pkl"
                trainer.save_checkpoint(checkpoint)
                trainer.load_checkpoint(checkpoint)
                import pickle

                with checkpoint.open("rb") as f:
                    payload = pickle.load(f)
                self.assertEqual(payload["checkpoint_version"], 7)
                with self.assertRaisesRegex(ValueError, "action space"):
                    validate_checkpoint_actions(payload)
                with self.assertRaisesRegex(ValueError, "reference"):
                    validate_checkpoint_actions(payload, "residual_mpc", "incorrect")
                with patch(
                    "a2rl_drone_training.hierarchical.evaluation.run_trials",
                    return_value={"success": np.array([False])},
                ):
                    with self.assertRaisesRegex(RuntimeError, "Zero-offset MPC"):
                        trainer.train()
            finally:
                trainer.close()


if __name__ == "__main__":
    unittest.main()
