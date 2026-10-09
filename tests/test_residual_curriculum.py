import sys
from dataclasses import replace
from pathlib import Path
import unittest
import tempfile
import pickle
from unittest.mock import patch
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_residual_curriculum as strategy
from run_residual_curriculum import STAGES, stage_command
from run_speed_curriculum import Runner
from a2rl_drone_training.config import ResidualConfig
from a2rl_drone_training.hierarchical.environment import ResidualMPCEnv
from a2rl_drone_training.hierarchical.planner import curvature_speed_profile
from a2rl_drone_training.train import build_parser


class ResidualCurriculumTests(unittest.TestCase):
    def test_curvature_profile_obeys_lateral_acceleration_and_braking(self):
        theta = np.linspace(0, np.pi, 100)
        points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(100)))
        tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(100)))
        second = -points
        time, speed = curvature_speed_profile(points, tangent, second, 4., 2., 3.)
        self.assertEqual(speed[0], 0.)
        self.assertTrue(np.all(np.diff(time) > 0))
        self.assertLessEqual(speed.max(), 2. + 1e-8)
        ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertTrue(np.all(np.diff(speed**2) <= 4 * ds + 1e-8))
        self.assertTrue(np.all(-np.diff(speed**2) <= 6 * ds + 1e-8))
        straight = np.column_stack((theta, np.zeros(100), np.zeros(100)))
        _, straight_speed = curvature_speed_profile(straight, np.tile([1., 0, 0], (100,1)),
                                                    np.zeros((100,3)), 4., 2., 3.)
        self.assertGreater(straight_speed.max(), speed.max())

    def test_recovery_authority_uses_original_path_and_smoothly_returns_to_zero(self):
        from types import SimpleNamespace
        from a2rl_drone_training.hierarchical.environment import smooth_offset
        env = object.__new__(ResidualMPCEnv)
        env.residual = ResidualConfig(recovery_only=True, position_limit_m=.1,
                                      speed_fraction=.15, speedup_fraction=0.)
        env.base = SimpleNamespace(gate_counter=jnp.zeros(5, dtype=int))
        nominal = jnp.zeros((5,17))
        env.reference = SimpleNamespace(project=lambda position, gates: jnp.zeros(5),
                                       sample=lambda time: (nominal,None,None))
        env.offset = jnp.ones((5,4)) * 100  # Must never influence deviation detection.
        state = nominal.at[:,0].set(jnp.array([0.,.04,.05,.1,.15]))
        np.testing.assert_allclose(env.recovery_authority(state), [0,0,0,.5,1], atol=1e-6)
        velocity_state = nominal.at[:,3].set(jnp.array([0.,.4,.5,1.,1.5]))
        np.testing.assert_allclose(env.recovery_authority(velocity_state), [0,0,0,.5,1], atol=1e-6)
        target = env.residual_target(jnp.ones((5,4))) * env.recovery_authority(state)[:,None]
        np.testing.assert_array_equal(target[:3], np.zeros((3,4)))
        np.testing.assert_array_equal(target[:,3], np.zeros(5))
        self.assertLessEqual(float(target[:,:3].max()), .100001)
        value, velocity, _, _ = smooth_offset(jnp.ones((1,4))*.1,jnp.zeros((1,4)),
                                              jnp.zeros((1,4)),2.,.2)
        self.assertLess(float(jnp.max(jnp.abs(value))), .0001)

    def test_critical_recovery_thresholds_gates_turns_and_straights(self):
        from types import SimpleNamespace
        env = object.__new__(ResidualMPCEnv)
        env.residual = ResidualConfig(recovery_only=True, recovery_critical_areas=True)
        env.base = SimpleNamespace(gate_counter=jnp.zeros(1,dtype=int))
        def sample(t):
            x=jnp.zeros(t.shape+(17,)).at[...,3].set(2.)
            a=jnp.zeros(t.shape+(3,))
            return x,None,a
        env.reference=SimpleNamespace(time=jnp.array([0.,2.,4.]),gate_indices=jnp.array([1]),
                                      project=lambda p,g:jnp.zeros(p.shape[0]), sample=sample)
        np.testing.assert_allclose(env.critical_area_weight(jnp.array([0.,2.,4.])),[0,1,0])
        state=jnp.zeros((1,17)).at[:,0].set(.03).at[:,3].set(2.)
        self.assertEqual(float(env.recovery_authority(state)[0]),0.)
        env.reference.project=lambda p,g:jnp.full(p.shape[0],2.)
        self.assertGreater(float(env.recovery_authority(state)[0]),0.)
        state=state.at[:,0].set(0.)
        self.assertEqual(float(env.recovery_authority(state)[0]),0.)
        def turn_sample(t):
            x,_,a=sample(t)
            return x,None,a.at[...,1].set(2.)  # curvature .5/m at 2 m/s
        env.reference.sample=turn_sample
        self.assertAlmostEqual(float(env.critical_area_weight(jnp.array([0.]))[0]),1.,places=5)

    def test_only_positive_rl_speed_allowance_changes(self):
        env = object.__new__(ResidualMPCEnv)
        action = jnp.array([[.2, -.2, .1, 1.], [.2, -.2, .1, -1.], [0.,0.,0.,0.]])
        for stage in STAGES:
            env.residual = ResidualConfig(speed_fraction=.3, speedup_fraction=stage['speedup'])
            target = np.asarray(env.residual_target(action))
            self.assertAlmostEqual(target[0,3], stage['speedup'], places=6)
            self.assertAlmostEqual(target[1,3], -.3, places=6)
            np.testing.assert_allclose(target[:2,:3], [[.1,-.1,.05]]*2)
            np.testing.assert_array_equal(target[2], np.zeros(4))
        env.residual = replace(env.residual, speedup_fraction=None)
        self.assertAlmostEqual(float(env.residual_target(action)[0,3]), .3, places=6)

    def test_curriculum_reuses_reference_and_checkpoint_across_stages(self):
        from test_speed_curriculum import result
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(Path(folder) / "run", {}, 1)
            trained = []
            def execute(key, command, outputs, **kwargs):
                for output in outputs:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.suffix == '.pkl':
                        update = int(command[command.index('--total-env-steps') + 1]) // 1024
                        trained.append((update, command))
                        output.write_bytes(pickle.dumps(dict(controller='residual_mpc', reference_fingerprint='same', updates=update)))
                    else:
                        output.touch()
            def evaluate(r, key, reference, directory, count, seed, checkpoint=None, condition='exact_state', nominal=False):
                return result(n=count), result(n=count) if checkpoint else None
            with patch.object(runner, 'run', side_effect=execute), patch.object(strategy, 'reference_metrics', return_value={'planner_cap_m_s': None, 'fingerprint':'same'}), patch.object(strategy, 'evaluate', side_effect=evaluate), patch.object(strategy, 'final_validate') as final:
                strategy.pipeline(runner)
            self.assertEqual([entry[0] for entry in trained], [32,64,96])
            self.assertNotIn('--restore-checkpoint', trained[0][1])
            self.assertIn('--restore-checkpoint', trained[1][1])
            self.assertIn('--restore-checkpoint', trained[2][1])
            final.assert_called_once()
            self.assertEqual(final.call_args.args[1], runner.directory / 'reference.npz')

    def test_stage_commands_keep_reference_and_raise_exploration_bounds(self):
        previous_ceiling = 0.
        for index, stage in enumerate(STAGES):
            args = build_parser().parse_args(stage_command(Path('same.npz'), Path('new'), 32*(index+1), stage, Path('previous.pkl'))[3:])
            self.assertEqual(args.reference_path, Path('same.npz'))
            self.assertEqual(args.restore_checkpoint, Path('previous.pkl'))
            self.assertEqual(args.residual_speedup_fraction, stage['speedup'])
            self.assertGreaterEqual(args.exploration_std_floor, previous_ceiling)
            self.assertEqual(args.exploration_std_start, args.exploration_std_end)
            self.assertEqual(args.total_env_steps, 32768*(index+1))
            self.assertEqual(args.schedule_env_steps, 384*1024)
            previous_ceiling = args.exploration_std_end


if __name__ == '__main__':
    unittest.main()
