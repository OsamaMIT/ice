import argparse
from dataclasses import replace
import json
import pickle
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_speed_curriculum as runner


def result(lap=20., n=8):
    return dict(success=np.ones(n, dtype=bool), collision=np.zeros(n, dtype=bool),
                fallbacks=np.zeros(n, dtype=int), gate_passes=np.ones((n, 12)),
                lap_time=np.full(n, lap), tracking_error=np.full(n, .01),
                minimum_clearance=np.full(n, .2), latency_s=np.full(n, .05),
                physics_steps=np.full(n, 10000), peak_speed=np.full(n, 8.), wall_seconds=np.array(100.))


class SpeedCurriculumTests(unittest.TestCase):
    def test_training_arguments_and_fixed_schedule(self):
        from a2rl_drone_training.train import build_parser
        args = build_parser().parse_args(runner.training_command(Path('ref.npz'), Path('ckpt'), 32)[3:])
        self.assertEqual(args.controller, 'residual_mpc')
        self.assertEqual(args.total_env_steps, 32768)
        self.assertEqual(args.schedule_env_steps, 131072)
        self.assertEqual(args.checkpoint_interval, 1)
        self.assertEqual(args.estimator_position_noise_std, .01)

    def test_equal_speed_allowed_but_cap_and_model_must_match(self):
        previous = dict(model_fingerprint='same', duration_s=53., peak_speed_m_s=3.)
        current = dict(previous, planner_cap_m_s=None)
        runner.require_faster_reference(current, previous, None)
        for bad in (dict(model_fingerprint='different'), dict(planner_cap_m_s=24.)):
            with self.assertRaises(ValueError):
                runner.require_faster_reference(dict(current, **bad), previous, None)

    def test_promotion_within_two_percent_and_failure_rejection(self):
        runner.require_pair(result(), result(20.4), 8)
        with self.assertRaises(ValueError):
            runner.require_pair(result(), result(20.401), 8)
        for key in ('collision', 'fallbacks', 'gate_passes', 'success'):
            data = result(19.)
            data[key].flat[0] = 1 if key in ('collision', 'fallbacks') else 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.require_pair(result(), data, 8)

    def test_checkpoint_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pkl'
            path.write_bytes(pickle.dumps(dict(controller='residual_mpc', reference_fingerprint='a', updates=32)))
            self.assertEqual(runner.checkpoint_updates(path, {'fingerprint': 'a'}), 32)
            with self.assertRaises(ValueError):
                runner.checkpoint_updates(path, {'fingerprint': 'b'})

    def test_speed_parser_and_metadata_roundtrip(self):
        from a2rl_drone_training.hierarchical.cli import parse_speed_cap
        from a2rl_drone_training.hierarchical.reference import Reference
        self.assertIsNone(parse_speed_cap('none'))
        self.assertEqual(parse_speed_cap('6'), 6.)
        for invalid in ('0', '-1', 'nan', 'inf', '-inf', 'bad'):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_speed_cap(invalid)
        states = np.zeros((2, 17)); states[:, 9] = 1
        ref = Reference(np.array([0., 1.]), states, np.zeros((2, 4)), np.array([1]),
                        dict(validated=True, planner={'max_speed_m_s': None}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ref.npz'; ref.save(path)
            self.assertIsNone(Reference.load(path).metadata['planner']['max_speed_m_s'])
            self.assertEqual(Reference.load(path).fingerprint, ref.fingerprint)

    def test_uncapped_keeps_other_dense_checks(self):
        from a2rl_drone_training.config import PlannerConfig
        from a2rl_drone_training.hierarchical.planner import validate_reference
        states = np.zeros((2, 17)); states[:, 9] = 1; states[:, 3] = 50
        states[:, 13:] = 1
        ref = SimpleNamespace(time=np.array([0., .002]), states=states,
                              actions=np.zeros((2, 4)), gate_indices=np.array([1]))
        course = SimpleNamespace(bounds_min=np.full(3, -100), bounds_max=np.full(3, 100),
                 centers=np.zeros((1, 3)), normals=np.array([[1., 0, 0]]),
                 right_axes=np.array([[0., 1, 0]]), widths=np.array([2.]), heights=np.array([2.]))
        trajectory = states[None].copy()
        model = SimpleNamespace(rpm_min=0., rpm_max=2., rpm_hover=1.,
                                rollout_intervals=lambda *args: trajectory)
        boxes = SimpleNamespace(clearance=lambda *args: np.ones((1, 2)),
                                swept_collision=lambda *args: np.zeros((1, 1), dtype=bool))
        config = PlannerConfig(max_speed_m_s=None)
        with patch('a2rl_drone_training.hierarchical.planner.frame_boxes', return_value=boxes):
            self.assertTrue(validate_reference(ref, course, model, config, .15)['validated'])
            self.assertFalse(validate_reference(ref, course, model, replace(config, max_speed_m_s=6.), .15)['validated'])
            for column, value in ((10, 20.), (13, 4.), (0, 200.), (3, np.nan)):
                saved = trajectory.copy()
                trajectory[:, :, column] = value
                self.assertFalse(validate_reference(ref, course, model, config, .15)['validated'])
                trajectory[:] = saved
            boxes.swept_collision = lambda *args: np.ones((1, 1), dtype=bool)
            self.assertFalse(validate_reference(ref, course, model, config, .15)['validated'])

    def test_completed_task_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'; out = Path(tmp) / 'out'
            r = runner.Runner(directory, {'v': 1}, 1)
            command = [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("done")', str(out)]
            r.run('test', command, [out], cpu=True)
            resumed = runner.Runner(directory, {'v': 1}, 1, True)
            with patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('reran completed work')):
                resumed.run('test', command, [out], cpu=True)
            out.write_text('changed')
            with self.assertRaises(ValueError):
                resumed.run('test', command, [out], cpu=True)
            with self.assertRaises(ValueError):
                runner.Runner(directory, {'v': 2}, 1, True)

    def test_deadline_stops_child_and_incomplete_work_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'; out = Path(tmp) / 'out'
            r = runner.Runner(directory, {}, .0001)
            with self.assertRaises(runner.BudgetExceeded):
                r.run('slow', [sys.executable, '-c', 'import time; time.sleep(10)'], [out], cpu=True)
            self.assertIsNone(r.state['child_pid'])
            self.assertFalse(r.state['tasks']['slow']['completed'])
            resumed = runner.Runner(directory, {}, 1, True)
            resumed.run('slow', [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', str(out)], [out], cpu=True)
            self.assertEqual(resumed.state['tasks']['slow']['attempt'], 2)

    def test_persistent_pair_recovery_and_tamper(self):
        from evaluate_seeded_condition import load_pair, save_pair
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.assertIsNone(load_pair(directory, {'seed': 1}))
            save_pair(directory, {'seed': 1}, result(n=1), result(n=1))
            a, b = load_pair(directory, {'seed': 1})
            np.testing.assert_array_equal(a['success'], [True])
            with self.assertRaises(ValueError):
                load_pair(directory, {'seed': 2})
            (directory / 'baseline.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                load_pair(directory, {'seed': 1})

    def test_rl_first_curriculum_and_failed_screening_blocks_promotion(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); run = runner.Runner(root / 'run', {}, 1)
                initial = root / 'initial.npz'; initial.touch()
                planned, training, screens = [], [], []
                def metrics(path):
                    cap = {'12mps': 12., '24mps': 24., 'uncapped': None}.get(path.parent.name)
                    return dict(model_fingerprint='m', fingerprint=path.parent.name,
                                planner_cap_m_s=cap, duration_s=20., peak_speed_m_s=11.)
                def execute(key, command, outputs, inputs=(), **kwargs):
                    if key.endswith('/planning'):
                        planned.append(command[command.index('--max-speed') + 1])
                        self.assertEqual(command[command.index('--motor-command-limit') + 1], '0.8')
                    for path in outputs:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if path.suffix == '.pkl':
                            self.assertNotIn('--restore-checkpoint', command)
                            self.assertIn('--residual-recovery-only', command)
                            self.assertEqual(command[command.index('--residual-speedup-fraction') + 1], '0')
                            self.assertEqual(command[command.index('--spawn-position-std') + 1], '.08')
                            training.append(key)
                            update = int(command[command.index('--total-env-steps')+1]) // 1024
                            path.write_bytes(pickle.dumps(dict(controller='residual_mpc',
                                reference_fingerprint=path.parent.parent.name, updates=update)))
                        else:
                            path.touch()
                def evaluate(r, key, ref, directory, trials, seed, checkpoint=None, condition='exact_state', nominal=False):
                    self.assertIsNotNone(checkpoint)
                    self.assertIn('/screen', key)
                    screens.append(key)
                    baseline, learned = result(n=trials), result(n=trials)
                    if fail:
                        learned['success'][0] = False
                    return baseline, learned
                with patch.object(run, 'run', side_effect=execute), patch.object(runner, 'reference_metrics', side_effect=metrics), patch.object(runner, 'evaluate', side_effect=evaluate), patch.object(runner, 'final_validate') as final:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'all 8 training blocks failed'):
                            runner.pipeline(run, initial, caps=[12.,24.,None], train_before_qualification=True, motor_command_limit=.8, screen_every=16, recovery_only=True)
                        self.assertEqual(planned, ['12.0'])
                        self.assertEqual(len(training), 8)
                        final.assert_not_called()
                    else:
                        runner.pipeline(run, initial, caps=[12.,24.,None], train_before_qualification=True, motor_command_limit=.8, screen_every=16, recovery_only=True)
                        self.assertEqual(planned, ['12.0','24.0','none'])
                        self.assertEqual(len(training), 3)
                        self.assertEqual(len(screens), 6)
                        final.assert_called_once()

    def test_development_validation_uses_140_pairs_without_full_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=runner.Runner(root/'run', {'final_trials':20}, 1)
            reference=root/'reference.npz'; reference.touch()
            checkpoint=root/'checkpoint.pkl'; checkpoint.touch()
            seeds=[]; commands=[]
            def execute(key, command, outputs, **kwargs):
                commands.append(command)
                self.assertEqual(command[command.index('--trials')+1], '20')
            def load(directory, signature):
                seeds.append(signature['seed'])
                return result(n=1),result(n=1)
            with patch.object(run,'run',side_effect=execute), patch.object(runner,'reference_metrics',return_value={'planner_cap_m_s':None}), patch('evaluate_seeded_condition.load_pair',side_effect=load):
                runner.final_validate(run,reference,checkpoint)
            self.assertEqual(len(commands),7)
            self.assertEqual(len(set(seeds)),140)
            self.assertEqual(run.state['completed_final_trials'],140)
            self.assertEqual(run.state['stage'],'development_validation_passed')
            report=json.loads((run.directory/'final_report.json').read_text())
            self.assertFalse(report['accepted'])
            self.assertTrue(report['development_validation_passed'])
            self.assertEqual(report['trials_per_condition'],20)

    def test_atomic_json_retries_transient_sharing_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json'
            original=Path.replace
            calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)==1:
                    raise PermissionError('reader sharing violation')
                return original(source,target)
            with patch.object(Path,'replace',replace), patch.object(runner.time,'sleep'):
                runner.atomic_json(path,{'ok':True})
            self.assertEqual(json.loads(path.read_text()),{'ok':True})
            self.assertEqual(len(calls),2)

    def test_full_orchestration_with_fake_flights(self):
        # Exercise every stage and all final seeds without costly simulation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); r = runner.Runner(root / 'run', {}, 1)
            initial = root / 'initial.npz'; initial.touch()
            def metrics(path):
                cap = {'6mps': 6., '12mps': 12., '24mps': 24., 'uncapped': None}.get(path.parent.name, 3.)
                return dict(model_fingerprint='m', fingerprint='r', planner_cap_m_s=cap,
                            duration_s=20., peak_speed_m_s=6.)
            def fake_run(key, command, outputs, inputs=(), **kwargs):
                for path in outputs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == '.pkl':
                        path.write_bytes(pickle.dumps(dict(controller='residual_mpc', reference_fingerprint='r', updates=32)))
                    else:
                        path.touch()
            seeds = []
            def fake_load_pair(directory, signature):
                seeds.append(signature['seed'])
                return result(n=1), result(n=1)
            def fake_evaluate(r, key, ref, directory, trials, seed, checkpoint=None, condition='exact_state', nominal=False):
                if key.startswith('final/'):
                    seeds.append(seed)
                return result(n=trials), result(n=trials) if checkpoint else None
            with patch.object(r, 'run', side_effect=fake_run), patch.object(runner, 'reference_metrics', side_effect=metrics), patch.object(runner, 'evaluate', side_effect=fake_evaluate), patch('evaluate_seeded_condition.load_pair', side_effect=fake_load_pair):
                runner.pipeline(r, initial)
            self.assertEqual(r.state['stage'], 'uncapped_simulation_accepted')
            self.assertEqual(len(seeds), 700)
            self.assertEqual(len(set(seeds)), 700)
            self.assertTrue(json.loads((r.directory / 'final_report.json').read_text())['accepted'])


if __name__ == '__main__':
    unittest.main()
