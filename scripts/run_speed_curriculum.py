"""Resumable, budgeted residual-PPO curriculum: 6, 12, 24 m/s, then uncapped."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

from run_staged_validation import pilot_command, require_clean_flights

ROOT = Path(__file__).resolve().parents[1]
CAPS = [6., 12., 24., None]
CONDITIONS = {
    'exact_state': (0., 0., 0.), 'position_1cm': (.01, 0., 0.),
    'position_3cm': (.03, 0., 0.), 'velocity_2cms': (0., .02, 0.),
    'velocity_5cms': (0., .05, 0.), 'delay_10ms': (0., 0., .01),
    'delay_30ms': (0., 0., .03),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def training_command(reference, checkpoints, updates):
    command = pilot_command(reference, checkpoints)
    for flag, value in {'--horizon': 256, '--total-env-steps': updates * 1024,
                        '--schedule-env-steps': 128 * 1024, '--update-epochs': 4,
                        '--checkpoint-interval': 1}.items():
        command[command.index(flag) + 1] = str(value)
    return command + ['--seed', '20260929', '--estimator-position-noise-std', '.01',
                      '--estimator-velocity-noise-std', '0']


def reference_metrics(path):
    import numpy as np
    from a2rl_drone_training.hierarchical.reference import Reference
    reference = Reference.load(path)
    metadata = reference.metadata
    return dict(duration_s=float(reference.time[-1]),
                peak_speed_m_s=float(np.linalg.norm(reference.states[:, 3:6], axis=1).max()),
                model_fingerprint=metadata['model_fingerprint'],
                planner_cap_m_s=metadata['planner']['max_speed_m_s'],
                fingerprint=reference.fingerprint, sha256=digest(path))


def require_faster_reference(current, previous, cap):
    # Kept as a compatibility name: speed increase is no longer required.
    if current['model_fingerprint'] != previous['model_fingerprint']:
        raise ValueError('Vehicle model changed')
    if current['planner_cap_m_s'] != cap:
        raise ValueError('Reference was not generated with the requested speed cap')


class QualificationFailure(ValueError):
    pass


def require_pair(baseline, learned, trials):
    import numpy as np
    try:
        require_clean_flights(baseline, trials)
        require_clean_flights(learned, trials)
    except ValueError as exc:
        raise QualificationFailure(str(exc)) from exc
    if np.mean(learned['lap_time']) > 1.02 * np.mean(baseline['lap_time']) + 1e-9:
        raise QualificationFailure('RL mean lap time exceeds MPC by more than 2%')


def require_promotion(directory, trials):
    import numpy as np
    from a2rl_drone_training.hierarchical.evaluation import compare_trials
    reports = {}
    for suite in ('exact_state', 'noisy_state'):
        with np.load(directory / f'{suite}_baseline.npz', allow_pickle=False) as data:
            baseline = dict(data)
        with np.load(directory / f'{suite}_learned.npz', allow_pickle=False) as data:
            learned = dict(data)
        require_pair(baseline, learned, trials)
        reports[suite] = compare_trials(baseline, learned)
    return reports


def checkpoint_updates(path, reference):
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)  # Only trusted local checkpoints.
    if payload.get('controller') != 'residual_mpc' or payload.get('reference_fingerprint') != reference['fingerprint']:
        raise ValueError('Checkpoint does not match this residual reference')
    return int(payload['updates'])


class BudgetExceeded(RuntimeError):
    pass


class Runner:
    def __init__(self, directory, spec, hours, resume=False):
        self.directory = Path(directory)
        self.limit = hours * 3600
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError('Budget must be positive and finite')
        self.started = time.monotonic()
        path = self.directory / 'status.json'
        if resume:
            self.state = json.loads(path.read_text())
            if self.state['spec'] != spec:
                raise ValueError('Configuration, source code or initial reference changed; use a fresh output directory')
            self.used = self.state['elapsed_seconds']
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.used = 0.
            self.state = dict(spec=spec, tasks={}, stages={}, elapsed_seconds=0., completed_final_trials=0)
        self.state.update(pid=os.getpid(), child_pid=None, process_alive=True)
        self.save('initializing')

    def remaining(self):
        return max(0., self.limit - self.used - (time.monotonic() - self.started))

    def save(self, stage=None, **extra):
        self.state.update(extra)
        if stage is not None:
            self.state['stage'] = stage
        self.state.update(updated_at=time.time(), remaining_seconds=self.remaining(),
                          elapsed_seconds=self.used + time.monotonic() - self.started)
        atomic_json(self.directory / 'status.json', self.state)

    def run(self, key, command, outputs, inputs=(), cpu=False, checkpoints=None, progress_file=None, completed_before=0):
        if self.remaining() <= 0:
            raise BudgetExceeded('Execution budget exhausted')
        signature = {str(path): digest(path) for path in inputs}
        task = self.state['tasks'].get(key)
        if task and task['inputs'] != signature:
            raise ValueError(f'Inputs changed for {key}')
        if task and task.get('completed'):
            if any(not Path(path).is_file() or digest(path) != sha for path, sha in task['outputs'].items()):
                raise ValueError(f'Completed outputs changed for {key}')
            return
        if self.remaining() <= 0:
            raise BudgetExceeded('Execution budget exhausted')
        attempt = task.get('attempt', 0) + 1 if task else 1
        self.state['tasks'][key] = dict(inputs=signature, attempt=attempt, completed=False)
        log = self.directory / (key.replace('/', '_') + f'.attempt{attempt}.log')
        env = dict(os.environ, JAX_PLATFORMS='cpu' if cpu else 'cuda',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   A2RL_RUN_DEADLINE=str(time.time() + self.remaining()))
        self.save(key, current_log=str(log), command=command)
        with log.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            self.save(child_pid=child.pid)
            try:
                while child.poll() is None:
                    if self.remaining() <= 0:
                        raise BudgetExceeded('Execution budget exhausted; last completed update is retained')
                    if checkpoints:
                        numbers = [int(p.stem.rsplit('_', 1)[-1]) for p in Path(checkpoints).glob('checkpoint_[0-9]*.pkl')]
                        self.state['update_count'] = max(numbers, default=0)
                    if progress_file and Path(progress_file).exists():
                        progress = json.loads(Path(progress_file).read_text())
                        self.state['completed_final_trials'] = completed_before + progress['completed_pairs']
                        self.state['completed_condition_trials'] = progress['completed_pairs']
                    self.save()
                    time.sleep(min(2., self.remaining()))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                self.save(child_pid=None)
        if child.returncode:
            raise RuntimeError(f'{key} exited {child.returncode}; see {log}')
        if any(not Path(path).is_file() for path in outputs):
            if self.remaining() < 10:
                raise BudgetExceeded('Training stopped at an update boundary near the deadline')
            raise RuntimeError(f'{key} did not produce all required outputs')
        self.state['tasks'][key].update(completed=True, outputs={str(p): digest(p) for p in outputs})
        self.save()


def eval_command(reference, directory, trials, seed, checkpoint=None, condition='exact_state', nominal=False):
    position, velocity, delay = CONDITIONS[condition]
    suite = 'exact_state' if condition == 'exact_state' else 'noisy_state'
    command = [str(ROOT / '.venv/bin/a2rl-drone-evaluate'), '--reference-path', str(reference),
               '--device', 'gpu', '--num-envs', str(min(4, trials)), '--trials', str(trials),
               '--seed', str(seed), '--spawn-position-std', '0' if nominal else '.02',
               '--spawn-velocity-std', '0' if nominal else '.05', '--max-episode-time', '120',
               '--output-dir', str(directory), '--suite', suite,
               '--position-noise-std', str(position), '--velocity-noise-std', str(velocity),
               '--sensor-delay', str(delay)]
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    return command, suite


def evaluate(runner, key, reference, directory, trials, seed, checkpoint=None, condition='exact_state', nominal=False):
    command, suite = eval_command(reference, directory, trials, seed, checkpoint, condition, nominal)
    outputs = [directory / 'report.json', directory / f'{suite}_baseline.npz']
    if checkpoint:
        outputs.append(directory / f'{suite}_learned.npz')
    runner.run(key, command, outputs, inputs=[reference] + ([checkpoint] if checkpoint else []))
    import numpy as np
    with np.load(outputs[1], allow_pickle=False) as data:
        baseline = dict(data)
    learned = None
    if checkpoint:
        with np.load(outputs[2], allow_pickle=False) as data:
            learned = dict(data)
    return baseline, learned


def combine(results):
    import numpy as np
    keys = results[0].keys()
    combined = {}
    for key in keys:
        if key == 'seed':
            continue
        if key == 'wall_seconds' or key.startswith('diagnostic_'):
            combined[key] = sum(float(r[key]) for r in results)
        else:
            combined[key] = np.concatenate([np.atleast_1d(r[key]) for r in results])
    return combined


def pipeline(runner, previous_path, *, caps=None, train_before_qualification=False, motor_command_limit=1.0, screen_every=32, recovery_only=False):
    import numpy as np
    from a2rl_drone_training.hierarchical.evaluation import summarize
    if screen_every not in (16, 32):
        raise ValueError("Screening interval must be 16 or 32 updates")
    previous = reference_metrics(previous_path)
    checkpoint = None
    for stage_index, cap in enumerate(CAPS if caps is None else caps):
        name = 'uncapped' if cap is None else f'{cap:g}mps'
        directory = runner.directory / name
        directory.mkdir(exist_ok=True)
        reference = directory / 'reference.npz'
        runner.save(speed_cap_m_s=cap, update_count=0)
        runner.run(name + '/planning', [str(ROOT / '.venv/bin/a2rl-drone-plan'),
                   '--max-speed', 'none' if cap is None else str(cap), '--warm-start', str(previous_path),
                   '--motor-command-limit', str(motor_command_limit),
                   '--nodes-per-segment', '24', '--max-refinements', '1', '--max-iterations', '2000',
                   '--output', str(reference)], [reference], inputs=[previous_path], cpu=True)
        metrics = reference_metrics(reference)
        require_faster_reference(metrics, previous, cap)
        runner.state['stages'].setdefault(name, {})['reference'] = metrics
        if train_before_qualification:
            runner.state['stages'][name]['pretraining_qualification'] = 'deferred_by_user'
            runner.save('training_reference_ready', reference=metrics,
                        qualification_note='Training allowed before MPC qualification; promotion checks remain mandatory')
        else:
            for label, trials, nominal in [('nominal', 1, True), ('perturbed', 4, False)]:
                baseline, _ = evaluate(runner, name + '/' + label, reference, directory / label,
                                       trials, 929 + stage_index * 100, nominal=nominal)
                require_clean_flights(baseline, trials)
        checkpoints = directory / 'checkpoints'
        passed = False
        for block, target in enumerate(range(screen_every, 129, screen_every), 1):
            checkpoint = checkpoints / f'checkpoint_{target:06d}.pkl'
            command = training_command(reference, checkpoints, target)
            if recovery_only:
                command += ['--residual-recovery-only', '--residual-recovery-critical-areas', '--residual-position-limit', '.1',
                            '--residual-speed-fraction', '.15', '--residual-speedup-fraction', '0']
                # Expose recoverable deviations during training; held-out tests remain unchanged.
                for flag, value in (('--spawn-position-std', '.08'), ('--spawn-velocity-std', '.15')):
                    command[command.index(flag) + 1] = value
            latest = checkpoints / 'checkpoint_latest.pkl'
            if latest.exists():
                updates = checkpoint_updates(latest, metrics)
                if not checkpoint.exists() and updates >= target:
                    raise ValueError('Expected numbered checkpoint is missing')
                command += ['--restore-checkpoint', str(latest)]
            runner.run(f'{name}/train{block}', command, [checkpoint], [reference], checkpoints=checkpoints)
            if checkpoint_updates(checkpoint, metrics) != target:
                raise ValueError('Numbered checkpoint update does not match its training block')
            runner.save(update_count=target)
            comparisons = {}
            try:
                for condition in ('exact_state', 'position_1cm'):
                    baseline, learned = evaluate(runner, f'{name}/screen{block}/{condition}', reference,
                        directory / f'screen{block}' / condition, 8,
                        1000000 + stage_index * 10000 + block * 100, checkpoint, condition)
                    require_pair(baseline, learned, 8)
                    comparisons[condition] = dict(baseline=summarize(baseline), learned=summarize(learned))
            except QualificationFailure as exc:
                runner.save(screen_failure=str(exc), block=block)
                runner.state['stages'][name][f'block{block}_failure'] = str(exc)
                continue
            runner.state['stages'][name].update(selected_checkpoint=str(checkpoint), comparison=comparisons)
            runner.save('stage_passed', block=block)
            passed = True
            break
        if not passed:
            raise RuntimeError(f'{name}: all {128 // screen_every} training blocks failed qualification')
        previous, previous_path = metrics, reference
    final_validate(runner, previous_path, checkpoint)


def final_validate(runner, previous_path, checkpoint):
    from a2rl_drone_training.hierarchical.evaluation import summarize
    if reference_metrics(previous_path)['planner_cap_m_s'] is not None:
        raise ValueError('Final reference is capped')
    trials = runner.state["spec"].get("final_trials", 100)
    if trials not in (20, 100):
        raise ValueError("Validation requires 20 or 100 pairs per condition")
    development = trials == 20
    reports = {}
    completed = 0
    for condition_index, condition in enumerate(CONDITIONS):
        baselines, learned_results = [], []
        from evaluate_seeded_condition import load_pair
        condition_dir = runner.directory / 'final' / condition
        runner.run(f'final/{condition}', [sys.executable, str(ROOT / 'scripts/evaluate_seeded_condition.py'),
            '--reference-path', str(previous_path), '--checkpoint', str(checkpoint),
            '--output-dir', str(condition_dir), '--condition', condition,
            '--seed-start', str(10000000 + condition_index * 1000), '--trials', str(trials)],
            [condition_dir / 'completed.json'], inputs=[previous_path, checkpoint],
            progress_file=condition_dir / 'progress.json', completed_before=condition_index * trials)
        for index in range(trials):
            seed = 10000000 + condition_index * 1000 + index
            signature = dict(reference=digest(previous_path), checkpoint=digest(checkpoint),
                             condition=condition, settings=list(CONDITIONS[condition]),
                             implementation='persistent-pairs-v1', seed=seed)
            baseline, learned = load_pair(condition_dir / f'seed{seed}', signature)
            require_clean_flights(baseline, 1)
            require_clean_flights(learned, 1)
            baselines.append(baseline); learned_results.append(learned)
            completed += 1
        runner.save('final_validation', condition=condition, completed_final_trials=completed,
                    completed_condition_trials=trials)
        baseline, learned = combine(baselines), combine(learned_results)
        require_pair(baseline, learned, trials)
        reports[condition] = dict(baseline=summarize(baseline), learned=summarize(learned))
        atomic_json(runner.directory / 'final_report.json', dict(accepted=False, development_validation_passed=False, trials_per_condition=trials, conditions=reports))
    if runner.remaining() <= 0:
        raise BudgetExceeded('Final validation exceeded execution budget')
    atomic_json(runner.directory / 'final_report.json', dict(accepted=not development, development_validation_passed=development,
                trials_per_condition=trials, conditions=reports,
                label='development validation passed' if development else 'uncapped simulation accepted',
                checkpoint=str(checkpoint), reference=str(previous_path)))
    runner.save('development_validation_passed' if development else 'uncapped_simulation_accepted', checkpoint=str(checkpoint), reason=None)


def build_spec(previous_reference):
    spec = dict(version=2, caps=CAPS, initial_reference=digest(previous_reference),
                updates_per_block=32, max_blocks=4, conditions=CONDITIONS,
                screening_trials=8, final_trials=100, max_lap_ratio=1.02,
                source={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src').rglob('*.py'))},
                runner=digest(Path(__file__)),
                staging_helper=digest(ROOT / 'scripts/run_staged_validation.py'),
                dependencies=digest(ROOT / 'pyproject.toml'),
                evaluation_worker=digest(ROOT / 'scripts/evaluate_seeded_condition.py'))
    # Normalize tuples exactly as JSON will restore them.
    spec = json.loads(json.dumps(spec))
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/uncapped_curriculum'))
    parser.add_argument('--previous-reference', type=Path, default=Path('artifacts/arena_reference.npz'))
    parser.add_argument('--budget-hours', type=float, default=24.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validation-trials', type=int, choices=(20,100), default=100)
    from a2rl_drone_training.hierarchical.cli import parse_speed_cap
    parser.add_argument('--caps', nargs='+', type=parse_speed_cap, default=CAPS)
    parser.add_argument('--recovery-only', action='store_true')
    parser.add_argument('--screen-every', type=int, choices=(16,32), default=32,
                        help='Evaluate promotion every N updates, up to 128 per reference.')
    parser.add_argument('--train-before-qualification', action='store_true',
                        help='Begin PPO before MPC qualification; keep promotion and final checks mandatory.')
    parser.add_argument('--motor-command-limit', type=float, default=1.0)
    parser.add_argument('--carry-budget-from', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.motor_command_limit) or not 0 < args.motor_command_limit <= 1:
        parser.error('--motor-command-limit must be finite and in (0, 1]')
    if args.caps[-1] is not None or any(c is None for c in args.caps[:-1]):
        parser.error('--caps must end with a single none stage')
    if any(a >= b for a, b in zip(args.caps[:-2], args.caps[1:-1])):
        parser.error('Numeric speed caps must increase')
    os.chdir(ROOT)
    import fcntl
    # Shared lock across output directories, released automatically on process exit.
    with (ROOT / 'artifacts/.speed_curriculum.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if int(proc.name) == os.getpid() or (proc / 'cwd').resolve() != ROOT:
                    continue
                command = (proc / 'cmdline').read_bytes().split(b'\0')
                if any(Path(arg.decode()).name in ('audit_sensor_robustness.py', 'run_staged_validation.py',
                       'run_speed_curriculum.py', 'run_residual_curriculum.py', 'evaluate_seeded_condition.py', 'benchmark_rollout_throughput.py', 'a2rl-drone-plan', 'a2rl-drone-evaluate', 'a2rl-drone-train')
                       or arg == b'a2rl_drone_training.train' for arg in command):
                    raise RuntimeError(f'Another simulation pipeline is active: PID {proc.name}')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        spec = build_spec(args.previous_reference)
        carried = None
        if args.carry_budget_from:
            carried = json.loads(args.carry_budget_from.read_text())
            if carried.get('process_alive'):
                raise ValueError('Previous runner must be stopped before carrying its budget')
        spec.update(final_trials=args.validation_trials, recovery_only=args.recovery_only, caps=args.caps, updates_per_block=args.screen_every, max_blocks=128 // args.screen_every,
                    train_before_qualification=args.train_before_qualification,
                    motor_command_limit=args.motor_command_limit,
                    prior_run=digest(args.carry_budget_from) if carried else None,
                    initial_elapsed_seconds=carried['elapsed_seconds'] if carried else 0.)
        runner = Runner(args.output_dir, spec, args.budget_hours, args.resume)
        if carried and not args.resume:
            runner.used = carried['elapsed_seconds']
            runner.save('initializing', budget_carried_from=str(args.carry_budget_from))
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        try:
            pipeline(runner, args.previous_reference, caps=args.caps,
                     train_before_qualification=args.train_before_qualification,
                     motor_command_limit=args.motor_command_limit, screen_every=args.screen_every, recovery_only=args.recovery_only)
        except BudgetExceeded as exc:
            runner.save('budget_exhausted', reason=str(exc))
        except BaseException as exc:
            runner.save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'blocked', reason=str(exc))
            raise
        finally:
            runner.save(process_alive=False, child_pid=None)


if __name__ == '__main__':
    main()
