"""Optimize one uncapped reference, then relax RL speed and exploration bounds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from run_speed_curriculum import (ROOT, Runner, BudgetExceeded, QualificationFailure,
    atomic_json, digest, build_spec, reference_metrics, training_command,
    checkpoint_updates, evaluate, require_pair, final_validate)
from run_staged_validation import require_clean_flights

STAGES = [dict(speedup=.10, std_floor=.20, std_ceiling=.35, entropy=.003),
          dict(speedup=.20, std_floor=.35, std_ceiling=.50, entropy=.005),
          dict(speedup=.30, std_floor=.50, std_ceiling=.65, entropy=.008)]


def stage_command(reference, checkpoints, target_updates, stage, restore=None):
    command = training_command(reference, checkpoints, target_updates)
    command[command.index('--schedule-env-steps') + 1] = str(384 * 1024)
    command += ['--residual-speedup-fraction', str(stage['speedup']),
                '--exploration-std-start', str(stage['std_ceiling']),
                '--exploration-std-end', str(stage['std_ceiling']),
                '--exploration-std-floor', str(stage['std_floor']),
                '--entropy-coef', str(stage['entropy']), '--entropy-coef-end', str(stage['entropy'])]
    if restore:
        command += ['--restore-checkpoint', str(restore)]
    return command


def pipeline(runner, motor_command_limit=1.0, warm_start=None):
    import numpy as np
    from a2rl_drone_training.hierarchical.evaluation import summarize
    reference = runner.directory / 'reference.npz'
    seed_args = ['--warm-start', str(warm_start)] if warm_start else ['--curvature-seed']
    runner.run('planning', [str(ROOT / '.venv/bin/a2rl-drone-plan'), '--max-speed', 'none',
        '--motor-command-limit', str(motor_command_limit), *seed_args,
        '--nodes-per-segment', '24', '--max-refinements', '1',
        '--max-iterations', '2000', '--output', str(reference)], [reference],
        inputs=[warm_start] if warm_start else [], cpu=True)
    metrics = reference_metrics(reference)
    if metrics['planner_cap_m_s'] is not None:
        raise ValueError('The shared reference must be uncapped')
    runner.save(reference=metrics, speed_cap_m_s=None)
    for name, count, nominal in [('nominal', 1, True), ('perturbed', 4, False)]:
        baseline, _ = evaluate(runner, name, reference, runner.directory / name, count, 170929, nominal=nominal)
        require_clean_flights(baseline, count)
    selected = None
    for index, stage in enumerate(STAGES):
        name = f'rl_stage{index + 1}'
        folder = runner.directory / name
        folder.mkdir(exist_ok=True)
        start = checkpoint_updates(selected, metrics) if selected else 0
        runner.save(rl_stage=index + 1, rl_speedup_fraction=stage['speedup'],
                    exploration_std_floor=stage['std_floor'], exploration_std_ceiling=stage['std_ceiling'])
        checkpoint_dir = folder / 'checkpoints'
        passed = False
        for block in range(1, 5):
            target = start + block * 32
            checkpoint = checkpoint_dir / f'checkpoint_{target:06d}.pkl'
            latest = checkpoint_dir / 'checkpoint_latest.pkl'
            restore = latest if latest.exists() else selected
            if restore:
                current = checkpoint_updates(restore, metrics)
                if current > target and not checkpoint.exists():
                    raise ValueError('Training checkpoint progression is inconsistent')
            runner.run(f'{name}/train{block}', stage_command(reference, checkpoint_dir, target, stage, restore),
                       [checkpoint], inputs=[reference] + ([selected] if selected else []), checkpoints=checkpoint_dir)
            if checkpoint_updates(checkpoint, metrics) != target:
                raise ValueError('Wrong checkpoint update count')
            runner.save(update_count=target, block=block)
            comparisons = {}
            try:
                for condition in ('exact_state', 'position_1cm'):
                    baseline, learned = evaluate(runner, f'{name}/screen{block}/{condition}', reference,
                        folder / f'screen{block}' / condition, 8,
                        2000000 + index * 10000 + block * 100, checkpoint, condition)
                    require_pair(baseline, learned, 8)
                    comparisons[condition] = dict(baseline=summarize(baseline), learned=summarize(learned))
            except QualificationFailure as exc:
                runner.state['stages'].setdefault(name, {})[f'block{block}_failure'] = str(exc)
                runner.save(screen_failure=str(exc))
                continue
            selected = checkpoint
            runner.state['stages'][name] = dict(settings=stage, selected_checkpoint=str(selected), comparisons=comparisons)
            runner.save('rl_stage_passed')
            passed = True
            break
        if not passed:
            raise RuntimeError(f'{name}: four training blocks failed; speed and exploration will not increase')
    final_validate(runner, reference, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/residual_curriculum'))
    parser.add_argument('--budget-hours', type=float, default=24.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--motor-command-limit', type=float, default=1.0)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--carry-budget-from', type=Path, default=Path('artifacts/uncapped_curriculum/status.json'))
    args = parser.parse_args()
    os.chdir(ROOT)
    import math
    if not math.isfinite(args.motor_command_limit) or not 0 < args.motor_command_limit <= 1:
        parser.error('--motor-command-limit must be finite and in (0, 1]')
    with (ROOT / 'artifacts/.speed_curriculum.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if int(proc.name) == os.getpid() or (proc / 'cwd').resolve() != ROOT:
                    continue
                parts = (proc / 'cmdline').read_bytes().split(b'\0')
                if any(Path(p.decode()).name in ('run_speed_curriculum.py','run_residual_curriculum.py',
                    'a2rl-drone-plan','a2rl-drone-evaluate','evaluate_seeded_condition.py',
                    'audit_sensor_robustness.py','run_staged_validation.py','benchmark_rollout_throughput.py')
                    or p == b'a2rl_drone_training.train' for p in parts):
                    raise RuntimeError(f'Another simulation job is active: {proc.name}')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        old = json.loads(args.carry_budget_from.read_text())
        if old.get('process_alive'):
            raise ValueError('Previous run must be stopped before transferring its budget')
        spec = build_spec(Path('artifacts/arena_reference.npz'))
        spec.update(strategy='uncapped_reference_progressive_residual', caps=[None], rl_stages=STAGES,
                    residual_runner=digest(Path(__file__)), initial_elapsed_seconds=old['elapsed_seconds'],
                    prior_run=digest(args.carry_budget_from),
                    motor_command_limit=args.motor_command_limit,
                    warm_start=str(args.warm_start) if args.warm_start else None,
                    warm_start_sha=digest(args.warm_start) if args.warm_start else None)
        spec = json.loads(json.dumps(spec))
        runner = Runner(args.output_dir, spec, args.budget_hours, args.resume)
        if not args.resume:
            runner.used = old['elapsed_seconds']
            runner.save('initializing', budget_carried_from=str(args.carry_budget_from))
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        try:
            pipeline(runner, args.motor_command_limit, args.warm_start)
        except BudgetExceeded as exc:
            runner.save('budget_exhausted', reason=str(exc))
        except BaseException as exc:
            runner.save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'blocked', reason=str(exc))
            raise
        finally:
            runner.save(process_alive=False, child_pid=None)


if __name__ == '__main__':
    main()
