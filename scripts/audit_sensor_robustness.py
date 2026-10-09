"""One-factor MPC robustness sweep; no training and no real-hardware claims."""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import time

CASES = {
    "clean": {},
    "position_1cm": dict(position_noise_std_m=.01),
    "position_3cm": dict(position_noise_std_m=.03),
    "velocity_2cms": dict(velocity_noise_std_m_s=.02),
    "velocity_5cms": dict(velocity_noise_std_m_s=.05),
    "delay_10ms": dict(sensor_delay_s=.01),
    "delay_30ms": dict(sensor_delay_s=.03),
    "dropout_5pct": dict(sensor_dropout_prob=.05),
    "position_bias_2cm": dict(position_bias_m=(.02, 0., 0.)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-path', type=Path, default=Path('artifacts/arena_reference.npz'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/sensor_audit'))
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES)[:7])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('Trials must be positive')
    from a2rl_drone_training.runtime import configure_runtime
    configure_runtime(args.device, .5 if args.device == 'gpu' else None)
    import numpy as np
    from a2rl_drone_training.config import TrainingConfig, ResidualConfig
    from a2rl_drone_training.course import course_by_name
    from a2rl_drone_training.hierarchical.evaluation import run_trials, summarize
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = TrainingConfig(controller='residual_mpc', residual=ResidualConfig(reference_path=args.reference_path,
        position_noise_std_m=0., velocity_noise_std_m_s=0.))
    config = replace(config, env=replace(config.env, num_envs=min(2, args.trials), device=args.device, max_episode_time_s=120))
    reference_hash = hashlib.sha256(args.reference_path.read_bytes()).hexdigest()
    report = dict(reference_sha256=reference_hash, seed=args.seed, trials_per_case=args.trials,
        scope='Synthetic position/velocity estimator errors; known map/gate index and exact attitude, rates, acceleration and rotor states. No opponents or wind.', cases={})
    def save(status, case=None, error=None):
        report.update(status=status, current_case=case, error=error, updated_at=time.time())
        path = args.output_dir / 'report.tmp'
        path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str))
        path.replace(args.output_dir / 'report.json')
    try:
        for name in dict.fromkeys(['clean'] + args.cases):
            if hashlib.sha256(args.reference_path.read_bytes()).hexdigest() != reference_hash:
                raise ValueError('Reference changed during audit')
            save('running', name)
            print('Sensor audit: ' + name, flush=True)
            suite = replace(config, residual=replace(config.residual, **CASES[name]))
            result = run_trials(suite, course_by_name('arena_38m_stacked'), trials=args.trials,
                seed=args.seed, record_trace=True, diagnostics_dir=args.output_dir / (name + '_live'))
            np.savez_compressed(args.output_dir / (name + '.npz'), **result)
            report['cases'][name] = dict(settings=asdict(suite.residual), summary=summarize(result),
                gates_passed=np.asarray(result['gate_passes']).sum(axis=1).tolist())
            save('running', name)
            if name == 'clean' and (not np.all(result['success']) or np.any(result['collision']) or np.any(result['fallbacks'])):
                raise RuntimeError('Clean baseline failed; stop before interpreting disturbance sensitivity')
            # Static-object JIT caches otherwise retain each closed environment.
            import jax
            jax.clear_caches()
        save('completed')
    except BaseException as exc:
        save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'blocked', report['current_case'], str(exc))
        raise


if __name__ == '__main__':
    main()
