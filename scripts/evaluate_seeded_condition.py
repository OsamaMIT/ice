"""Persistent seeded paired evaluation; reuse compiled environments and policy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import time

from run_speed_curriculum import CONDITIONS, atomic_json, digest, combine
from run_staged_validation import require_clean_flights


def load_pair(directory, signature):
    import numpy as np
    marker = directory / 'completed.json'
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    if saved['signature'] != signature:
        raise ValueError('Pair configuration changed')
    results = []
    for name in ('baseline', 'learned'):
        path = directory / (name + '.npz')
        if digest(path) != saved['hashes'][name]:
            raise ValueError('Completed pair artifact changed')
        with np.load(path, allow_pickle=False) as data:
            results.append(dict(data))
    return results


def save_pair(directory, signature, baseline, learned):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, result in (('baseline', baseline), ('learned', learned)):
        path = directory / (name + '.npz')
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **result)
        temporary.replace(path)
        hashes[name] = digest(path)
    atomic_json(directory / 'completed.json', dict(signature=signature, hashes=hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--condition', choices=list(CONDITIONS), required=True)
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('Trials must be positive')
    from a2rl_drone_training.runtime import configure_runtime
    configure_runtime('gpu', .5)
    from a2rl_drone_training.course import course_by_name
    from a2rl_drone_training.trainer import PPOTrainer, _deterministic_action
    from a2rl_drone_training.hierarchical.environment import residual_training_config
    from a2rl_drone_training.hierarchical.evaluation import EvaluationPool, run_trials, summarize
    with args.checkpoint.open('rb') as stream:
        config = pickle.load(stream)['config']
    position, velocity, delay = CONDITIONS[args.condition]
    config = residual_training_config(replace(config,
        env=replace(config.env, device='gpu', num_envs=1, max_episode_time_s=120),
        residual=replace(config.residual, reference_path=args.reference_path,
            spawn_position_std_m=.02, spawn_velocity_std_m_s=.05, estimation_noise_std=0.,
            position_noise_std_m=position, velocity_noise_std_m_s=velocity, sensor_delay_s=delay,
            sensor_dropout_prob=0., position_bias_m=(0.,0.,0.), velocity_bias_m_s=(0.,0.,0.))))
    course = course_by_name('arena_38m_stacked')
    trainer = None
    pool = EvaluationPool()
    baselines, learned_results = [], []
    signature_base = dict(reference=digest(args.reference_path), checkpoint=digest(args.checkpoint),
                          condition=args.condition, settings=list(CONDITIONS[args.condition]),
                          implementation='persistent-pairs-v1')
    try:
        for index in range(args.trials):
            seed = args.seed_start + index
            directory = args.output_dir / f'seed{seed}'
            signature = dict(signature_base, seed=seed)
            pair = load_pair(directory, signature)
            if pair is None:
                if trainer is None:
                    trainer = PPOTrainer(config, course=course, restore_checkpoint=args.checkpoint)
                policy = lambda obs: _deterministic_action(trainer.state.actor_params, trainer._normalize(obs), config.obs)
                baseline = run_trials(config, course, trials=1, seed=seed, record_trace=True,
                    pool=pool, diagnostics_dir=directory / 'baseline_live')
                learned = run_trials(config, course, trials=1, seed=seed, record_trace=True,
                    pool=pool, policy=policy, diagnostics_dir=directory / 'learned_live')
                save_pair(directory, signature, baseline, learned)
            else:
                baseline, learned = pair
            require_clean_flights(baseline, 1)
            require_clean_flights(learned, 1)
            baselines.append(baseline); learned_results.append(learned)
            atomic_json(args.output_dir / 'progress.json', dict(completed_pairs=index + 1,
                        trials=args.trials, condition=args.condition, seed=seed, updated_at=time.time()))
            print(f'{args.condition}: {index + 1}/{args.trials} pairs complete', flush=True)
        atomic_json(args.output_dir / 'completed.json', dict(signature=signature_base,
            trials=args.trials, seed_start=args.seed_start,
            baseline=summarize(combine(baselines)), learned=summarize(combine(learned_results))))
    finally:
        pool.close()
        if trainer is not None:
            trainer.close()


if __name__ == '__main__':
    main()
