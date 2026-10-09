"""Measure warmed rollout throughput; does not change training hyperparameters."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/rollout_throughput.json'))
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()
    from a2rl_drone_training.runtime import configure_runtime
    configure_runtime('gpu', .5)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from a2rl_drone_training.config import TrainingConfig, ResidualConfig
    from a2rl_drone_training.hierarchical.environment import ResidualMPCEnv
    results = {}
    for count in (4, 8, 16):
        cfg = TrainingConfig(controller='residual_mpc', residual=ResidualConfig(reference_path=args.reference_path,
            position_noise_std_m=.01, velocity_noise_std_m_s=0.))
        cfg = replace(cfg, env=replace(cfg.env, device='gpu', num_envs=count, auto_reset=False, max_episode_time_s=120))
        env = ResidualMPCEnv(cfg)
        try:
            times, trajectories = {}, {}
            for profiled in (True, False):
                label = 'profiled' if profiled else 'normal'
                env.base.profile_timing = profiled
                env.sensor_tick = 0
                env.reset(seed=991)
                action = jnp.full((count, 4), .02)
                jax.block_until_ready(env.step(action))
                env.sensor_tick = 0
                env.reset(seed=991)
                jax.block_until_ready(env.observe())
                samples = []
                start = time.perf_counter()
                for _ in range(args.steps):
                    obs, reward, term, trunc, info = env.step(action)
                    jax.block_until_ready((obs, reward, term, trunc, info))
                    samples.append(np.r_[np.asarray(env.model.pack(env.base.sim.data.states)).ravel(),
                                         np.asarray(reward).ravel(), np.asarray(term).ravel(),
                                         np.asarray(trunc).ravel(), np.asarray(info['gate_pass_events']).ravel()])
                times[label] = time.perf_counter() - start
                trajectories[label] = np.asarray(samples)
                print(f'{count} worlds / {label}: {times[label]:.3f}s', flush=True)
            np.testing.assert_allclose(trajectories['normal'], trajectories['profiled'], atol=2e-4, rtol=2e-4)
            memory = jax.devices()[0].memory_stats() or {}
            results[str(count)] = dict(seconds=times, policy_transitions_per_second=count * args.steps / times['normal'],
                profiling_speedup=times['profiled'] / times['normal'], equivalent=True,
                max_abs_difference=float(np.max(np.abs(trajectories['normal'] - trajectories['profiled']))),
                peak_bytes_in_use=memory.get('peak_bytes_in_use'))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))
        finally:
            env.close()
            del env
            jax.clear_caches()
            gc.collect()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
