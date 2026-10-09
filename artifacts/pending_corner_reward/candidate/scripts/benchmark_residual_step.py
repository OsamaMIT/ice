"""Compare eager/compiled transition paths on the same deterministic flight."""
import argparse
from pathlib import Path
from dataclasses import replace
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/residual_step_benchmark.json"))
    parser.add_argument("--reference-path", type=Path, default=Path("artifacts/arena_reference.npz"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    from a2rl_drone_training.runtime import configure_runtime
    configure_runtime(args.device, 0.5)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from a2rl_drone_training.config import TrainingConfig, ResidualConfig
    from a2rl_drone_training.hierarchical.environment import ResidualMPCEnv
    cfg = TrainingConfig(controller="residual_mpc", residual=ResidualConfig(reference_path=args.reference_path))
    cfg = replace(cfg, env=replace(cfg.env, device=args.device, num_envs=1, auto_reset=False, max_episode_time_s=120))
    env = ResidualMPCEnv(cfg)
    results, trajectories = {}, {}
    counters = [(env, "mpc_wall_seconds"), (env.base, "physics_wall_seconds"),
                (env.base, "gate_reward_wall_seconds"), (env.base, "observation_wall_seconds"),
                (env, "observation_wall_seconds"), (env, "reference_wall_seconds"),
                (env, "bookkeeping_wall_seconds")]
    try:
        env.base.profile_timing = True
        for compiled in (False, True):
            name = "compiled" if compiled else "eager"
            env.base.compile_step_kernels = compiled
            print(f"Warming {name} path...", flush=True)
            env.reset(seed=123)
            jax.block_until_ready(env.step(jnp.zeros((1, 4))))
            env.reset(seed=123)
            for obj, key in counters:
                setattr(obj, key, 0.0)
            jax.block_until_ready(env.base.sim.data.states)
            started = time.perf_counter()
            trace = []
            for step in range(args.steps):
                output = env.step(jnp.zeros((1, 4)))
                jax.block_until_ready(output)
                obs, reward, term, trunc, info = output
                trace.append(np.r_[np.asarray(env.model.pack(env.base.sim.data.states)).ravel(),
                    np.asarray(obs).ravel(), np.asarray(reward).ravel(), np.asarray(term).ravel(),
                    np.asarray(trunc).ravel(), np.asarray(info["gate_pass_events"]).ravel()])
                print(f"{name}: {step + 1}/{args.steps}", flush=True)
                if bool(jnp.any(term | trunc)):
                    break
            wall = time.perf_counter() - started
            trajectories[name] = np.asarray(trace)
            results[name] = dict(steps=len(trace), wall_seconds=wall,
                seconds_per_policy_step=wall / len(trace), mpc_seconds=env.mpc_wall_seconds,
                physics_seconds=env.base.physics_wall_seconds,
                gate_reward_seconds=env.base.gate_reward_wall_seconds,
                observation_seconds=env.base.observation_wall_seconds + env.observation_wall_seconds,
                reference_seconds=env.reference_wall_seconds, bookkeeping_seconds=env.bookkeeping_wall_seconds)
        np.testing.assert_allclose(trajectories["compiled"], trajectories["eager"], atol=2e-4, rtol=2e-4)
        results["equivalent"] = True
        results["maximum_absolute_difference"] = float(np.max(np.abs(trajectories["compiled"] - trajectories["eager"])))
        results["speedup"] = results["eager"]["seconds_per_policy_step"] / results["compiled"]["seconds_per_policy_step"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
