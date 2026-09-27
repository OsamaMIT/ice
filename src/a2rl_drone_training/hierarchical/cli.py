"""Offline planning and paired evaluation entrypoints."""

import argparse
from dataclasses import replace
import json
from pathlib import Path


def plan_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optimize and validate a gate-aware reference trajectory."
    )
    parser.add_argument(
        "--course",
        default="arena_38m_stacked",
        choices=["arena_38m_stacked", "compact_slalom"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drone-model", default="cf2x_L250")
    parser.add_argument("--vehicle-radius", type=float, default=0.15)
    parser.add_argument("--tracking-margin", type=float, default=0.10)
    parser.add_argument("--frame-depth", type=float, default=0.10)
    parser.add_argument("--nodes-per-segment", type=int, default=24)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--max-refinements", type=int, default=2)
    parser.add_argument("--max-speed", type=float, default=12.0)
    args = parser.parse_args(argv)
    from a2rl_drone_training.config import RacingEnvConfig, PlannerConfig
    from a2rl_drone_training.course import course_by_name
    from a2rl_drone_training.env import CrazyflowRacingEnv
    from .dynamics import DroneModel
    from .planner import optimize_reference

    env = CrazyflowRacingEnv(
        RacingEnvConfig(num_envs=1, drone_model=args.drone_model),
        course=course_by_name(args.course),
    )
    try:
        config = PlannerConfig(
            tracking_margin_m=args.tracking_margin,
            frame_depth_m=args.frame_depth,
            nodes_per_segment=args.nodes_per_segment,
            max_iterations=args.max_iterations,
            max_refinements=args.max_refinements,
            max_speed_m_s=args.max_speed,
        )
        ref = optimize_reference(
            env.course, DroneModel.from_env(env), config, args.vehicle_radius
        )
        ref.save(args.output)
        print(
            json.dumps(
                {
                    "reference": str(args.output),
                    "duration_s": float(ref.time[-1]),
                    "fingerprint": ref.fingerprint,
                    **ref.metadata,
                },
                indent=2,
            )
        )
    finally:
        env.close()


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate zero-offset MPC and optionally a residual RL checkpoint."
    )
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--course",
        default="arena_38m_stacked",
        choices=["arena_38m_stacked", "compact_slalom"],
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-episode-time", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1000123)
    parser.add_argument("--spawn-position-std", type=float, default=0.02)
    parser.add_argument("--spawn-velocity-std", type=float, default=0.05)
    parser.add_argument("--noisy-state-std", type=float, default=0.02)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/residual_evaluation")
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    from a2rl_drone_training.runtime import configure_runtime

    configure_runtime(args.device, None)
    import numpy as np
    import pickle
    from a2rl_drone_training.config import TrainingConfig, ResidualConfig
    from a2rl_drone_training.course import course_by_name
    from a2rl_drone_training.trainer import PPOTrainer, _deterministic_action
    from .environment import residual_training_config
    from .evaluation import run_trials, compare_trials, summarize, plot_comparison
    from .reference import Reference

    config = TrainingConfig(
        controller="residual_mpc",
        residual=ResidualConfig(reference_path=args.reference_path),
    )
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("controller") != "residual_mpc":
            raise ValueError("Evaluation requires a residual MPC checkpoint")
        config = payload["config"]
    reference = Reference.load(args.reference_path)
    config = replace(
        config,
        env=replace(
            config.env,
            device=args.device,
            num_envs=args.num_envs,
            max_episode_time_s=(
                config.env.max_episode_time_s
                if args.max_episode_time is None
                else args.max_episode_time
            ),
        ),
        residual=replace(
            config.residual,
            reference_path=args.reference_path,
            spawn_position_std_m=args.spawn_position_std,
            spawn_velocity_std_m_s=args.spawn_velocity_std,
            estimation_noise_std=0.0,
        ),
    )
    config = residual_training_config(config)
    course = course_by_name(args.course)
    trainer = (
        PPOTrainer(config, course=course, restore_checkpoint=args.checkpoint)
        if args.checkpoint
        else None
    )
    policy = (
        (
            lambda raw: _deterministic_action(
                trainer.state.actor_params, trainer._normalize(raw), config.obs
            )
        )
        if trainer
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    try:
        for name, noise in [
            ("exact_state", 0.0),
            ("noisy_state", args.noisy_state_std),
        ]:
            suite = replace(
                config, residual=replace(config.residual, estimation_noise_std=noise)
            )
            baseline = run_trials(
                suite, course, trials=args.trials, seed=args.seed, record_trace=True
            )
            learned = (
                run_trials(
                    suite,
                    course,
                    trials=args.trials,
                    seed=args.seed,
                    policy=policy,
                    record_trace=True,
                )
                if policy
                else None
            )
            reports[name] = (
                compare_trials(baseline, learned)
                if learned is not None
                else {
                    "baseline": summarize(baseline),
                    "accepted": False,
                    "reason": "No learned checkpoint supplied",
                }
            )
            np.savez_compressed(args.output_dir / f"{name}_baseline.npz", **baseline)
            if learned is not None:
                np.savez_compressed(args.output_dir / f"{name}_learned.npz", **learned)
            if args.plot:
                plot_comparison(
                    reference,
                    course,
                    baseline,
                    learned,
                    args.output_dir / f"{name}.png",
                )
        reports["accepted"] = all(
            reports[s]["accepted"] for s in ("exact_state", "noisy_state")
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(reports, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(reports, indent=2, allow_nan=False))
    finally:
        if trainer:
            trainer.close()
