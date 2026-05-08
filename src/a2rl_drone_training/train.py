from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from a2rl_drone_training.config import (
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RacingEnvConfig,
    TrainingConfig,
)
from a2rl_drone_training.course import course_by_name
from a2rl_drone_training.trainer import PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an A2RL-style FPV racing policy.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-env-steps", type=int, default=1_000_000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--physics",
        type=str,
        default="so_rpy_rotor_drag",
        choices=["first_principles", "so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"],
    )
    parser.add_argument("--gate-context", type=int, default=3)
    parser.add_argument("--control-hz", type=int, default=100)
    parser.add_argument("--sim-hz", type=int, default=500)
    parser.add_argument("--max-episode-time", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--gate-window-start", type=float, default=1.8)
    parser.add_argument("--gate-window-end", type=float, default=1.0)
    parser.add_argument("--curriculum-end-fraction", type=float, default=0.75)
    parser.add_argument("--no-start-gate-curriculum", action="store_true")
    parser.add_argument("--start-gate-curriculum-end-fraction", type=float, default=0.85)
    parser.add_argument("--mixed-start-random-fraction", type=float, default=0.4)
    parser.add_argument("--mixed-start-random-min-gate", type=int, default=2)
    parser.add_argument("--mixed-start-random-max-gate", type=int, default=None)
    parser.add_argument("--mixed-start-focus-gate", type=int, default=None)
    parser.add_argument("--mixed-start-focus-fraction", type=float, default=0.0)
    parser.add_argument("--racing-line-spawn-min-distance", type=float, default=1.0)
    parser.add_argument("--racing-line-spawn-max-distance", type=float, default=3.0)
    parser.add_argument("--racing-line-spawn-speed", type=float, default=2.0)
    parser.add_argument("--spawn-distance", type=float, default=1.3)
    parser.add_argument("--gate-pass-bonus", type=float, default=6.0)
    parser.add_argument("--lap-finish-bonus", type=float, default=25.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--early-gate-bonus", type=float, default=0.0)
    parser.add_argument("--late-gate-reward-base", type=float, default=1.0)
    parser.add_argument("--late-gate-reward-cap", type=float, default=4.0)
    parser.add_argument("--lookahead-progress-reward", type=float, default=0.35)
    parser.add_argument("--centered-crossing-bonus", type=float, default=2.0)
    parser.add_argument("--path-alignment-reward", type=float, default=0.02)
    parser.add_argument("--stall-penalty", type=float, default=-0.02)
    parser.add_argument(
        "--course",
        type=str,
        default="arena_38m_stacked",
        choices=["arena_38m_stacked", "compact_slalom"],
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    obs = replace(ObservationConfig(), gate_context=args.gate_context)
    env = replace(
        RacingEnvConfig(),
        num_envs=args.num_envs,
        sim_hz=args.sim_hz,
        control_hz=args.control_hz,
        max_episode_time_s=args.max_episode_time,
        device=args.device,
        physics=args.physics,
        curriculum_enabled=not args.no_curriculum,
        curriculum_gate_window_start=args.gate_window_start,
        curriculum_gate_window_end=args.gate_window_end,
        curriculum_end_fraction=args.curriculum_end_fraction,
        start_gate_curriculum_enabled=not args.no_start_gate_curriculum,
        start_gate_curriculum_end_fraction=args.start_gate_curriculum_end_fraction,
        mixed_start_random_fraction=args.mixed_start_random_fraction,
        mixed_start_random_min_gate=args.mixed_start_random_min_gate,
        mixed_start_random_max_gate=args.mixed_start_random_max_gate,
        mixed_start_focus_gate=args.mixed_start_focus_gate,
        mixed_start_focus_fraction=args.mixed_start_focus_fraction,
        racing_line_spawn_min_distance_m=args.racing_line_spawn_min_distance,
        racing_line_spawn_max_distance_m=args.racing_line_spawn_max_distance,
        racing_line_spawn_speed_m_s=args.racing_line_spawn_speed,
        spawn_distance_m=args.spawn_distance,
        gate_pass_bonus=args.gate_pass_bonus,
        lap_finish_bonus=args.lap_finish_bonus,
        time_penalty=args.time_penalty,
        early_gate_bonus=args.early_gate_bonus,
        late_gate_reward_base=args.late_gate_reward_base,
        late_gate_reward_cap=args.late_gate_reward_cap,
        lookahead_progress_reward=args.lookahead_progress_reward,
        centered_crossing_bonus=args.centered_crossing_bonus,
        path_alignment_reward=args.path_alignment_reward,
        stall_penalty=args.stall_penalty,
    )
    ppo = replace(
        PPOConfig(),
        total_env_steps=args.total_env_steps,
        horizon=args.horizon,
        minibatches=args.minibatches,
        update_epochs=args.update_epochs,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_coef=args.clip_coef,
        value_clip_coef=args.value_clip_coef,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    config = TrainingConfig(
        env=env,
        obs=obs,
        net=NetworkConfig(),
        ppo=ppo,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = PPOTrainer(
        config,
        course=course_by_name(args.course),
        restore_checkpoint=args.restore_checkpoint,
    )
    trainer.train()


if __name__ == "__main__":
    main()
