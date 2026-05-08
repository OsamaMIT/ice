from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed-size state representation consumed by the policy."""

    gate_context: int = 3
    core_dim: int = 21
    gate_feature_dim: int = 10
    max_gate_distance: float = 25.0
    camera_fov_x_rad: float = 1.7453292519943295  # 100 deg
    camera_fov_y_rad: float = 1.3089969389957472  # 75 deg
    pnp_dropout_prob: float = 0.02
    additive_noise_std: float = 0.01

    @property
    def dim(self) -> int:
        return self.core_dim + self.gate_context * self.gate_feature_dim


@dataclass(frozen=True)
class RacingEnvConfig:
    """Crazyflow-backed racing task configuration."""

    num_envs: int = 64
    sim_hz: int = 500
    control_hz: int = 100
    max_episode_time_s: float = 24.0
    laps: int = 1
    device: str = "cpu"
    physics: Literal["first_principles", "so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"] = (
        "so_rpy_rotor_drag"
    )
    drone_model: str = "cf2x_L250"
    gate_radius_m: float = 0.55
    gate_pass_bonus: float = 6.0
    lap_finish_bonus: float = 25.0
    crash_penalty: float = -5.0
    miss_penalty: float = -2.0
    time_penalty: float = 0.0
    plane_progress_reward: float = 1.8
    distance_progress_reward: float = 0.8
    radial_progress_reward: float = 0.25
    centerline_reward: float = 0.06
    crossing_center_reward: float = 0.08
    path_alignment_reward: float = 0.02
    early_gate_bonus: float = 0.0
    late_gate_reward_base: float = 1.0
    late_gate_reward_cap: float = 4.0
    lookahead_progress_reward: float = 0.35
    centered_crossing_bonus: float = 2.0
    stall_penalty: float = -0.02
    stall_progress_threshold_m: float = 0.005
    stall_patience_steps: int = 25
    action_penalty: float = 0.008
    max_roll_rad: float = 0.7853981633974483
    max_pitch_rad: float = 0.7853981633974483
    max_yaw_rate_rad_s: float = 3.141592653589793
    thrust_min_n: float | None = None
    thrust_hover_n: float | None = None
    thrust_max_n: float | None = None
    spawn_pos_noise_m: float = 0.12
    spawn_vel_noise_m_s: float = 0.15
    gate_miss_depth_m: float = 1.0
    bounds_xyz_m: tuple[float, float, float] = (20.0, 8.0, 4.0)
    auto_reset: bool = True
    curriculum_enabled: bool = True
    gate_window_scale: float = 1.0
    curriculum_gate_window_start: float = 1.8
    curriculum_gate_window_end: float = 1.0
    curriculum_end_fraction: float = 0.75
    start_gate_curriculum_enabled: bool = True
    start_gate_curriculum_end_fraction: float = 0.85
    mixed_start_random_fraction: float = 0.4
    mixed_start_random_min_gate: int = 2
    mixed_start_random_max_gate: int | None = None
    mixed_start_focus_gate: int | None = None
    mixed_start_focus_fraction: float = 0.0
    racing_line_spawn_min_distance_m: float = 1.0
    racing_line_spawn_max_distance_m: float = 3.0
    racing_line_spawn_speed_m_s: float = 2.0
    spawn_distance_m: float = 1.3

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_time_s * self.control_hz)


@dataclass(frozen=True)
class NetworkConfig:
    action_dim: int = 4
    state_hidden: tuple[int, ...] = (128, 128)
    state_latent_dim: int = 128
    gate_hidden: tuple[int, ...] = (64, 64)
    gate_latent_dim: int = 64
    fusion_hidden: tuple[int, ...] = (256, 128)
    critic_hidden: tuple[int, ...] = (256, 256)
    initial_log_std: float = -0.5


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 1_000_000
    horizon: int = 128
    minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    seed: int = 7
    log_interval: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    env: RacingEnvConfig = RacingEnvConfig()
    obs: ObservationConfig = ObservationConfig()
    net: NetworkConfig = NetworkConfig()
    ppo: PPOConfig = PPOConfig()
    checkpoint_dir: Path | None = None
