from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import ObservationConfig, RacingEnvConfig
from a2rl_drone_training.course import GateCourse, arena_38m_stacked_course
from a2rl_drone_training.observations import (
    build_racing_observation,
    quat_to_yaw_xyzw,
    wrap_pi,
    yaw_to_quat_xyzw,
)


class CrazyflowRacingEnv:
    """Vectorized Crazyflow racing environment for PPO rollout collection."""

    def __init__(
        self,
        env_config: RacingEnvConfig = RacingEnvConfig(),
        obs_config: ObservationConfig = ObservationConfig(),
        course: GateCourse | None = None,
    ):
        self.config = env_config
        self.obs_config = obs_config
        self.course = course or arena_38m_stacked_course()
        self.total_gate_passes = self.course.num_gates * self.config.laps
        self.n_substeps = self.config.sim_hz // self.config.control_hz
        if self.n_substeps < 1 or self.config.sim_hz % self.config.control_hz:
            raise ValueError("sim_hz must be a positive integer multiple of control_hz")

        try:
            from crazyflow.control import Control
            from crazyflow.sim import Physics, Sim
        except ImportError as exc:
            raise RuntimeError(
                "Crazyflow is required for the training environment. Install with `pip install -e .`."
            ) from exc

        self.sim = Sim(
            n_worlds=self.config.num_envs,
            n_drones=1,
            drone_model=self.config.drone_model,
            physics=Physics(self.config.physics),
            control=Control.attitude,
            freq=self.config.sim_hz,
            attitude_freq=self.config.control_hz,
            device=self.config.device,
        )
        self.gate_centers = jnp.asarray(self.course.centers, dtype=jnp.float32)
        self.gate_normals = jnp.asarray(self.course.normals, dtype=jnp.float32)
        self.gate_right_axes = jnp.asarray(self.course.right_axes, dtype=jnp.float32)
        self.gate_widths = jnp.asarray(self.course.widths, dtype=jnp.float32)
        self.gate_heights = jnp.asarray(self.course.heights, dtype=jnp.float32)
        self.bounds_min = jnp.asarray(self.course.bounds_min, dtype=jnp.float32)
        self.bounds_max = jnp.asarray(self.course.bounds_max, dtype=jnp.float32)
        (
            self.approach_starts,
            self.approach_dirs,
            self.approach_right_axes,
            self.approach_lengths,
        ) = self._build_racing_line_spawn_geometry()
        self.gate_window_scale = (
            float(self.config.curriculum_gate_window_start)
            if self.config.curriculum_enabled
            else float(self.config.gate_window_scale)
        )
        self.max_reset_gate = self.course.num_gates - 1
        self.rng = jax.random.key(0)
        self.gate_counter = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.reset_gate = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.random_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        self.focused_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        self.elapsed_steps = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.last_action = jnp.zeros((self.config.num_envs, 4), dtype=jnp.float32)
        self.yaw_cmd = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.prev_vel = jnp.zeros((self.config.num_envs, 3), dtype=jnp.float32)
        self.prev_gate_plane = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.stall_steps = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.episode_return = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.episode_length = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.thrust_min, self.thrust_hover, self.thrust_max = self._resolve_thrust_bounds()
        self.reset(seed=0)

    @property
    def observation_dim(self) -> int:
        return self.obs_config.dim

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self, *, seed: int | None = None, mask: Array | None = None) -> Array:
        if seed is not None:
            self.rng = jax.random.key(seed)
            self.sim.seed(seed)
        if mask is None:
            mask = jnp.ones((self.config.num_envs,), dtype=jnp.bool_)
        else:
            mask = jnp.asarray(mask, dtype=jnp.bool_)

        self.sim.reset(mask=mask)
        self.rng, reset_key = jax.random.split(self.rng)
        pos, vel, quat, reset_gate, random_start, focused_start = self._sample_initial_state(
            reset_key,
            mask,
        )
        states = self.sim.data.states
        mask3 = mask[:, None, None]
        mask4 = mask[:, None, None]
        new_states = states.replace(
            pos=jnp.where(mask3, pos[:, None, :], states.pos),
            vel=jnp.where(mask3, vel[:, None, :], states.vel),
            quat=jnp.where(mask4, quat[:, None, :], states.quat),
            ang_vel=jnp.where(mask3, jnp.zeros_like(states.ang_vel), states.ang_vel),
        )
        self.sim.data = self.sim.data.replace(states=new_states)

        reset_yaw = quat_to_yaw_xyzw(quat)
        self.yaw_cmd = jnp.where(mask, reset_yaw, self.yaw_cmd)
        self.gate_counter = jnp.where(mask, reset_gate, self.gate_counter)
        self.reset_gate = jnp.where(mask, reset_gate, self.reset_gate)
        self.random_start = jnp.where(mask, random_start, self.random_start)
        self.focused_start = jnp.where(mask, focused_start, self.focused_start)
        self.elapsed_steps = jnp.where(mask, 0.0, self.elapsed_steps)
        self.last_action = jnp.where(mask[:, None], 0.0, self.last_action)
        self.prev_vel = jnp.where(mask[:, None], vel, self.prev_vel)
        plane, _, _ = self._gate_metrics(pos, reset_gate % self.course.num_gates)
        self.prev_gate_plane = jnp.where(mask, plane, self.prev_gate_plane)
        self.stall_steps = jnp.where(mask, 0, self.stall_steps)
        self.episode_return = jnp.where(mask, 0.0, self.episode_return)
        self.episode_length = jnp.where(mask, 0.0, self.episode_length)
        return self.observe()

    def observe(self) -> Array:
        states = self.sim.data.states
        pos = states.pos[:, 0, :]
        quat = states.quat[:, 0, :]
        vel = states.vel[:, 0, :]
        ang_vel = states.ang_vel[:, 0, :]
        acc = self._acceleration_world(vel)
        self.rng, obs_key = jax.random.split(self.rng)
        return build_racing_observation(
            pos_world=pos,
            quat_xyzw=quat,
            vel_world=vel,
            ang_vel_world=ang_vel,
            acc_world=acc,
            gate_centers_world=self.gate_centers,
            gate_normals_world=self.gate_normals,
            gate_counter=self.gate_counter,
            total_gate_passes=self.total_gate_passes,
            last_action=self.last_action,
            config=self.obs_config,
            key=obs_key,
        )

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        pos_before = self.sim.data.states.pos[:, 0, :]
        gate_id = self.gate_counter % self.course.num_gates
        lookahead_gate_id = (gate_id + 1) % self.course.num_gates
        plane_before, radial_before, _ = self._gate_metrics(pos_before, gate_id)
        distance_before = self._gate_distance(pos_before, gate_id)
        lookahead_distance_before = self._gate_distance(pos_before, lookahead_gate_id)

        self.yaw_cmd = wrap_pi(
            self.yaw_cmd + action[:, 3] * self.config.max_yaw_rate_rad_s * self.config.dt
        )
        cmd = self._fpv_action_to_crazyflow_attitude(action)
        self.sim.attitude_control(cmd[:, None, :])
        self.sim.step(self.n_substeps)

        states = self.sim.data.states
        pos = states.pos[:, 0, :]
        vel = states.vel[:, 0, :]
        gate_id = self.gate_counter % self.course.num_gates
        speed = jnp.linalg.norm(vel, axis=-1)
        forward_speed = jnp.sum(vel * self.gate_normals[gate_id], axis=-1)
        time_to_gate = distance_before / jnp.maximum(jnp.abs(forward_speed), 0.1)
        plane_after, radial_after, inside_gate = self._gate_metrics(pos, gate_id)
        distance_after = self._gate_distance(pos, gate_id)
        lookahead_distance_after = self._gate_distance(pos, lookahead_gate_id)
        crossed_plane = (plane_before < 0.0) & (plane_after >= 0.0)
        cross_pos = self._gate_crossing_position(pos_before, pos, plane_before, plane_after)
        _, cross_radial, cross_inside = self._gate_metrics(cross_pos, gate_id)
        passed_gate = crossed_plane & cross_inside
        missed_gate = (plane_after > self.config.gate_miss_depth_m) & ~passed_gate & ~inside_gate
        next_gate_counter = self.gate_counter + passed_gate.astype(jnp.int32)
        finished = next_gate_counter >= self.total_gate_passes

        crashed = pos[:, 2] < 0.05
        out_of_bounds = jnp.any((pos < self.bounds_min) | (pos > self.bounds_max), axis=-1)
        terminated = crashed | out_of_bounds | missed_gate | finished
        truncated = self.elapsed_steps + 1 >= self.config.max_episode_steps
        distance_progress = distance_before - distance_after
        stalled_step = (distance_progress < self.config.stall_progress_threshold_m) & ~passed_gate
        next_stall_steps = jnp.where(stalled_step, self.stall_steps + 1, 0)
        stalled = next_stall_steps >= self.config.stall_patience_steps

        reward = self._reward(
            plane_before=plane_before,
            plane_after=plane_after,
            radial_before=radial_before,
            radial_after=radial_after,
            distance_before=distance_before,
            distance_after=distance_after,
            lookahead_distance_before=lookahead_distance_before,
            lookahead_distance_after=lookahead_distance_after,
            cross_radial=cross_radial,
            pos=pos,
            vel=vel,
            stalled=stalled,
            action=action,
            gate_id=gate_id,
            passed_gate=passed_gate,
            missed_gate=missed_gate,
            crashed=crashed | out_of_bounds,
            finished=finished,
        )
        done = terminated | truncated

        self.gate_counter = jnp.where(finished, self.gate_counter, next_gate_counter)
        self.elapsed_steps = self.elapsed_steps + 1.0
        self.last_action = action
        self.prev_vel = vel
        self.stall_steps = jnp.where(done, 0, next_stall_steps)
        new_gate_id = self.gate_counter % self.course.num_gates
        new_plane, _, _ = self._gate_metrics(pos, new_gate_id)
        self.prev_gate_plane = new_plane

        final_return = self.episode_return + reward
        final_length = self.episode_length + 1.0
        self.episode_return = jnp.where(done, 0.0, final_return)
        self.episode_length = jnp.where(done, 0.0, final_length)

        info = {
            "passed_gate": passed_gate,
            "missed_gate": missed_gate,
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
            "finished": finished,
            "gate_counter": self.gate_counter,
            "gate_id": gate_id,
            "episode_gate_progress": self.gate_counter - self.reset_gate,
            "reset_gate": self.reset_gate,
            "random_start": self.random_start,
            "focused_start": self.focused_start,
            "speed": speed,
            "forward_speed": forward_speed,
            "time_to_gate": time_to_gate,
            "stalled": stalled,
            "max_reset_gate": jnp.full(
                (self.config.num_envs,),
                self.max_reset_gate,
                dtype=jnp.float32,
            ),
            "gate_window_scale": jnp.full(
                (self.config.num_envs,),
                self.gate_window_scale,
                dtype=jnp.float32,
            ),
            "final_return": jnp.where(done, final_return, jnp.nan),
            "final_length": jnp.where(done, final_length, jnp.nan),
        }

        reset_occurred = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        reset_random_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        reset_focused_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        if self.config.auto_reset and bool(jnp.any(done)):
            obs = self.reset(mask=done)
            reset_occurred = done
            reset_random_start = done & self.random_start
            reset_focused_start = done & self.focused_start
        else:
            obs = self.observe()
        info["reset_occurred"] = reset_occurred
        info["reset_random_start"] = reset_random_start
        info["reset_focused_start"] = reset_focused_start
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs: Any) -> Any:
        return self.sim.render(**kwargs)

    def close(self) -> None:
        self.sim.close()

    def set_training_progress(self, progress: float) -> None:
        if not self.config.curriculum_enabled:
            self.gate_window_scale = float(self.config.gate_window_scale)
        else:
            end_fraction = max(float(self.config.curriculum_end_fraction), 1.0e-6)
            mix = float(np.clip(progress / end_fraction, 0.0, 1.0))
            start = float(self.config.curriculum_gate_window_start)
            end = float(self.config.curriculum_gate_window_end)
            self.gate_window_scale = start + mix * (end - start)

        if not self.config.start_gate_curriculum_enabled:
            self.max_reset_gate = 0
            return
        start_end = max(float(self.config.start_gate_curriculum_end_fraction), 1.0e-6)
        start_mix = float(np.clip(progress / start_end, 0.0, 1.0))
        max_gate = self.course.num_gates - 1
        self.max_reset_gate = int(round((1.0 - start_mix) * max_gate))

    def _resolve_thrust_bounds(self) -> tuple[float, float, float]:
        thrust_min = self.config.thrust_min_n
        thrust_max = self.config.thrust_max_n
        if thrust_min is None or thrust_max is None:
            try:
                from drone_controllers.mellinger.params import ForceTorqueParams

                params = ForceTorqueParams.load(self.config.drone_model)
                thrust_min = float(params.thrust_min * 4.0) if thrust_min is None else thrust_min
                thrust_max = float(params.thrust_max * 4.0) if thrust_max is None else thrust_max
            except Exception:
                thrust_min = 0.0 if thrust_min is None else thrust_min
                thrust_max = 0.65 if thrust_max is None else thrust_max

        if self.config.thrust_hover_n is not None:
            thrust_hover = self.config.thrust_hover_n
        else:
            try:
                mass = float(jnp.mean(self.sim.data.params.mass))
                thrust_hover = mass * 9.81
            except Exception:
                thrust_hover = 0.5 * (float(thrust_min) + float(thrust_max))
        thrust_hover = float(np.clip(thrust_hover, thrust_min, thrust_max))
        return float(thrust_min), thrust_hover, float(thrust_max)

    def _fpv_action_to_crazyflow_attitude(self, action: Array) -> Array:
        throttle = action[:, 0]
        roll = action[:, 1] * self.config.max_roll_rad
        pitch = action[:, 2] * self.config.max_pitch_rad
        thrust_up = self.thrust_max - self.thrust_hover
        thrust_down = self.thrust_hover - self.thrust_min
        thrust = self.thrust_hover + jnp.where(
            throttle >= 0.0,
            throttle * thrust_up,
            throttle * thrust_down,
        )
        return jnp.stack([roll, pitch, self.yaw_cmd, thrust], axis=-1)

    def _build_racing_line_spawn_geometry(self) -> tuple[Array, Array, Array, Array]:
        centers = np.asarray(self.course.centers, dtype=np.float32)
        line = np.asarray(self.course.nominal_racing_line, dtype=np.float32)
        if line.shape[0] >= self.course.num_gates + 1:
            starts = line[: self.course.num_gates]
        else:
            starts = np.empty_like(centers)
            starts[0] = centers[0] - self.config.spawn_distance_m * self.course.normals[0]
            starts[1:] = centers[:-1]

        normals = np.asarray(self.course.normals, dtype=np.float32)
        vectors = centers - starts
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        line_dirs = vectors / np.maximum(lengths, 1.0e-6)
        normal_alignment = np.sum(line_dirs * normals, axis=-1, keepdims=True)
        use_line_dir = (lengths > 1.0e-6) & (normal_alignment > 0.25)
        dirs = np.where(use_line_dir, line_dirs, normals)
        lengths = np.where(
            use_line_dir,
            lengths,
            np.maximum(lengths, float(self.config.racing_line_spawn_max_distance_m)),
        )

        up = np.asarray(self.course.world_up, dtype=np.float32)
        right = np.cross(up[None, :], dirs)
        right_lengths = np.linalg.norm(right, axis=-1, keepdims=True)
        fallback_right = np.asarray(self.course.right_axes, dtype=np.float32)
        right = np.where(
            right_lengths > 1.0e-6,
            right / np.maximum(right_lengths, 1.0e-6),
            fallback_right,
        )

        return (
            jnp.asarray(starts, dtype=jnp.float32),
            jnp.asarray(dirs, dtype=jnp.float32),
            jnp.asarray(right, dtype=jnp.float32),
            jnp.asarray(lengths[:, 0], dtype=jnp.float32),
        )

    def _sample_initial_state(
        self,
        key: Array,
        reset_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        (
            gate_key,
            branch_key,
            focus_key,
            distance_key,
            lateral_key,
            vertical_key,
            vel_key,
            yaw_key,
        ) = jax.random.split(key, 8)
        random_fraction = float(np.clip(self.config.mixed_start_random_fraction, 0.0, 1.0))
        if random_fraction > 0.0:
            min_gate = int(np.clip(self.config.mixed_start_random_min_gate, 1, self.course.num_gates))
            max_gate_config = self.config.mixed_start_random_max_gate
            max_gate = self.course.num_gates if max_gate_config is None else int(max_gate_config)
            max_gate = int(np.clip(max_gate, min_gate, self.course.num_gates))
            random_gate = jax.random.randint(
                gate_key,
                (self.config.num_envs,),
                minval=min_gate - 1,
                maxval=max_gate,
                dtype=jnp.int32,
            )
            random_start = _balanced_random_start_mask(
                branch_key,
                reset_mask,
                self.random_start,
                random_fraction,
            )
            focus_fraction = float(
                np.clip(self.config.mixed_start_focus_fraction, 0.0, 1.0)
            )
            focus_gate_config = self.config.mixed_start_focus_gate
            if focus_gate_config is not None and focus_fraction > 0.0:
                current_focused_start = jnp.where(
                    reset_mask,
                    jnp.zeros_like(self.focused_start),
                    self.focused_start,
                )
                focused_start = _balanced_random_start_mask(
                    focus_key,
                    reset_mask & random_start,
                    current_focused_start,
                    random_fraction * focus_fraction,
                )
                focus_gate = int(
                    np.clip(int(focus_gate_config), min_gate, max_gate)
                ) - 1
                random_gate = jnp.where(
                    focused_start,
                    jnp.asarray(focus_gate, dtype=jnp.int32),
                    random_gate,
                )
            else:
                focused_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
            reset_gate = jnp.where(random_start, random_gate, 0)
        else:
            reset_gate = jax.random.randint(
                gate_key,
                (self.config.num_envs,),
                minval=0,
                maxval=max(self.max_reset_gate + 1, 1),
                dtype=jnp.int32,
            )
            random_start = reset_gate != 0
            focused_start = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        gate_id = reset_gate % self.course.num_gates
        centers = self.gate_centers[gate_id]
        approach_dirs = self.approach_dirs[gate_id]
        right_axes = self.approach_right_axes[gate_id]
        approach_lengths = self.approach_lengths[gate_id]
        widths = self.gate_widths[gate_id] * self.gate_window_scale
        heights = self.gate_heights[gate_id] * self.gate_window_scale
        spawn_min = min(
            float(self.config.racing_line_spawn_min_distance_m),
            float(self.config.racing_line_spawn_max_distance_m),
        )
        spawn_max = max(
            float(self.config.racing_line_spawn_min_distance_m),
            float(self.config.racing_line_spawn_max_distance_m),
        )
        max_distance = jnp.minimum(spawn_max, jnp.maximum(0.1, 0.85 * approach_lengths))
        min_distance = jnp.minimum(spawn_min, max_distance)
        approach_distance = jax.random.uniform(
            distance_key,
            (self.config.num_envs,),
            minval=min_distance,
            maxval=max_distance,
        )
        lateral_noise = (
            jax.random.uniform(lateral_key, (self.config.num_envs,), minval=-0.35, maxval=0.35)
            * widths
        )
        vertical_noise = (
            jax.random.uniform(vertical_key, (self.config.num_envs,), minval=-0.25, maxval=0.25)
            * heights
        )
        pos = (
            centers
            - approach_distance[:, None] * approach_dirs
            + lateral_noise[:, None] * right_axes
            + vertical_noise[:, None] * jnp.array([0.0, 0.0, 1.0])
        )
        pos = pos.at[:, 2].set(jnp.maximum(pos[:, 2], 0.35))
        vel = (
            jax.random.normal(vel_key, (self.config.num_envs, 3))
            * self.config.spawn_vel_noise_m_s
        )
        vel = vel + jnp.where(
            random_start[:, None],
            self.config.racing_line_spawn_speed_m_s * approach_dirs,
            0.0,
        )
        fallback_normals = self.gate_normals[gate_id]
        yaw_dirs = jnp.where(
            jnp.linalg.norm(approach_dirs[:, :2], axis=-1, keepdims=True) > 1.0e-5,
            approach_dirs,
            fallback_normals,
        )
        course_yaw = jnp.arctan2(yaw_dirs[:, 1], yaw_dirs[:, 0])
        yaw = course_yaw + jax.random.normal(yaw_key, (self.config.num_envs,)) * 0.08
        quat = yaw_to_quat_xyzw(yaw)
        return (
            pos.astype(jnp.float32),
            vel.astype(jnp.float32),
            quat.astype(jnp.float32),
            reset_gate,
            random_start,
            focused_start,
        )

    def _acceleration_world(self, vel: Array) -> Array:
        sim_acc = getattr(self.sim.data.states_deriv, "acc", None)
        if sim_acc is not None:
            return sim_acc[:, 0, :]
        return (vel - self.prev_vel) / self.config.dt

    def _gate_metrics(self, pos: Array, gate_id: Array) -> tuple[Array, Array, Array]:
        centers = self.gate_centers[gate_id]
        normals = self.gate_normals[gate_id]
        right_axes = self.gate_right_axes[gate_id]
        widths = self.gate_widths[gate_id] * self.gate_window_scale
        heights = self.gate_heights[gate_id] * self.gate_window_scale
        up = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), pos.shape)
        rel = pos - centers
        plane = jnp.sum(rel * normals, axis=-1)
        lateral = jnp.sum(rel * right_axes, axis=-1)
        vertical = jnp.sum(rel * up, axis=-1)
        radial = jnp.sqrt(jnp.square(lateral) + jnp.square(vertical))
        inside = (jnp.abs(lateral) <= 0.5 * widths) & (jnp.abs(vertical) <= 0.5 * heights)
        return plane, radial, inside

    def _gate_distance(self, pos: Array, gate_id: Array) -> Array:
        centers = self.gate_centers[gate_id]
        return jnp.linalg.norm(pos - centers, axis=-1)

    @staticmethod
    def _gate_crossing_position(
        pos_before: Array,
        pos_after: Array,
        plane_before: Array,
        plane_after: Array,
    ) -> Array:
        denom = plane_after - plane_before
        t = jnp.where(jnp.abs(denom) > 1.0e-6, -plane_before / denom, 1.0)
        t = jnp.clip(t, 0.0, 1.0)
        return pos_before + t[:, None] * (pos_after - pos_before)

    def _reward(
        self,
        *,
        plane_before: Array,
        plane_after: Array,
        radial_before: Array,
        radial_after: Array,
        distance_before: Array,
        distance_after: Array,
        lookahead_distance_before: Array,
        lookahead_distance_after: Array,
        cross_radial: Array,
        pos: Array,
        vel: Array,
        stalled: Array,
        action: Array,
        gate_id: Array,
        passed_gate: Array,
        missed_gate: Array,
        crashed: Array,
        finished: Array,
    ) -> Array:
        progress = jnp.clip(plane_after - plane_before, -0.25, 0.25)
        distance_progress = jnp.clip(distance_before - distance_after, -0.25, 0.25)
        lookahead_progress = jnp.clip(
            lookahead_distance_before - lookahead_distance_after,
            -0.25,
            0.25,
        )
        lookahead_active = (plane_after > -2.0) | passed_gate
        radial_progress = jnp.clip(radial_before - radial_after, -0.25, 0.25)
        center = jnp.exp(-2.5 * radial_after / self.course.radius_m)
        cross_center = jnp.exp(-2.5 * cross_radial / self.course.radius_m)
        target_direction = self.gate_centers[gate_id] - pos
        velocity_alignment = self._horizontal_alignment(vel, target_direction)
        positive_alignment = jnp.maximum(velocity_alignment, 0.0)
        action_cost = self.config.action_penalty * jnp.sum(jnp.square(action), axis=-1)
        reward = (
            self.config.plane_progress_reward * progress
            + self.config.distance_progress_reward * distance_progress
            + self.config.lookahead_progress_reward
            * lookahead_active.astype(jnp.float32)
            * lookahead_progress
            + self.config.radial_progress_reward * radial_progress
            + self.config.centerline_reward * center
            + self.config.crossing_center_reward * cross_center
            + self.config.path_alignment_reward * positive_alignment
            + self.config.stall_penalty * stalled.astype(jnp.float32)
            - action_cost
        )
        pass_reward = (
            self.config.gate_pass_bonus
            + self.config.centered_crossing_bonus * cross_center
        )
        gate_rank = (gate_id % self.course.num_gates).astype(jnp.float32)
        late_gate_multiplier = jnp.minimum(
            self.config.late_gate_reward_base**gate_rank,
            self.config.late_gate_reward_cap,
        )
        pass_reward = pass_reward * late_gate_multiplier
        reward = reward + passed_gate.astype(jnp.float32) * pass_reward
        reward = reward + finished.astype(jnp.float32) * self.config.lap_finish_bonus
        reward = reward + missed_gate.astype(jnp.float32) * self.config.miss_penalty
        reward = reward + crashed.astype(jnp.float32) * self.config.crash_penalty
        return reward.astype(jnp.float32)

    @staticmethod
    def _horizontal_alignment(a: Array, b: Array) -> Array:
        horizontal_mask = jnp.array([1.0, 1.0, 0.0], dtype=a.dtype)
        a_xy = a * horizontal_mask
        b_xy = b * horizontal_mask
        a_xy = a_xy / jnp.maximum(
            jnp.linalg.norm(a_xy, axis=-1, keepdims=True),
            1.0e-6,
        )
        b_xy = b_xy / jnp.maximum(
            jnp.linalg.norm(b_xy, axis=-1, keepdims=True),
            1.0e-6,
        )
        return jnp.sum(a_xy * b_xy, axis=-1)


def _balanced_random_start_mask(
    key: Array,
    reset_mask: Array,
    current_random_start: Array,
    random_fraction: float,
) -> Array:
    """Choose reset branches so active environments stay near the requested mix."""

    reset_mask = jnp.asarray(reset_mask, dtype=jnp.bool_)
    current_random_start = jnp.asarray(current_random_start, dtype=jnp.bool_)
    env_count = reset_mask.shape[0]
    target_count = jnp.rint(float(random_fraction) * env_count).astype(jnp.int32)
    kept_random_count = jnp.sum((~reset_mask & current_random_start).astype(jnp.int32))
    reset_count = jnp.sum(reset_mask.astype(jnp.int32))
    random_needed = jnp.clip(target_count - kept_random_count, 0, reset_count)
    priority = jnp.where(
        reset_mask,
        jax.random.uniform(key, reset_mask.shape),
        2.0,
    )
    ranks = jnp.argsort(jnp.argsort(priority))
    return reset_mask & (ranks < random_needed)
