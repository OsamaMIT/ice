"""20 Hz residual-policy interface over a 100 Hz MPC and 500 Hz simulator."""

from dataclasses import replace
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.env import CrazyflowRacingEnv
from a2rl_drone_training.rewards import REWARD_COMPONENT_NAMES
from a2rl_drone_training.observations import (
    quat_to_yaw_xyzw,
    quat_multiply_xyzw,
    quat_conjugate_xyzw,
    quat_rotate_xyzw,
)
from .dynamics import DroneModel, acceleration_quaternion
from .geometry import fingerprint
from .reference import Reference
from .mpc import QuaternionMPC

EXTRA_CORE_DIM = 28
RESIDUAL_ACTION_SPACE = "world_position_speed_residual_v1"
RESIDUAL_ACTION_NAMES = ("offset_x", "offset_y", "offset_z", "speed_offset")


def residual_training_config(config):
    """Normalize new-mode configuration once, including physical-time discounting."""
    if config.controller != "residual_mpc":
        return config
    r = config.residual
    if r.reference_path is None:
        raise ValueError(
            "residual_mpc requires --reference-path from the offline planner"
        )
    if (
        config.env.physics != "first_principles"
        or config.env.laps != 1
        or config.env.corner_reset_bank
    ):
        raise ValueError(
            "Residual MPC requires first_principles, one course run, and no corner reset bank"
        )
    if (
        min(r.policy_hz, config.mpc.update_hz, config.env.sim_hz) <= 0
        or config.env.sim_hz % config.mpc.update_hz
        or config.mpc.update_hz % r.policy_hz
    ):
        raise ValueError(
            "Physics, MPC, and policy rates must be positive integer multiples"
        )
    if (
        not all(
            np.isfinite(v)
            for v in (
                r.smoothing_time_s,
                r.position_limit_m,
                r.speed_fraction,
                r.discount_per_second,
                r.gae_lambda_per_second,
                r.change_penalty,
                r.estimation_noise_std,
                r.spawn_position_std_m,
                r.spawn_velocity_std_m_s,
                config.env.max_episode_time_s,
            )
        )
        or r.smoothing_time_s <= 0
        or r.position_limit_m < 0
        or min(
            r.change_penalty,
            r.estimation_noise_std,
            r.spawn_position_std_m,
            r.spawn_velocity_std_m_s,
        )
        < 0
        or config.env.max_episode_time_s <= 0
        or not 0 <= r.speed_fraction < 1
        or not 0 < r.discount_per_second <= 1
        or not 0 < r.gae_lambda_per_second <= 1
    ):
        raise ValueError(
            "Invalid residual bounds, smoothing, or discount configuration"
        )
    if config.obs.core_dim not in (24, 24 + EXTRA_CORE_DIM):
        raise ValueError("Unsupported residual observation schema")
    if config.critic_obs.dim not in (37, 50):
        raise ValueError("Unsupported residual critic observation schema")
    return replace(
        config,
        obs=replace(
            config.obs,
            core_dim=24 + EXTRA_CORE_DIM,
            sensor_noise_scale=0.0,
            pnp_dropout_prob=0.0,
        ),
        critic_obs=replace(config.critic_obs, dim=50),
        env=replace(
            config.env,
            control_hz=r.policy_hz,
            gate_window_scale=1.0,
            reset_distribution="evaluation",
            gate_frame_collisions=True,
            gate_frame_depth_m=config.planner.frame_depth_m,
        ),
        curriculum=replace(config.curriculum, enabled=False, corner_practice=False),
        ppo=replace(
            config.ppo,
            gamma=r.discount_per_second ** (1 / r.policy_hz),
            gae_lambda=r.gae_lambda_per_second ** (1 / r.policy_hz),
        ),
    )


@jax.jit
def smooth_offset(value, derivative, target, t, tau):
    """Exact critically damped filter, including derivatives and time integral."""
    e = value - target
    b = derivative + e / tau
    decay = jnp.exp(-t / tau)
    position = target + (e + b * t) * decay
    velocity = (b - (e + b * t) / tau) * decay
    acceleration = (-2 * b / tau + (e + b * t) / tau**2) * decay
    integral = (
        target * t
        + e * tau * (1 - decay)
        + b * (tau * tau * (1 - decay) - tau * t * decay)
    )
    return position, velocity, acceleration, integral


class ResidualMPCEnv:
    def __init__(self, training_config, *, env_config=None, course=None):
        self.training_config = residual_training_config(training_config)
        cfg = self.training_config
        self.config = env_config or cfg.env
        self.obs_config = cfg.obs
        self.residual = cfg.residual
        self.base = CrazyflowRacingEnv(
            replace(
                self.config,
                control_hz=self.config.sim_hz,
                auto_reset=False,
                gate_frame_collisions=True,
                gate_window_scale=1.0,
                reset_distribution="evaluation",
            ),
            replace(cfg.obs, core_dim=24),
            course=course,
        )
        self.model = DroneModel.from_env(self.base)
        self.reference = Reference.load(
            self.residual.reference_path,
            course_fingerprint=fingerprint(self.base.course),
            model_fingerprint=self.model.fingerprint,
        )
        if (
            self.reference.metadata.get("vehicle_radius_m")
            != self.config.vehicle_radius_m
            or self.reference.metadata.get("planner", {}).get("frame_depth_m")
            != cfg.planner.frame_depth_m
        ):
            raise ValueError(
                "Reference vehicle radius or frame depth differs from runtime geometry"
            )
        self.mpc = QuaternionMPC(self.model, cfg.mpc, self.config.num_envs)
        n = self.config.num_envs
        self.offset = jnp.zeros((n, 4))
        self.offset_velocity = jnp.zeros((n, 4))
        self.target = jnp.zeros((n, 4))
        self.previous_residual = jnp.zeros((n, 4))
        self.done = jnp.zeros(n, dtype=bool)
        self.returns = jnp.zeros(n)
        self.lengths = jnp.zeros(n)
        self.episode_ticks = jnp.zeros(n, dtype=jnp.int32)
        self.physics_steps_total = 0
        self.policy_steps_total = 0
        self.rng = jax.random.key(0)
        self.last_controller_metrics = {}
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def observation_dim(self):
        return self.obs_config.dim

    def set_curriculum(self, parameters):
        # The baseline and residual policy always use physical openings.
        self.base.gate_window_scale = 1.0
        self.base.time_cost_scale = 1.0

    def reset(self, seed=None, mask=None, forced_reset_gates=None):
        n = self.config.num_envs
        mask = (
            jnp.ones(n, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
        )
        if forced_reset_gates is not None and bool(
            jnp.any(jnp.asarray(forced_reset_gates) != 0)
        ):
            raise ValueError("Residual mode uses full-course starts only")
        if seed is not None:
            self.rng = jax.random.key(seed)
        self.base.reset(
            seed=seed, mask=mask, forced_reset_gates=jnp.zeros(n, dtype=jnp.int32)
        )
        x = jnp.broadcast_to(
            jnp.asarray(self.reference.states[0], dtype=jnp.float32), (n, 17)
        )
        self.rng, kp, kv = jax.random.split(self.rng, 3)
        x = x.at[:, :3].add(
            jax.random.normal(kp, (n, 3)) * self.residual.spawn_position_std_m
        )
        x = x.at[:, 3:6].add(
            jax.random.normal(kv, (n, 3)) * self.residual.spawn_velocity_std_m_s
        )
        states = self.base.sim.data.states

        def choose(new, old):
            return jnp.where(mask[:, None, None], new[:, None], old)

        states = states.replace(
            pos=choose(x[:, :3], states.pos),
            vel=choose(x[:, 3:6], states.vel),
            quat=choose(x[:, 6:10], states.quat),
            ang_vel=choose(x[:, 10:13], states.ang_vel),
            rotor_vel=choose(x[:, 13:] * self.model.rpm_hover, states.rotor_vel),
        )
        self.base.sim.data = self.base.sim.data.replace(states=states)
        self.base.prev_vel = jnp.where(mask[:, None], x[:, 3:6], self.base.prev_vel)
        for name in ("offset", "offset_velocity", "target", "previous_residual"):
            setattr(self, name, jnp.where(mask[:, None], 0.0, getattr(self, name)))
        self.done = jnp.where(mask, False, self.done)
        self.returns = jnp.where(mask, 0.0, self.returns)
        self.lengths = jnp.where(mask, 0.0, self.lengths)
        self.episode_ticks = jnp.where(mask, 0, self.episode_ticks)
        self.mpc.reset(mask)
        return self.observe()

    def estimated_state(self):
        state = self.model.pack(self.base.sim.data.states)
        if self.residual.estimation_noise_std:
            self.rng, key = jax.random.split(self.rng)
            # Position/velocity noise only; quaternion and rates retain exact
            # values in this initial estimator-free interface.
            state = state.at[:, :6].add(
                jax.random.normal(key, state[:, :6].shape)
                * self.residual.estimation_noise_std
            )
        return state

    def observe(self, **kwargs):
        original = self.base.observe(noisy=False)
        x = self.estimated_state()
        # Crazyflow first-principles angular rates are body-frame quantities.
        # The policy and MPC also share the same noisy velocity sample here.
        original = original.at[:, :3].set(x[:, 10:13])
        original = original.at[:, 10:13].set(
            quat_rotate_xyzw(quat_conjugate_xyzw(x[:, 6:10]), x[:, 3:6])
        )
        t = self.reference.project(
            x[:, :3] - self.offset[:, :3], self.base.gate_counter
        )
        nominal, _, _ = self.reference.sample(t)
        ahead, _, _ = self.reference.sample(
            t[:, None] + jnp.array([0.1, 0.25, 0.5])[None]
        )
        extra = jnp.concatenate(
            (
                nominal[:, :3] + self.offset[:, :3] - x[:, :3],
                nominal[:, 3:6] * (1 + self.offset[:, 3:4])
                + self.offset_velocity[:, :3]
                - x[:, 3:6],
                (ahead[:, :, :3] - x[:, None, :3]).reshape(self.config.num_envs, 9),
                jnp.linalg.norm(nominal[:, 3:6], axis=-1, keepdims=True),
                self.offset,
                self.offset_velocity,
                self.previous_residual,
            ),
            axis=-1,
        )
        return jnp.concatenate((original[:, :24], extra, original[:, 24:]), axis=-1)

    def privileged_observe(self):
        state = self.model.pack(self.base.sim.data.states)
        progress = self.reference.project(
            state[:, :3] - self.offset[:, :3], self.base.gate_counter
        )
        return jnp.concatenate(
            (
                self.base.privileged_observe(),
                self.offset,
                self.offset_velocity,
                self.previous_residual,
                progress[:, None] / self.reference.time[-1],
            ),
            axis=-1,
        )

    def references(self, x):
        return self._references_for(
            x, self.offset, self.offset_velocity, self.target, self.base.gate_counter
        )

    @partial(jax.jit, static_argnums=0)
    def _references_for(self, x, offset, offset_velocity, target, gate_counter):
        t = self.reference.project(x[:, :3] - offset[:, :3], gate_counter)
        future = jnp.arange(self.mpc.horizon + 1) * self.mpc.config.prediction_dt
        o, od, odd, integral = smooth_offset(
            offset[:, None, :],
            offset_velocity[:, None, :],
            target[:, None, :],
            future[None, :, None],
            self.residual.smoothing_time_s,
        )
        nominal, feedforward, acc = self.reference.sample(
            t[:, None] + future[None, :] + integral[:, :, 3]
        )
        speed = 1 + o[:, :, 3:4]
        modified = (
            nominal.at[:, :, :3]
            .add(o[:, :, :3])
            .at[:, :, 3:6]
            .set(nominal[:, :, 3:6] * speed + od[:, :, :3])
        )
        desired_acc = (
            acc * speed**2 + nominal[:, :, 3:6] * od[:, :, 3:4] + odd[:, :, :3]
        )
        yaw = quat_to_yaw_xyzw(nominal[:, :, 6:10])
        body_velocity = quat_rotate_xyzw(
            quat_conjugate_xyzw(nominal[:, :, 6:10]), modified[:, :, 3:6]
        )
        body_drag = jnp.einsum(
            "ij,...j->...i",
            jnp.asarray(self.model.parameters["drag_matrix"]),
            body_velocity,
        )
        world_drag_acc = (
            quat_rotate_xyzw(nominal[:, :, 6:10], body_drag)
            / self.model.parameters["mass"]
        )
        quat = jax.vmap(jax.vmap(acceleration_quaternion))(
            desired_acc - world_drag_acc, yaw
        )
        zero = jnp.all((o == 0) & (od == 0), axis=-1)
        modified = modified.at[:, :, 6:10].set(
            jnp.where(zero[:, :, None], nominal[:, :, 6:10], quat)
        )
        q = modified[:, :, 6:10]
        difference = quat_multiply_xyzw(quat_conjugate_xyzw(q[:, :-1]), q[:, 1:])
        rates = (
            2
            * jnp.where(
                difference[:, :, 3:4] < 0, -difference[:, :, :3], difference[:, :, :3]
            )
            / self.mpc.config.prediction_dt
        )
        rates = jnp.concatenate((rates, rates[:, -1:]), axis=1)
        modified = modified.at[:, :, 10:13].set(
            jnp.where(zero[:, :, None], nominal[:, :, 10:13], rates)
        )
        return modified, feedforward[:, :-1]

    def step(self, action):
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (self.config.num_envs, 4) or not bool(
            jnp.all(jnp.isfinite(action))
        ):
            raise ValueError("Residual action must be finite with shape (num_envs, 4)")
        action = jnp.clip(action, -1, 1)
        delta = action - self.previous_residual
        self.target = action * jnp.array(
            [self.residual.position_limit_m] * 3 + [self.residual.speed_fraction]
        )
        self.previous_residual = action
        substeps = self.config.sim_hz // self.residual.policy_hz
        mpc_stride = self.config.sim_hz // self.mpc.config.update_hz
        n = self.config.num_envs
        active = ~self.done
        initial_active = active
        terminated = jnp.zeros(n, dtype=bool)
        truncated = jnp.zeros(n, dtype=bool)
        reward = jnp.zeros(n)
        undiscounted = jnp.zeros(n)
        count = jnp.zeros(n, dtype=jnp.int32)
        accumulated = {}
        final_info = None
        pass_events = jnp.zeros((n, self.course.num_gates))
        clearance_events = jnp.zeros_like(pass_events)
        time_events = jnp.zeros_like(pass_events)
        motor = jnp.zeros((n, 4))
        fallback_count = jnp.zeros(n)
        minimum_clearance = jnp.full(n, jnp.inf)
        latency = []
        # Freeze completed worlds at their terminal state until the macro-step
        # ends. No reward, reset, or transition from a new episode leaks in.
        tracked = (
            "gate_counter",
            "reset_gate",
            "started_local",
            "elapsed_steps",
            "segment_steps",
            "last_action",
            "prev_vel",
            "current_acc",
            "stall_steps",
            "episode_return",
            "episode_length",
        )
        for tick in range(substeps):
            if tick % mpc_stride == 0:
                x = self.estimated_state()
                references, ff = self.references(x)
                motor, controller_info = self.mpc.command(x, references, ff)
                fallback_count += controller_info["mpc_fallback"] & active
                latency.append(controller_info["mpc_latency_s"])
            old_states = self.base.sim.data.states
            old = {key: getattr(self.base, key) for key in tracked}
            _, _, term, trunc, info = self.base.step(motor)

            def keep(new, previous):
                return jnp.where(
                    active.reshape((n,) + (1,) * (new.ndim - 1)), new, previous
                )

            self.base.sim.data = self.base.sim.data.replace(
                states=jax.tree.map(keep, self.base.sim.data.states, old_states)
            )
            for key in tracked:
                setattr(self.base, key, keep(getattr(self.base, key), old[key]))
            clearance = self.base.frame_geometry.clearance(
                self.base.sim.data.states.pos[:, 0], self.config.vehicle_radius_m
            )
            minimum_clearance = jnp.where(
                active, jnp.minimum(minimum_clearance, clearance), minimum_clearance
            )
            term = term & active
            trunc = trunc & active
            components = {name: jnp.zeros(n) for name in REWARD_COMPONENT_NAMES}
            components["reward_time"] = jnp.full(n, -1 / self.config.sim_hz)
            # The current gate counter anchors progress, including stacked turns.
            old_time = self.reference.project(old_states.pos[:, 0], old["gate_counter"])
            new_time = self.reference.project(
                self.base.sim.data.states.pos[:, 0], self.base.gate_counter
            )
            gate_times = jnp.asarray(self.reference.time[self.reference.gate_indices])

            def potential(t, gates):
                gate = jnp.minimum(gates, self.course.num_gates - 1)
                begin = jnp.where(gate == 0, 0.0, gate_times[jnp.maximum(gate - 1, 0)])
                fraction = jnp.clip(
                    (t - begin) / jnp.maximum(gate_times[gate] - begin, 1e-6), 0, 1
                )
                return jnp.where(
                    gates >= self.course.num_gates,
                    float(self.course.num_gates),
                    gates + fraction,
                )

            phi = potential(old_time, old["gate_counter"])
            next_phi = potential(new_time, self.base.gate_counter)
            # True terminal potentials are zero; truncations preserve bootstrap.
            next_phi = jnp.where(term, 0.0, next_phi)
            components["reward_potential_progress"] = (
                self.config.potential_reward_weight
                * (
                    self.residual.discount_per_second ** (1 / self.config.sim_hz)
                    * next_phi
                    - phi
                )
            )
            components["reward_gate"] = (
                self.config.gate_pass_bonus * info["passed_gate"]
            )
            components["reward_finish"] = (
                self.config.lap_finish_bonus * info["course_finished"]
            )
            components["reward_crash"] = -self.config.v2_crash_penalty * (
                info["crashed"] | info["out_of_bounds"]
            )
            components["reward_miss"] = -self.config.v2_miss_penalty * (
                info["missed_gate"] | info["deadline"]
            )
            if tick == 0:
                components["reward_smoothness"] = (
                    -self.residual.change_penalty * jnp.sum(delta * delta, axis=-1)
                )
            components["reward_total"] = sum(
                v for k, v in components.items() if k != "reward_total"
            )
            info.update(components)
            physics_reward = jnp.where(active, components["reward_total"], 0.0)
            reward += (
                self.residual.discount_per_second ** (tick / self.config.sim_hz)
                * physics_reward
            )
            undiscounted += physics_reward
            count += active
            self.returns += physics_reward
            self.episode_ticks += active
            self.lengths = self.episode_ticks / self.config.sim_hz
            measurement = info["passed_gate"] & active
            onehot = (
                jax.nn.one_hot(info["gate_id"], self.course.num_gates)
                * measurement[:, None]
            )
            pass_events += onehot
            clearance_events += onehot * jnp.nan_to_num(info["gate_clearance"])[:, None]
            time_events += onehot * jnp.nan_to_num(info["segment_time"])[:, None]
            if final_info is None:
                final_info = info.copy()
            else:
                final_info = {k: keep(v, final_info[k]) for k, v in info.items()}
            for key, value in info.items():
                if key.startswith("reward_") or key in ("track_progress_delta",):
                    accumulated[key] = accumulated.get(
                        key, jnp.zeros_like(value)
                    ) + jnp.where(active, value, 0.0)
                elif value.dtype == jnp.bool_:
                    accumulated[key] = accumulated.get(key, jnp.zeros_like(value)) | (
                        value & active
                    )
            terminated |= term
            truncated |= trunc
            update_mask = active
            active &= ~(term | trunc)
            dt = 1 / self.config.sim_hz
            o, v, _, _ = smooth_offset(
                self.offset,
                self.offset_velocity,
                self.target,
                dt,
                self.residual.smoothing_time_s,
            )
            self.offset = jnp.where(update_mask[:, None], o, self.offset)
            self.offset_velocity = jnp.where(
                update_mask[:, None], v, self.offset_velocity
            )
            if not bool(jnp.any(active)):
                break
        self.done |= terminated | truncated
        self.physics_steps_total += int(jnp.sum(count))
        self.policy_steps_total += int(jnp.sum(initial_active))
        info = {**final_info, **accumulated}
        info.update(
            gate_pass_events=pass_events,
            gate_clearance_events=clearance_events,
            gate_segment_time_events=time_events,
            physics_steps=count,
            transition_seconds=count / self.config.sim_hz,
            action_delta_squared=delta * delta,
            final_return=jnp.where(terminated | truncated, self.returns, jnp.nan),
            final_length=jnp.where(
                terminated | truncated, self.lengths * self.residual.policy_hz, jnp.nan
            ),
            mpc_fallback_count=fallback_count,
            minimum_frame_clearance=minimum_clearance,
        )
        # Capture the extended actor observation and privileged state before reset.
        info["final_observation"] = self.observe()
        info["final_critic_observation"] = self.privileged_observe()
        self.last_controller_metrics = {
            "physics_steps": self.physics_steps_total,
            "policy_steps": self.policy_steps_total,
            "mpc_fallbacks": int(jnp.sum(fallback_count)),
            "mpc_latency_mean_s": float(np.mean(latency)),
            "mpc_latency_max_s": float(np.max(latency)),
            "mpc_latencies_s": latency,
        }
        reset = (
            (terminated | truncated)
            if self.config.auto_reset
            else jnp.zeros(n, dtype=bool)
        )
        obs = (
            self.reset(mask=reset)
            if bool(jnp.any(reset))
            else info["final_observation"]
        )
        info["reset_occurred"] = reset
        for key in ("reset_started_local", "reset_random_start", "reset_focused_start"):
            info[key] = jnp.zeros(n, dtype=bool)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.base.close()
