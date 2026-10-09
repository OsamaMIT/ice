"""Shared 17-state model: [position, velocity, xyzw, body rates, RPM/hover]."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.observations import (
    quat_multiply_xyzw,
    quat_rotate_xyzw,
    quat_conjugate_xyzw,
)
from .geometry import fingerprint


def rotation_matrix(q):
    return jax.vmap(lambda v: quat_rotate_xyzw(q, v))(jnp.eye(3)).T


def attitude_error(q, target):
    error = quat_multiply_xyzw(quat_conjugate_xyzw(target), q)
    return 2.0 * jnp.where(error[3] < 0, -error[:3], error[:3])


def rotvec_quaternion(v):
    # Analytic small-angle branch avoids undefined norm derivatives at hover.
    squared = jnp.dot(v, v)
    angle = jnp.sqrt(jnp.maximum(squared, 1e-12))
    scale = jnp.where(squared < 1e-8, 0.5 - squared / 48, jnp.sin(angle / 2) / angle)
    return jnp.concatenate((v * scale, jnp.array([jnp.cos(angle / 2)])))


def acceleration_quaternion(acceleration, yaw):
    """Desired thrust direction and nominal heading, with a robust SO(3) conversion."""
    from jax.scipy.spatial.transform import Rotation

    z = acceleration + jnp.array([0.0, 0.0, 9.81])
    z = z / jnp.maximum(jnp.linalg.norm(z), 1e-6)
    heading = jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0])
    y = jnp.cross(z, heading)
    alternative = jnp.cross(z, jnp.array([0.0, 1.0, 0.0]))
    y = jnp.where(jnp.linalg.norm(y) < 1e-4, alternative, y)
    y = y / jnp.maximum(jnp.linalg.norm(y), 1e-6)
    return Rotation.from_matrix(jnp.stack((jnp.cross(y, z), y, z), axis=1)).as_quat()


@dataclass(eq=False)
class DroneModel:
    parameters: dict
    rpm_min: float
    rpm_hover: float
    rpm_max: float
    physics_dt: float = 0.002

    @classmethod
    def from_env(cls, env):
        p = env.sim.data.params
        names = (
            "mass",
            "L",
            "prop_inertia",
            "gravity_vec",
            "J",
            "J_inv",
            "rpm2thrust",
            "rpm2torque",
            "mixing_matrix",
            "drag_matrix",
            "rotor_dyn_coef",
        )
        params = {name: np.asarray(getattr(p, name)).copy() for name in names}
        params["mass"] = float(params["mass"].reshape(-1)[0])
        params["J"] = params["J"].reshape(-1, 3, 3)[0]
        params["J_inv"] = params["J_inv"].reshape(-1, 3, 3)[0]
        return cls(
            params,
            float(env.motor_rpm_min),
            float(env.motor_rpm_hover),
            float(env.motor_rpm_max),
            1 / env.config.sim_hz,
        )

    @property
    def fingerprint(self):
        return fingerprint(self)

    def pack(self, states):
        return jnp.concatenate(
            (
                states.pos[:, 0],
                states.vel[:, 0],
                states.quat[:, 0],
                states.ang_vel[:, 0],
                states.rotor_vel[:, 0] / self.rpm_hover,
            ),
            axis=-1,
        )

    def rpm(self, u):
        return self.rpm_hover + jnp.where(
            u >= 0,
            u * (self.rpm_max - self.rpm_hover),
            u * (self.rpm_hover - self.rpm_min),
        )

    def action(self, rpm):
        delta = rpm - self.rpm_hover
        return jnp.clip(
            delta
            / jnp.where(
                delta >= 0, self.rpm_max - self.rpm_hover, self.rpm_hover - self.rpm_min
            ),
            -1.0,
            1.0,
        )

    def derivative(self, x, u):
        from drone_models.first_principles import dynamics

        result = dynamics(
            pos=x[:3],
            vel=x[3:6],
            quat=x[6:10],
            ang_vel=x[10:13],
            rotor_vel=x[13:17] * self.rpm_hover,
            cmd=self.rpm(u),
            **{k: jnp.asarray(v) for k, v in self.parameters.items()},
        )
        # Use quaternion exponential integration, not the model's quaternion derivative.
        return jnp.concatenate(
            (result[0], result[2], jnp.zeros(4), result[3], result[4] / self.rpm_hover)
        )

    def tick(self, x, u, dt):
        dx = self.derivative(x, u)
        next_x = x + dt * dx
        q = quat_multiply_xyzw(x[6:10], rotvec_quaternion(x[10:13] * dt))
        return next_x.at[6:10].set(q)

    @partial(jax.jit, static_argnums=(0, 3))
    def step(self, x, u, dt):
        count = max(1, int(round(dt / self.physics_dt)))
        return jax.lax.fori_loop(
            0, count, lambda _, state: self.tick(state, u, dt / count), x
        )

    def symbolic_step(self, substeps):
        """CasADi model with the same Euler/exponential rule as Crazyflow."""
        import casadi as ca
        from drone_models.first_principles import symbolic_dynamics

        derivative, original_x, original_u, _ = symbolic_dynamics(**self.parameters)
        f = ca.Function("physical_derivative", [original_x, original_u], [derivative])
        x, u, dt = ca.MX.sym("x", 17), ca.MX.sym("u", 4), ca.MX.sym("dt")
        state = x
        rpm = self.rpm_hover + ca.if_else(
            u >= 0,
            u * (self.rpm_max - self.rpm_hover),
            u * (self.rpm_hover - self.rpm_min),
        )
        h = dt / substeps
        for _ in range(substeps):
            raw = ca.vertcat(
                state[:3],
                state[6:10],
                state[3:6],
                state[10:13],
                state[13:17] * self.rpm_hover,
            )
            d = f(raw, rpm)
            v = state[10:13] * h
            sq = ca.dot(v, v)
            angle = ca.sqrt(ca.fmax(sq, 1e-12))
            scale = ca.if_else(sq < 1e-8, 0.5 - sq / 48, ca.sin(angle / 2) / angle)
            r, w = v * scale, ca.cos(angle / 2)
            qv, qw = state[6:9], state[9]
            q = ca.vertcat(qw * r + w * qv + ca.cross(qv, r), qw * w - ca.dot(qv, r))
            q = q / ca.sqrt(ca.dot(q, q))
            state = ca.vertcat(
                state[:3] + h * d[:3],
                state[3:6] + h * d[7:10],
                q,
                state[10:13] + h * d[10:13],
                state[13:17] + h * d[13:17] / self.rpm_hover,
            )
        return ca.Function("step", [x, u, dt], [state])

    @partial(jax.jit, static_argnums=(0, 4))
    def rollout_intervals(self, states, commands, durations, count):
        def tick(current, _):
            next_states = jax.vmap(self.tick)(current, commands, durations / count)
            return next_states, next_states

        _, tail = jax.lax.scan(tick, states, None, length=count)
        return jnp.concatenate((states[:, None], jnp.swapaxes(tail, 0, 1)), axis=1)

    @partial(jax.jit, static_argnums=0)
    def geometric_action(self, x, reference):
        """Bounded position/attitude fallback; never used as the nominal MPC."""
        p = {k: jnp.asarray(v) for k, v in self.parameters.items()}
        desired_acc = 6 * (reference[:3] - x[:3]) + 4 * (reference[3:6] - x[3:6])
        thrust_world = p["mass"] * (desired_acc - p["gravity_vec"])
        body_z = quat_rotate_xyzw(x[6:10], jnp.array([0.0, 0.0, 1.0]))
        force = jnp.maximum(jnp.dot(thrust_world, body_z), 0.0)
        yaw = jnp.arctan2(
            2 * (reference[9] * reference[8] + reference[6] * reference[7]),
            1 - 2 * (reference[7] ** 2 + reference[8] ** 2),
        )
        target_q = acceleration_quaternion(desired_acc, yaw)
        error = attitude_error(x[6:10], target_q)
        torque = p["J"] @ (-60 * error - 12 * (x[10:13] - reference[10:13]))
        curve, moment = p["rpm2thrust"], p["rpm2torque"]
        ratio = (moment[1] * self.rpm_hover + moment[2] * self.rpm_hover**2) / (
            curve[1] * self.rpm_hover + curve[2] * self.rpm_hover**2
        )
        allocation = jnp.concatenate(
            (
                jnp.ones((1, 4)),
                p["mixing_matrix"] * jnp.array([p["L"], p["L"], ratio])[:, None],
            ),
            axis=0,
        )
        forces = jnp.linalg.solve(
            allocation, jnp.concatenate((jnp.array([force]), torque))
        )
        rpm = (
            -curve[1]
            + jnp.sqrt(
                jnp.maximum(curve[1] ** 2 - 4 * curve[2] * (curve[0] - forces), 0)
            )
        ) / (2 * curve[2])
        return self.action(rpm)
