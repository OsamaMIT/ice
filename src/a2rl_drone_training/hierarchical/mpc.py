"""Batched bounded iLQR MPC with rotor dynamics and quaternion attitude error."""

from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import MPCConfig
from .dynamics import attitude_error


class QuaternionMPC:
    def __init__(self, model, config=MPCConfig(), num_envs=1):
        self.model, self.config, self.num_envs = model, config, num_envs
        self.horizon = int(round(config.horizon_s / config.prediction_dt))
        weights = (
            config.position_weight,
            config.velocity_weight,
            config.attitude_weight,
            config.angular_velocity_weight,
            config.motor_weight,
            config.motor_change_weight,
        )
        if (
            not all(np.isfinite(w) and w >= 0 for w in weights)
            or config.regularization <= 0
        ):
            raise ValueError(
                "MPC weights must be finite/nonnegative and regularization positive"
            )
        if (
            self.horizon < 2
            or config.iterations < 1
            or config.update_hz <= 0
            or config.prediction_dt <= 0
        ):
            raise ValueError("Invalid MPC rates, horizon, or iteration count")
        if not np.isclose(self.horizon * config.prediction_dt, config.horizon_s):
            raise ValueError("MPC horizon must be divisible by prediction_dt")
        if not np.isclose(
            config.prediction_dt / model.physics_dt,
            round(config.prediction_dt / model.physics_dt),
        ):
            raise ValueError(
                "MPC prediction_dt must be a multiple of the physics timestep"
            )
        self.warm = jnp.zeros((num_envs, self.horizon, 4))
        self.previous = jnp.zeros((num_envs, 4))
        self.valid = jnp.zeros(num_envs, dtype=bool)
        self.failures = jnp.zeros(num_envs, dtype=jnp.int32)
        self.total_fallbacks = jnp.zeros(num_envs, dtype=jnp.int32)
        self.last_latency_s = 0.0
        c = config
        self.weights = jnp.sqrt(
            jnp.array(
                [c.position_weight] * 3
                + [c.velocity_weight] * 3
                + [c.attitude_weight] * 3
                + [c.angular_velocity_weight] * 3
                + [c.motor_weight] * 4
            )
        )

    def reset(self, mask):
        self.warm = jnp.where(mask[:, None, None], 0.0, self.warm)
        self.previous = jnp.where(mask[:, None], 0.0, self.previous)
        self.valid = jnp.where(mask, False, self.valid)
        self.failures = jnp.where(mask, 0, self.failures)
        self.total_fallbacks = jnp.where(mask, 0, self.total_fallbacks)

    def transition(self, z, u):
        return jnp.concatenate(
            (self.model.step(z[:17], u, self.config.prediction_dt), u)
        )

    def residual(self, z, u, reference):
        x = z[:17]
        state_error = jnp.concatenate(
            (
                x[:6] - reference[:6],
                attitude_error(x[6:10], reference[6:10]),
                x[10:17] - reference[10:17],
            )
        )
        return jnp.concatenate(
            (
                self.weights * state_error,
                jnp.sqrt(self.config.motor_change_weight) * (u - z[17:]),
            )
        )

    def terminal_residual(self, z, reference):
        return jnp.sqrt(5.0) * self.residual(z, z[17:], reference)[:16]

    def rollout(self, z, commands):
        def tick(x, u):
            y = self.transition(x, u)
            return y, y

        _, tail = jax.lax.scan(tick, z, commands)
        return jnp.concatenate((z[None], tail))

    def cost(self, states, commands, reference):
        r = jax.vmap(self.residual)(states[:-1], commands, reference[:-1])
        terminal = self.terminal_residual(states[-1], reference[-1])
        return 0.5 * (jnp.sum(r * r) + jnp.sum(terminal * terminal))

    def solve_one(self, x, previous, warm, reference):
        z = jnp.concatenate((x, previous))
        states = self.rollout(z, warm)
        initial_cost = self.cost(states, warm, reference)

        def iteration(_, carry):
            states, commands, cost, healthy = carry
            a, b = jax.vmap(jax.jacfwd(self.transition, argnums=(0, 1)))(
                states[:-1], commands
            )
            residual = jax.vmap(self.residual)(states[:-1], commands, reference[:-1])
            rx, ru = jax.vmap(jax.jacfwd(self.residual, argnums=(0, 1)))(
                states[:-1], commands, reference[:-1]
            )
            terminal = self.terminal_residual(states[-1], reference[-1])
            terminal_jac = jax.jacfwd(self.terminal_residual)(states[-1], reference[-1])
            value_x, value_xx = terminal_jac.T @ terminal, terminal_jac.T @ terminal_jac

            def backward(value, data):
                vx, vxx = value
                ai, bi, ri, rxi, rui = data
                qx = rxi.T @ ri + ai.T @ vx
                qu = rui.T @ ri + bi.T @ vx
                qxx = rxi.T @ rxi + ai.T @ vxx @ ai
                qux = rui.T @ rxi + bi.T @ vxx @ ai
                quu = (
                    rui.T @ rui
                    + bi.T @ vxx @ bi
                    + self.config.regularization * jnp.eye(4)
                )
                gains = -jnp.linalg.solve(
                    quu, jnp.concatenate((qu[:, None], qux), axis=1)
                )
                k, K = gains[:, 0], gains[:, 1:]
                vx = qx + K.T @ quu @ k + K.T @ qu + qux.T @ k
                vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
                return (vx, (vxx + vxx.T) / 2), (k, K)

            _, (ks, Ks) = jax.lax.scan(
                backward, (value_x, value_xx), (a, b, residual, rx, ru), reverse=True
            )

            def candidate(alpha):
                def forward(current, data):
                    old, u, k, K = data
                    # Align quaternion hemisphere before the local linear update.
                    current = current.at[6:10].set(
                        jnp.where(
                            jnp.dot(current[6:10], old[6:10]) < 0,
                            -current[6:10],
                            current[6:10],
                        )
                    )
                    control = jnp.clip(u + alpha * k + K @ (current - old), -1.0, 1.0)
                    nxt = self.transition(current, control)
                    return nxt, (nxt, control)

                _, (tail, us) = jax.lax.scan(
                    forward, z, (states[:-1], commands, ks, Ks)
                )
                xs = jnp.concatenate((z[None], tail))
                return xs, us, self.cost(xs, us, reference)

            candidates = jax.vmap(candidate)(jnp.array([1.0, 0.5, 0.25, 0.1]))
            costs = jnp.where(jnp.isfinite(candidates[2]), candidates[2], jnp.inf)
            best = jnp.argmin(costs)
            improve = costs[best] < cost
            return (
                jnp.where(improve, candidates[0][best], states),
                jnp.where(improve, candidates[1][best], commands),
                jnp.minimum(cost, costs[best]),
                healthy & jnp.any(jnp.isfinite(costs)),
            )

        states, commands, cost, healthy = jax.lax.fori_loop(
            0,
            self.config.iterations,
            iteration,
            (states, warm, initial_cost, jnp.array(True)),
        )
        valid = (
            healthy
            & jnp.isfinite(cost)
            & jnp.all(jnp.isfinite(states))
            & jnp.all(jnp.isfinite(commands))
        )
        return commands, valid, cost

    @partial(jax.jit, static_argnums=0)
    def _batch_solve(self, x, previous, warm, reference):
        return jax.vmap(self.solve_one)(x, previous, warm, reference)

    def command(self, states, references, feedforward):
        """Return one motor command per world and numerical/fallback telemetry."""
        fraction = min(1.0, 1 / (self.config.update_hz * self.config.prediction_dt))
        shifted = (
            self.warm * (1 - fraction)
            + jnp.concatenate((self.warm[:, 1:], self.warm[:, -1:]), axis=1) * fraction
        )
        warm = jnp.where(self.valid[:, None, None], shifted, feedforward)
        started = time.perf_counter()
        controls, success, cost = self._batch_solve(
            states, self.previous, warm, references
        )
        controls.block_until_ready()
        self.last_latency_s = time.perf_counter() - started
        emergency = jax.vmap(self.model.geometric_action)(states, references[:, 0])
        emergency = jnp.nan_to_num(emergency, nan=0.0, posinf=1.0, neginf=-1.0)
        reuse = self.valid & (self.failures == 0)
        fallback = jnp.where(reuse[:, None], shifted[:, 0], emergency)
        action = jnp.clip(
            jnp.where(success[:, None], controls[:, 0], fallback), -1.0, 1.0
        )
        self.warm = jnp.where(success[:, None, None], controls, shifted)
        self.valid = self.valid | success
        self.failures = jnp.where(success, 0, self.failures + 1)
        self.total_fallbacks += ~success
        self.previous = action
        return action, {
            "mpc_success": success,
            "mpc_cost": cost,
            "mpc_fallback": ~success,
            "mpc_reused": ~success & reuse,
            "mpc_latency_s": self.last_latency_s,
        }
