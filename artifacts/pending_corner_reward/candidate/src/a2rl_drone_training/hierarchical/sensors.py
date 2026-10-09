"""Synthetic position/velocity estimator errors, sampled at the MPC rate.

Attitude, body rates and rotor states remain exact. This is not a sensor fusion
implementation or a hardware-calibrated sensor model.
"""
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np


def validate_sensor_config(config, hz):
    position = config.position_noise_std_m
    velocity = config.velocity_noise_std_m_s
    position = config.estimation_noise_std if position is None else position
    velocity = config.estimation_noise_std if velocity is None else velocity
    values = (position, velocity, config.sensor_delay_s, config.sensor_dropout_prob,
              *config.position_bias_m, *config.velocity_bias_m_s)
    if not np.isfinite(values).all() or min(position, velocity, config.sensor_delay_s) < 0:
        raise ValueError("Sensor settings must be finite; noise and delay must be nonnegative")
    if not 0 <= config.sensor_dropout_prob <= 1 or config.sensor_delay_s > 1:
        raise ValueError("Dropout must be in [0,1] and sensor delay at most one second")
    if len(config.position_bias_m) != 3 or len(config.velocity_bias_m_s) != 3:
        raise ValueError("Sensor bias must contain three axis values")
    lag = config.sensor_delay_s * hz
    if not np.isclose(lag, round(lag)):
        raise ValueError("Sensor delay must be an integer multiple of the MPC sampling period")
    return float(position), float(velocity), int(round(lag))


class StateSensor:
    def __init__(self, config, hz, physics_hz):
        self.position_std, self.velocity_std, self.lag = validate_sensor_config(config, hz)
        self.config = config
        self.stride = physics_hz // hz
        self.rng = jax.random.key(0)
        self.history = None
        self.last_tick = None
        self.estimate = None

    def reset(self, state, mask, seed=None):
        if seed is not None:
            self.rng = jax.random.key(seed + 7919)
        if self.history is None:
            self.history = deque([state] * (self.lag + 1), maxlen=self.lag + 1)
            self.estimate = state
        else:
            self.history = deque([jnp.where(mask[:, None], state, old) for old in self.history], maxlen=self.lag + 1)
            self.estimate = jnp.where(mask[:, None], state, self.estimate)
        # History starts at the initial state, so reset never leaks a previous episode.
        if bool(jnp.all(mask)):
            self.last_tick = None

    def read(self, state, tick):
        if self.last_tick == tick or (self.last_tick is not None and tick - self.last_tick < self.stride):
            return self.estimate
        if (self.lag == 0 and self.position_std == 0 and self.velocity_std == 0
                and self.config.sensor_dropout_prob == 0
                and not any(self.config.position_bias_m) and not any(self.config.velocity_bias_m_s)):
            self.estimate, self.last_tick = state, tick
            return state
        self.history.append(state)
        delayed = self.history[0]
        self.rng, noise_key, dropout_key = jax.random.split(self.rng, 3)
        scales = jnp.array([self.position_std] * 3 + [self.velocity_std] * 3)
        bias = jnp.asarray((*self.config.position_bias_m, *self.config.velocity_bias_m_s))
        measured = delayed[:, :6] + bias + jax.random.normal(noise_key, delayed[:, :6].shape) * scales
        dropped = jax.random.uniform(dropout_key, (len(state), 1)) < self.config.sensor_dropout_prob
        # Only the modeled position/velocity channels are delayed or held.
        self.estimate = state.at[:, :6].set(jnp.where(dropped, self.estimate[:, :6], measured))
        self.last_tick = tick
        return self.estimate
