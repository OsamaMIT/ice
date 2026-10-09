"""Check the configured simulator motor mapping in level free flight."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from a2rl_drone_training.config import RacingEnvConfig
from a2rl_drone_training.env import CrazyflowRacingEnv

def main():
    env = CrazyflowRacingEnv(RacingEnvConfig(num_envs=5, auto_reset=False))
    try:
        states = env.sim.data.states
        env.sim.data = env.sim.data.replace(states=states.replace(
            pos=jnp.tile(jnp.array([18., 20., 2.]), (5, 1, 1)),
            vel=jnp.zeros_like(states.vel), ang_vel=jnp.zeros_like(states.ang_vel),
            quat=jnp.tile(jnp.array([0., 0., 0., 1.]), (5, 1, 1)),
            rotor_vel=jnp.full_like(states.rotor_vel, env.motor_rpm_hover)))
        actions = jnp.concatenate([jnp.zeros((1,4)), jnp.eye(4)*0.1])
        for _ in range(100):
            env.step(actions)
        omega = np.asarray(env.sim.data.states.ang_vel[:,0])
        expected = np.asarray(env.sim.data.params.mixing_matrix).T
        actual = np.sign(omega[1:])
        hover_position = np.asarray(env.sim.data.states.pos[0,0])
        result = dict(hover_position=hover_position.tolist(), hover_angular_velocity=omega[0].tolist(),
                      motor_angular_velocities=omega[1:].tolist(), expected_torque_signs=expected.tolist(),
                      signs_match=bool(np.array_equal(actual, np.sign(expected))),
                      hover_error_m=float(np.linalg.norm(hover_position-[18.,20.,2.])),
                      rpm_bounds=[float(env.motor_rpm_min),float(env.motor_rpm_hover),float(env.motor_rpm_max)],
                      collective_thrust_max_n=env.thrust_max, hover_thrust_n=env.thrust_hover,
                      max_static_hover_tilt_deg=float(np.degrees(np.arccos(env.thrust_hover/env.thrust_max))))
        path=Path("artifacts/motor_diagnostics/motor_response.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        assert result["hover_error_m"] < 1e-4 and result["signs_match"]
    finally:
        env.close()


if __name__ == "__main__":
    main()
