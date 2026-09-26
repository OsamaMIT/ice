"""Deterministic policy traces; never writes training checkpoints."""
from __future__ import annotations
import argparse
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["course", "skill"], required=True)
    parser.add_argument("--gate-window-scale", type=float, help="Skill aperture; defaults to checkpoint curriculum scale.")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--course", default="arena_38m_stacked")
    parser.add_argument("--legacy-spawns", action="store_true",
                        help="Reproduce the old path-relative spawn bug for comparison only.")
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from a2rl_drone_training.actions import validate_checkpoint_actions
    from a2rl_drone_training.course import course_by_name
    from a2rl_drone_training.env import CrazyflowRacingEnv, SPAWN_GEOMETRY_VERSION
    from a2rl_drone_training.networks import mode_action
    from a2rl_drone_training.normalization import normalization_state_from_dict, actor_normalization_spec, normalize_actor_observation
    with args.checkpoint.open("rb") as f:
        payload = pickle.load(f)
    validate_checkpoint_actions(payload)
    cfg = payload["config"]
    course = course_by_name(args.course)
    from a2rl_drone_training.trainer import _course_fingerprint
    if payload.get("course_fingerprint") != _course_fingerprint(course):
        raise ValueError("Diagnostic course does not match checkpoint geometry")
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    class LegacySpawnEnv(CrazyflowRacingEnv):
        def _sample_initial_state(self, *values, **kwargs):
            gate_axes = self.gate_right_axes
            self.gate_right_axes = self.approach_right_axes
            try:
                return super()._sample_initial_state(*values, **kwargs)
            finally:
                self.gate_right_axes = gate_axes

    n = cfg.evaluation.num_envs if args.scenario == "course" else course.num_gates * cfg.evaluation.skill_attempts_per_gate
    obs_cfg = replace(cfg.obs, sensor_noise_scale=0.0, pnp_dropout_prob=0.0)
    env_cfg = replace(cfg.env, num_envs=n, device="cpu", reset_distribution="evaluation", auto_reset=False, artificial_time_limit_s=None, gate_window_scale=1.0)
    env_class = LegacySpawnEnv if args.legacy_spawns else CrazyflowRacingEnv
    env = env_class(env_cfg, obs_cfg, course)
    forced = None
    if args.scenario == "skill":
        scale = args.gate_window_scale
        if scale is None:
            scale = payload["curriculum_state"]["gate_window_scale"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("--gate-window-scale must be finite and positive")
        env.set_skill_audit_window_scale(scale)
        forced = jnp.repeat(jnp.arange(course.num_gates), cfg.evaluation.skill_attempts_per_gate)
    raw = env.reset(seed=args.seed + (10000 if forced is not None else 0), forced_reset_gates=forced)
    norm = normalization_state_from_dict(payload["normalization_state"], obs_cfg)
    spec = actor_normalization_spec(obs_cfg)
    @jax.jit
    def policy(raw):
        return mode_action(payload["actor_params"], normalize_actor_observation(norm, raw, spec, obs_cfg.normalization_clip, obs_cfg.normalization_epsilon), obs_cfg)
    active = np.ones(n, dtype=bool)
    records = []
    start = np.asarray(env.sim.data.states.pos[:, 0]).copy()
    start_gates = np.asarray(env.reset_gate).copy()
    endpoints = [None] * n
    events = []
    try:
        for step in range(max(1, round(args.seconds / env_cfg.dt))):
            action = jnp.where(jnp.asarray(active)[:, None], policy(raw), 0.0)
            raw, reward, terminated, truncated, info = env.step(action)
            pos = np.asarray(env.sim.data.states.pos[:, 0])
            quat = np.asarray(env.sim.data.states.quat[:, 0])
            vel = np.asarray(env.sim.data.states.vel[:, 0])
            angular = np.asarray(env.sim.data.states.ang_vel[:, 0])
            gate = np.asarray(info["gate_id"])
            offsets = np.asarray(jnp.stack(env._gate_offsets(jnp.asarray(pos), jnp.asarray(gate)), axis=-1))
            tilt = np.degrees(np.arccos(np.clip(1-2*(quat[:,0]**2+quat[:,1]**2), -1, 1)))
            passed = np.asarray(info["passed_gate"]) & active
            done = np.asarray(terminated | truncated) & active
            success = passed & (gate == start_gates) if forced is not None else np.zeros(n, dtype=bool)
            end = done | success
            for i in np.flatnonzero(passed | end):
                event = dict(env=int(i), step=step+1, seconds=(step+1)*env_cfg.dt, gate=int(gate[i])+1,
                             start_gate=int(start_gates[i])+1, passed=bool(passed[i]), strict_passed=bool(info["strict_passed_gate"][i]), pos=pos[i].tolist(), vel=vel[i].tolist(),
                             tilt_deg=float(tilt[i]), rpm=np.asarray(env.sim.data.states.rotor_vel[i,0]).tolist(), angular=angular[i].tolist(), offsets=offsets[i].tolist(), action=np.asarray(action[i]).tolist(),
                             reasons=[k for k in ("crashed", "out_of_bounds", "missed_gate", "deadline", "course_finished") if bool(info[k][i])])
                events.append(event)
                if end[i]: endpoints[i] = event
            if step % 10 == 0 or np.any(passed | end):
                records.append(dict(time=(step+1)*env_cfg.dt, pos=pos.copy(), vel=vel.copy(), quat=quat.copy(), angular=angular.copy(),
                                    action=np.asarray(action), rpm=np.asarray(env.sim.data.states.rotor_vel[:,0]),
                                    gate=gate.copy(), offsets=offsets.copy(), active=active.copy(), tilt=tilt.copy()))
            active &= ~end
            if not active.any(): break
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / "traces.npz", **{k:np.array([r[k] for r in records]) for k in records[0]})
        summary = dict(checkpoint=str(args.checkpoint), update=payload["updates"], scenario=args.scenario, seed=args.seed,
                       spawn_geometry_version=1 if args.legacy_spawns else SPAWN_GEOMETRY_VERSION,
                       course=args.course, gate_window_scale=env.gate_window_scale, config=dict(hz=env_cfg.control_hz, horizon=cfg.ppo.horizon, gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda),
                       initial_positions=start.tolist(), start_gates=start_gates.tolist(), events=events, endpoints=endpoints,
                       still_active=np.flatnonzero(active).tolist())
        summary["gate_outcomes"] = {
            str(g + 1): {
                "passes": sum(e["passed"] and e["gate"] == g + 1 for e in events),
                "strict_passes": sum(e["strict_passed"] and e["gate"] == g + 1 for e in events),
                "failures": {
                    reason: sum(e is not None and e["gate"] == g + 1 and reason in e["reasons"] for e in endpoints)
                    for reason in ("crashed", "out_of_bounds", "missed_gate", "deadline")
                },
            }
            for g in range(course.num_gates)
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(update=payload["updates"], scenario=args.scenario, events=len(events), remaining=int(active.sum()),
              ends={reason:sum(e is not None and reason in e["reasons"] for e in endpoints) for reason in ("crashed","out_of_bounds","missed_gate","deadline")},
              passes={str(g+1):sum(e["passed"] and e["gate"]==g+1 for e in events) for g in range(course.num_gates)})), flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()
