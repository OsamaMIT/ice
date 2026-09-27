"""Paired held-out evaluation; performance claims require measured evidence."""

from dataclasses import replace
import time
import jax.numpy as jnp
import numpy as np

from .environment import ResidualMPCEnv, residual_training_config


def run_trials(
    config, course, *, trials=100, seed=123, policy=None, record_trace=False
):
    if trials < 1:
        raise ValueError("trials must be positive")
    config = residual_training_config(config)
    batch_size = min(trials, config.env.num_envs)
    records = []
    traces = []
    latencies = []
    started = time.perf_counter()
    for first in range(0, trials, batch_size):
        size = min(batch_size, trials - first)
        cfg = replace(
            config,
            env=replace(
                config.env,
                num_envs=size,
                auto_reset=False,
                artificial_time_limit_s=None,
            ),
        )
        env = ResidualMPCEnv(cfg, course=course)
        try:
            # Trial IDs are stable for a fixed evaluation batch size, and the
            # same seeds/batches are reused by nominal and learned controllers.
            obs = env.reset(seed=seed + first)
            active = np.ones(size, dtype=bool)
            success = np.zeros(size, dtype=bool)
            elapsed = np.full(size, np.nan)
            collisions = np.zeros(size, dtype=bool)
            fallbacks = np.zeros(size, dtype=int)
            clearance = np.full(size, np.inf)
            error_sum = np.zeros(size)
            samples = np.zeros(size)
            passes = np.zeros((size, course.num_gates))
            gate_clearances = np.zeros_like(passes)
            gate_times = np.zeros_like(passes)
            for _ in range(cfg.env.max_episode_steps):
                action = jnp.zeros((size, 4)) if policy is None else policy(obs)
                obs, _, term, trunc, info = env.step(action)
                x = np.asarray(env.model.pack(env.base.sim.data.states))
                t = env.reference.project(
                    jnp.asarray(x[:, :3]) - env.offset[:, :3], env.base.gate_counter
                )
                reference, _, _ = env.reference.sample(t)
                error_sum += active * np.linalg.norm(
                    x[:, :3] - np.asarray(reference[:, :3]), axis=1
                )
                samples += active
                physical_clearance = np.asarray(info["minimum_frame_clearance"])
                clearance = np.where(
                    active, np.minimum(clearance, physical_clearance), clearance
                )
                collisions |= active & np.asarray(info["frame_collision"])
                fallbacks += active * np.asarray(info["mpc_fallback_count"], dtype=int)
                passes += np.asarray(info["gate_pass_events"]) * active[:, None]
                gate_clearances += (
                    np.asarray(info["gate_clearance_events"]) * active[:, None]
                )
                gate_times += (
                    np.asarray(info["gate_segment_time_events"]) * active[:, None]
                )
                completed = np.asarray(term | trunc) & active
                success |= np.asarray(info["course_finished"]) & completed
                elapsed = np.where(completed, np.asarray(env.lengths), elapsed)
                latencies.extend(env.last_controller_metrics["mpc_latencies_s"])
                if record_trace and first == 0:
                    traces.append(
                        np.r_[
                            float(env.lengths[0]),
                            x[0, :3],
                            np.linalg.norm(x[0, 3:6]),
                            physical_clearance[0],
                            bool(np.asarray(info["frame_collision"])[0]),
                        ]
                    )
                active &= ~completed
                if not active.any():
                    break
            records.append(
                dict(
                    success=success,
                    lap_time=elapsed,
                    collision=collisions,
                    fallbacks=fallbacks,
                    minimum_clearance=clearance,
                    tracking_error=error_sum / np.maximum(samples, 1),
                    gate_passes=passes,
                    gate_clearances=gate_clearances,
                    gate_times=gate_times,
                    physics_steps=np.array([env.physics_steps_total]),
                )
            )
        finally:
            env.close()
    result = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    result["trace"] = np.asarray(traces).reshape(-1, 7)
    result["latency_s"] = np.asarray(latencies)
    result["wall_seconds"] = time.perf_counter() - started
    result["seed"] = seed
    return result


def trainer_metrics(result):
    passes = result["gate_passes"].sum(axis=0)
    attempts = np.r_[len(result["success"]), passes[:-1]]
    return dict(
        episodes=len(result["success"]),
        finishes=int(result["success"].sum()),
        completion_rate=float(result["success"].mean()),
        gate_attempts=attempts,
        gate_passes=passes,
        gate_pass_rates=np.divide(
            passes, attempts, out=np.zeros_like(passes), where=attempts > 0
        ),
        gate_clearance=np.divide(
            result["gate_clearances"].sum(axis=0),
            passes,
            out=np.zeros_like(passes),
            where=passes > 0,
        ),
        gate_segment_time=np.divide(
            result["gate_times"].sum(axis=0),
            passes,
            out=np.zeros_like(passes),
            where=passes > 0,
        ),
        seed=result["seed"],
        mpc_fallbacks=int(result["fallbacks"].sum()),
    )


def summarize(result):
    valid = result["success"] & np.isfinite(result["lap_time"])
    times = result["lap_time"][valid]
    return dict(
        trials=len(valid),
        completion_rate=float(valid.mean()),
        collision_rate=float(result["collision"].mean()),
        mean_lap_time_s=float(times.mean()) if len(times) else None,
        median_lap_time_s=float(np.median(times)) if len(times) else None,
        lap_time_std_s=float(times.std()) if len(times) else None,
        minimum_clearance_m=(
            float(np.min(result["minimum_clearance"]))
            if np.isfinite(result["minimum_clearance"]).any()
            else None
        ),
        mean_tracking_error_m=float(np.mean(result["tracking_error"])),
        fallbacks=int(result["fallbacks"].sum()),
        mpc_latency_median_s=float(np.median(result["latency_s"])),
        mpc_latency_p95_s=float(np.quantile(result["latency_s"], 0.95)),
        physics_steps_per_second=float(
            result["physics_steps"].sum() / max(result["wall_seconds"], 1e-9)
        ),
    )


def compare_trials(baseline, learned, seed=321):
    if len(baseline["success"]) != len(learned["success"]):
        raise ValueError("Paired comparisons need equal trial counts")
    paired = (
        baseline["success"]
        & learned["success"]
        & np.isfinite(baseline["lap_time"])
        & np.isfinite(learned["lap_time"])
    )
    difference = baseline["lap_time"][paired] - learned["lap_time"][paired]
    low = high = None
    if len(difference) >= 2:
        rng = np.random.default_rng(seed)
        estimates = np.mean(
            rng.choice(difference, size=(10000, len(difference)), replace=True), axis=1
        )
        low, high = map(float, np.quantile(estimates, [0.025, 0.975]))
    return dict(
        baseline=summarize(baseline),
        learned=summarize(learned),
        paired_successes=int(paired.sum()),
        mean_improvement_s=float(difference.mean()) if len(difference) else None,
        improvement_95pct_ci_s=[low, high],
        accepted=bool(
            len(paired) >= 100
            and learned["success"].mean() >= 0.95
            and low is not None
            and low > 0
        ),
        failures={
            "baseline": np.flatnonzero(~baseline["success"]).tolist(),
            "learned": np.flatnonzero(~learned["success"]).tolist(),
        },
    )


def plot_comparison(reference, course, baseline, learned, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(221, projection="3d")
    ax.plot(*reference.states[:, :3].T, label="Planned")
    for name, result in [("Zero-offset MPC", baseline), ("Residual RL", learned)]:
        if result is None or not len(result["trace"]):
            continue
        trace = result["trace"]
        ax.plot(*trace[:, 1:4].T, label=name)
        collision = trace[:, 6] > 0
        ax.scatter(*trace[collision, 1:4].T, color="red", marker="x")
    ax.scatter(*course.centers.T, color="black", s=10, label="Gates")
    ax.legend()
    for panel, column, label in [
        (222, 4, "Speed (m/s)"),
        (223, 5, "Frame clearance (m)"),
    ]:
        axis = fig.add_subplot(panel)
        for name, result in [("Zero-offset MPC", baseline), ("Residual RL", learned)]:
            if result is not None and len(result["trace"]):
                axis.plot(result["trace"][:, 0], result["trace"][:, column], label=name)
        if column == 4:
            axis.plot(
                reference.time,
                np.linalg.norm(reference.states[:, 3:6], axis=1),
                label="Planned",
                linestyle="--",
            )
        else:
            axis.axhline(0, color="red", linestyle="--")
        axis.set(xlabel="Time (s)", ylabel=label)
        axis.legend()
    axis = fig.add_subplot(224)
    for name, result in [("Zero-offset MPC", baseline), ("Residual RL", learned)]:
        if result is not None:
            times = result["lap_time"][result["success"]]
            if len(times):
                axis.hist(times, alpha=0.5, label=name)
    axis.set(xlabel="Successful lap time (s)", ylabel="Trials")
    if axis.has_data():
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
