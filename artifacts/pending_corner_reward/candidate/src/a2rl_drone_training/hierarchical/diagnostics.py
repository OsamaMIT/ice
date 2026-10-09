"""Live flight diagnostics; partial files are observations, never checkpoints."""

import json
from pathlib import Path
import time
import numpy as np


TRACE_COLUMNS = (
    "trial", "time_s", "gates_passed", "x", "y", "z",
    "target_x", "target_y", "target_z", "motor_0", "motor_1", "motor_2", "motor_3",
    "tracking_error_m", "speed_m_s", "collision", "fallbacks",
)


class FlightDiagnostics:
    def __init__(self, directory, *, interval=1.0, save_interval=5.0, clock=time.perf_counter, profile_timing=True):
        if not np.isfinite(interval) or interval <= 0 or not np.isfinite(save_interval) or save_interval <= 0:
            raise ValueError("Diagnostic intervals must be finite and positive")
        self.profile_timing = profile_timing
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval, self.save_interval, self.clock = interval, save_interval, clock
        self.started = clock()
        self.last_log_wall = self.last_save_wall = self.started
        self.last_log_sim = self.last_save_sim = 0.0
        self.batch = None
        self.rows = []
        self.latest = {"status": "initializing", "trials": [], "timing": {}}
        print(f"Flight diagnostics: {self.directory} (initializing; first step includes compilation)", flush=True)
        self.save()

    def observe(self, *, first, times, positions, targets, motors, gates, speeds,
                collisions, fallbacks, success, active, sampled, total_gates, mpc_s, physics_s, breakdown=None):
        now = self.clock()
        elapsed = now - self.started
        sim = float(np.max(times))
        if self.batch != first:
            self.batch = first
            self.last_log_sim = self.last_save_sim = 0.0
        error = np.linalg.norm(positions - targets, axis=1)
        for i in np.flatnonzero(sampled):
            self.rows.append(np.r_[first + i, times[i], gates[i], positions[i], targets[i],
                                   motors[i], error[i], speeds[i], collisions[i], fallbacks[i]])
        trials = {r["trial"]: r for r in self.latest["trials"]}
        for i in range(len(times)):
            trials[first + i] = dict(trial=first + i, simulated_seconds=float(times[i]),
                gates_passed=int(gates[i]), total_gates=int(total_gates),
                next_gate=(int(gates[i]) + 1 if gates[i] < total_gates else None),
                tracking_error_m=float(error[i]), collision=bool(collisions[i]),
                fallbacks=int(fallbacks[i]), success=bool(success[i]), active=bool(active[i]))
        self.latest = dict(status="running", wall_seconds=elapsed, profiling_enabled=self.profile_timing,
            trials=list(trials.values()), timing=dict(mpc_seconds=float(mpc_s),
            physics_seconds=float(physics_s), other_seconds=max(0.0, elapsed - mpc_s - physics_s)))
        if breakdown:
            self.latest["timing"].update(breakdown)
            self.latest["timing"]["unclassified_seconds"] = max(
                0.0, self.latest["timing"]["other_seconds"] - sum(breakdown.values()))
        terminal = not np.any(active)
        if sim - self.last_log_sim >= self.interval - 1e-6 or now - self.last_log_wall >= 30 or terminal:
            print(f"Flight {first + 1}-{first + len(times)} | sim {sim:.1f}s | wall {elapsed:.0f}s | "
                  f"passed {int(np.min(gates))}-{int(np.max(gates))}/{total_gates}\n  "
                  f"error max {float(np.max(error)):.2f}m | collisions {int(np.sum(collisions))} | "
                  f"fallbacks {int(np.sum(fallbacks))}", flush=True)
            if self.profile_timing:
                print(f"  time: MPC {mpc_s:.1f}s | physics {physics_s:.1f}s | "
                      f"other {self.latest['timing']['other_seconds']:.1f}s", flush=True)
                if breakdown:
                    print("  other detail: " + " | ".join(
                        f"{key.removesuffix('_seconds')} {value:.1f}s" for key, value in breakdown.items()), flush=True)
            self.last_log_sim, self.last_log_wall = sim, now
        if sim - self.last_save_sim >= self.save_interval - 1e-6 or now - self.last_save_wall >= 30 or terminal:
            self.save()
            self.last_save_sim, self.last_save_wall = sim, now

    def save(self):
        # Replace each file atomically: interruption cannot truncate the previous snapshot.
        trace = np.asarray(self.rows, dtype=float).reshape(-1, len(TRACE_COLUMNS))
        with (self.directory / "trace.tmp").open("wb") as stream:
            np.savez_compressed(stream, trace=trace, columns=np.asarray(TRACE_COLUMNS))
        (self.directory / "trace.tmp").replace(self.directory / "trace.npz")
        temporary = self.directory / "progress.tmp"
        def finite(value):
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        temporary.write_text(json.dumps(finite(self.latest), indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "progress.json")

    def finish(self, status, error=None):
        self.latest["status"] = status
        self.latest["wall_seconds"] = self.clock() - self.started
        if error is not None:
            self.latest["error"] = str(error)
        self.save()
        print(f"Flight diagnostics {status}: {self.directory / 'progress.json'}", flush=True)
