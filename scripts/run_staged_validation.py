"""Run nominal flight, disturbed starts, then a bounded residual-PPO pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def require_clean_flights(result, expected_trials, gates=12):
    import numpy as np
    success = np.asarray(result["success"])
    if success.shape != (expected_trials,) or not np.all(success == True):
        raise ValueError("Not every trial completed the course")
    for key in ("collision", "fallbacks"):
        values = np.asarray(result[key])
        if values.shape != (expected_trials,) or not np.all(values == 0):
            raise ValueError(f"Baseline contains {key}")
    passes = np.asarray(result["gate_passes"])
    if passes.shape != (expected_trials, gates) or not np.all(passes == 1):
        raise ValueError("Each trial must pass every gate exactly once")
    for key in ("lap_time", "tracking_error", "minimum_clearance"):
        values = np.asarray(result[key])
        if values.shape != (expected_trials,) or not np.isfinite(values).all():
            raise ValueError(f"Missing or nonfinite {key}")
    if np.any(result["lap_time"] <= 0) or np.any(result["tracking_error"] < 0) or np.any(result["minimum_clearance"] < 0):
        raise ValueError("Invalid flight time, tracking error, or frame clearance")


def process_token(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def verify_baseline_process(pid, reference, directory):
    proc = Path(f"/proc/{pid}")
    token = process_token(pid)
    if token is None:
        raise ValueError("Baseline process is not running")
    command = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
    if not any(Path(part).name == "a2rl-drone-evaluate" for part in command):
        raise ValueError("PID is not an evaluator")
    cwd = (proc / "cwd").resolve()
    def argument(flag):
        return command[command.index(flag) + 1]
    if (cwd / argument("--reference-path")).resolve() != reference.resolve():
        raise ValueError("Baseline uses a different reference")
    if (cwd / argument("--output-dir")).resolve() != directory.resolve():
        raise ValueError("Baseline uses a different output directory")
    expected = {"--device": "gpu", "--num-envs": "1", "--trials": "1",
                "--spawn-position-std": "0", "--spawn-velocity-std": "0", "--max-episode-time": "120"}
    if any(argument(key) != value for key, value in expected.items()) or "--exact-state-only" not in command or "--checkpoint" in command:
        raise ValueError("Baseline must be one nominal, zero-offset GPU flight")
    if (directory / "report.json").exists() or (directory / "exact_state_baseline.npz").exists():
        raise ValueError("Refusing an attached directory with existing completed results")
    return token, command


def pilot_command(reference, checkpoints):
    return [sys.executable, "-m", "a2rl_drone_training.train",
            "--controller", "residual_mpc", "--reference-path", str(reference),
            "--device", "gpu", "--gpu-memory-fraction", "0.5", "--num-envs", "4",
            "--physics", "first_principles", "--sim-hz", "500", "--policy-hz", "20",
            "--mpc-hz", "100", "--mpc-horizon", "0.5", "--mpc-prediction-dt", "0.02", "--mpc-iterations", "5",
            "--spawn-position-std", "0.02", "--spawn-velocity-std", "0.05",
            "--max-episode-time", "120", "--horizon", "64", "--minibatches", "4", "--update-epochs", "2",
            "--total-env-steps", "1024", "--schedule-env-steps", "2000000",
            "--log-interval", "1", "--checkpoint-interval", "1", "--no-evaluation",
            "--checkpoint-dir", str(checkpoints)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=Path("artifacts/arena_reference.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/staged_validation"))
    parser.add_argument("--baseline-pid", type=int, help="Attach to a verified, currently running nominal evaluator in WSL.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/feasibility_baseline_optimized"))
    args = parser.parse_args()
    os.chdir(ROOT)
    reference = args.reference_path.resolve()
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    token, command = (verify_baseline_process(args.baseline_pid, reference, args.baseline_dir)
                      if args.baseline_pid else (None, None))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = dict(reference=str(reference), reference_sha256=digest,
                  baseline_pid=args.baseline_pid, baseline_command=command,
                  disturbed_trials=4, spawn_position_std_m=0.02, spawn_velocity_std_m_s=0.05,
                  required="All trials finish 12 gates, with no collisions or MPC fallbacks",
                  pilot_updates=4, pilot_policy_transitions=1024)
    def update(stage, **extra):
        status.update(stage=stage, updated_at=time.time(), **extra)
        temporary = args.output_dir / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output_dir / "status.json")
        print(stage + (": " + str(extra) if extra else ""), flush=True)
    def unchanged():
        if hashlib.sha256(reference.read_bytes()).hexdigest() != digest:
            raise ValueError("Reference changed during validation; start a new run")
    def run(command, log):
        unchanged()
        environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with log.open("w") as stream:
            # Child has its own log; no pipe buffering can stall a long flight.
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            try:
                result = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
        if result:
            raise RuntimeError(f"Stage exited with code {result}; see {log}")
    def evaluate(directory, trials, position, velocity):
        run([str(ROOT / ".venv/bin/a2rl-drone-evaluate"), "--reference-path", str(reference),
             "--device", "gpu", "--num-envs", str(trials), "--trials", str(trials),
             "--exact-state-only", "--spawn-position-std", str(position), "--spawn-velocity-std", str(velocity),
             "--seed", "321", "--max-episode-time", "120", "--output-dir", str(directory)],
            args.output_dir / (directory.name + ".log"))
    def qualify(directory, trials):
        unchanged()
        import numpy as np
        progress = json.loads((directory / "exact_state_baseline_live/progress.json").read_text())
        if progress["status"] != "completed":
            raise ValueError("Evaluation did not finish")
        with np.load(directory / "exact_state_baseline.npz", allow_pickle=False) as result:
            require_clean_flights(result, trials)
    try:
        nominal = args.baseline_dir if args.baseline_pid else args.output_dir / "nominal"
        if args.baseline_pid:
            update("waiting_for_nominal")
            while True:
                unchanged()
                current = process_token(args.baseline_pid)
                report = nominal / "report.json"
                if report.exists():
                    try:
                        json.loads(report.read_text())
                        break
                    except json.JSONDecodeError:
                        pass
                if current != token:
                    raise RuntimeError("Attached baseline ended without a complete report")
                time.sleep(10)
        else:
            update("running_nominal")
            evaluate(nominal, 1, 0, 0)
        qualify(nominal, 1)
        update("running_disturbance_tests", nominal_passed=True)
        disturbed = args.output_dir / "disturbed"
        evaluate(disturbed, 4, 0.02, 0.05)
        qualify(disturbed, 4)
        update("running_ppo_pilot", disturbed_passed=True)
        checkpoints = args.output_dir / "pilot_checkpoints"
        if checkpoints.exists():
            raise ValueError("Pilot checkpoint directory must be new")
        run(pilot_command(reference, checkpoints), args.output_dir / "pilot.log")
        if not (checkpoints / "checkpoint_latest.pkl").is_file():
            raise RuntimeError("Pilot finished without its expected checkpoint")
        update("completed", checkpoint=str(checkpoints / "checkpoint_latest.pkl"))
    except BaseException as exc:
        update("interrupted" if isinstance(exc, KeyboardInterrupt) else "blocked", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
