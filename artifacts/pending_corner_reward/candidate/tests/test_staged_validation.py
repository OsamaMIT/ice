import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location("staged_validation", Path(__file__).resolve().parents[1] / "scripts/run_staged_validation.py")
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


def clean(trials=1):
    return dict(success=np.ones(trials, dtype=bool), collision=np.zeros(trials, dtype=bool),
                fallbacks=np.zeros(trials, dtype=int), gate_passes=np.ones((trials, 12)),
                lap_time=np.full(trials, 53.), tracking_error=np.full(trials, 0.02),
                minimum_clearance=np.full(trials, 0.1))


class StagedValidationTests(unittest.TestCase):
    def test_requires_every_gate_and_every_trial(self):
        staged.require_clean_flights(clean(4), 4)
        for key in ("success", "gate_passes"):
            result = clean(4)
            result[key].flat[-1] = 0
            with self.assertRaises(ValueError):
                staged.require_clean_flights(result, 4)
        with self.assertRaises(ValueError):
            staged.require_clean_flights(clean(1), 4)

    def test_rejects_collisions_fallbacks_and_nonfinite_metrics(self):
        for key in ("collision", "fallbacks"):
            result = clean()
            result[key][0] = 1
            with self.assertRaises(ValueError):
                staged.require_clean_flights(result, 1)
        for key in ("lap_time", "tracking_error", "minimum_clearance"):
            result = clean()
            result[key][0] = np.nan
            with self.assertRaises(ValueError):
                staged.require_clean_flights(result, 1)

    def test_pilot_command_is_valid_and_bounded(self):
        from a2rl_drone_training.train import build_parser
        command = staged.pilot_command(Path("reference.npz"), Path("fresh"))
        args = build_parser().parse_args(command[3:])
        self.assertEqual(args.controller, "residual_mpc")
        self.assertEqual(args.total_env_steps // (args.num_envs * args.horizon), 4)
        self.assertIsNone(args.restore_checkpoint)

    def run_pipeline(self, failing_stage=None):
        commands = []
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            reference.write_bytes(b"fixed reference for orchestration test")
            output = root / "run"
            class Child:
                def __init__(self, command, **kwargs):
                    self.command = command
                    commands.append(command)
                def wait(self):
                    cmd = self.command
                    if "--trials" in cmd:
                        target = Path(cmd[cmd.index("--output-dir") + 1])
                        trials = int(cmd[cmd.index("--trials") + 1])
                        live = target / "exact_state_baseline_live"
                        live.mkdir(parents=True)
                        (live / "progress.json").write_text(json.dumps(dict(status="completed")))
                        result = clean(trials)
                        if target.name == failing_stage:
                            result["success"][0] = False
                        np.savez(target / "exact_state_baseline.npz", **result)
                        (target / "report.json").write_text(json.dumps(dict(accepted=False)))
                    else:
                        target = Path(cmd[cmd.index("--checkpoint-dir") + 1])
                        target.mkdir()
                        (target / "checkpoint_latest.pkl").write_bytes(b"test checkpoint")
                    return 0
            try:
                with patch.object(staged, "ROOT", root), patch.object(staged.subprocess, "Popen", Child), patch.object(staged.sys, "argv", ["staged", "--reference-path", str(reference), "--output-dir", str(output)]):
                    if failing_stage:
                        with self.assertRaises(ValueError):
                            staged.main()
                    else:
                        staged.main()
                status = json.loads((output / "status.json").read_text())
            finally:
                os.chdir(original_cwd)
        return commands, status

    def test_nominal_failure_blocks_later_stages(self):
        commands, status = self.run_pipeline("nominal")
        self.assertEqual(len(commands), 1)
        self.assertEqual(status["stage"], "blocked")

    def test_disturbance_failure_blocks_training(self):
        commands, status = self.run_pipeline("disturbed")
        self.assertEqual(len(commands), 2)
        self.assertEqual(status["stage"], "blocked")

    def test_clean_flights_allow_only_the_small_pilot(self):
        commands, status = self.run_pipeline()
        self.assertEqual(len(commands), 3)
        self.assertEqual(status["stage"], "completed")
        self.assertTrue(status["nominal_passed"])
        self.assertTrue(status["disturbed_passed"])


if __name__ == "__main__":
    unittest.main()
