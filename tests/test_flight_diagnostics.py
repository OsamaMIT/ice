import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from a2rl_drone_training.hierarchical.diagnostics import FlightDiagnostics


class FlightDiagnosticsTests(unittest.TestCase):
    def sample(self, reporter, *, first=0, sim=1.0, sampled=True, active=True):
        reporter.observe(first=first, times=np.array([sim]), positions=np.zeros((1, 3)),
            targets=np.ones((1, 3)), motors=np.zeros((1, 4)), gates=np.array([1]),
            speeds=np.zeros(1), collisions=np.array([False]), fallbacks=np.array([2]),
            success=np.array([False]), active=np.array([active]), sampled=np.array([sampled]),
            total_gates=12, mpc_s=1.0, physics_s=2.0)

    def test_wall_heartbeat_and_saved_trace_survive_interruption(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            clock = [0.0]
            reporter = FlightDiagnostics(directory, clock=lambda: clock[0])
            clock[0] = 31.0
            self.sample(reporter, sim=0.05)
            self.assertIn("sim 0.1s", output.getvalue())
            path = Path(directory)
            progress = json.loads((path / "progress.json").read_text())
            self.assertEqual(progress["timing"]["other_seconds"], 28.0)
            self.assertEqual(progress["trials"][0]["next_gate"], 2)
            with np.load(path / "trace.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["trace"].shape, (1, 17))
                self.assertEqual(saved["columns"][6], "target_x")
            self.sample(reporter, sim=0.1)
            reporter.finish("interrupted")
            with np.load(path / "trace.npz") as saved:
                self.assertEqual(len(saved["trace"]), 2)
            self.assertEqual(json.loads((path / "progress.json").read_text())["status"], "interrupted")

    def test_batches_preserve_trial_records_without_duplicate_terminal_samples(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            reporter = FlightDiagnostics(directory)
            self.sample(reporter, sim=1, active=False)
            self.sample(reporter, sim=1, sampled=False, active=False)
            self.sample(reporter, first=1, sim=0.1, active=False)
            reporter.latest["trials"][1]["tracking_error_m"] = float("nan")
            reporter.finish("completed")
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual([t["trial"] for t in progress["trials"]], [0, 1])
            self.assertIsNone(progress["trials"][1]["tracking_error_m"])
            with np.load(Path(directory) / "trace.npz") as saved:
                np.testing.assert_array_equal(saved["trace"][:, 0], [0, 1])

    def test_evaluation_interrupt_flushes_diagnostics(self):
        from a2rl_drone_training.hierarchical.evaluation import run_trials
        def interrupt(*args, diagnostics, **kwargs):
            self.sample(diagnostics, sim=0.05)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("a2rl_drone_training.hierarchical.evaluation._run_trials", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_trials(None, None, diagnostics_dir=directory)
            self.assertEqual(json.loads((Path(directory) / "progress.json").read_text())["status"], "interrupted")
            with np.load(Path(directory) / "trace.npz") as saved:
                self.assertEqual(len(saved["trace"]), 1)

    def test_intervals_reject_nonpositive_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in [0, -1, float("nan"), float("inf")]:
                with self.assertRaises(ValueError):
                    FlightDiagnostics(directory, interval=value)


if __name__ == "__main__":
    unittest.main()
