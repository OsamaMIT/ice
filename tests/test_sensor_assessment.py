import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

spec = importlib.util.spec_from_file_location('assessment', Path(__file__).resolve().parents[1] / 'scripts/assess_sensor_audit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def case(completion=1., lap=53.):
    return {'summary': dict(trials=2, completion_rate=completion, collision_rate=0.,
                            fallbacks=0, mean_lap_time_s=lap, mean_tracking_error_m=.001,
                            minimum_clearance_m=.5)}


class AssessmentTests(unittest.TestCase):
    def test_incomplete_sweep_defers_changes(self):
        result = module.assess(dict(status='running', cases={'clean': case()}))
        self.assertIn('defer', result['next_action'])

    def test_clean_failure_takes_priority(self):
        result = module.assess(dict(status='blocked', cases={'clean': case(0.)}))
        self.assertIn('clean reference/controller failure', result['next_action'])

    def test_failed_condition_does_not_get_survivor_lap_comparison(self):
        result = module.assess(dict(status='completed', cases={'clean': case(), 'delay': case(.5, 40.)}))
        self.assertEqual(result['failing_cases'], ['delay'])
        self.assertIsNone(result['cases'][1]['mean_lap_time_s_delta_from_clean'])
        self.assertIn('additional independent seeds', result['next_action'])

    def test_success_compares_metrics_without_claiming_rl_needed(self):
        result = module.assess(dict(status='completed', cases={'clean': case(), 'noise': case(lap=54.)}))
        self.assertEqual(result['cases'][1]['mean_lap_time_s_delta_from_clean'], 1.)
        self.assertIn('does not establish', result['next_action'])
        with tempfile.TemporaryDirectory() as folder:
            module.write_assessment(Path(folder), result)
            self.assertEqual(json.loads((Path(folder) / 'assessment.json').read_text()), result)
            self.assertIn('noise', (Path(folder) / 'assessment.md').read_text())

    def test_interrupted_and_missing_baseline(self):
        self.assertIn('audit error', module.assess(dict(status='interrupted', cases={}))['next_action'])
        self.assertIn('missing', module.assess(dict(status='completed', cases={'noise': case()}))['next_action'])


if __name__ == '__main__':
    unittest.main()
