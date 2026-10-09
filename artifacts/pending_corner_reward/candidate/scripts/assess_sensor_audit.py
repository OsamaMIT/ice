"""Assess completed sensor-audit cases without changing the running experiment."""
import argparse
import json
import math
from pathlib import Path
import time


def assess(report):
    cases = report.get('cases', {})
    baseline = cases.get('clean', {}).get('summary')
    rows = []
    issues = []
    for name, case in cases.items():
        summary = case['summary']
        healthy = (summary.get('completion_rate') == 1.0
                   and summary.get('collision_rate') == 0.0
                   and summary.get('fallbacks') == 0)
        if not healthy:
            issues.append(name)
        row = dict(case=name, all_trials_passed=healthy,
                   trials=summary.get('trials'),
                   completion_rate=summary.get('completion_rate'),
                   collision_rate=summary.get('collision_rate'),
                   fallbacks=summary.get('fallbacks'))
        for metric in ('mean_lap_time_s', 'mean_tracking_error_m', 'minimum_clearance_m'):
            value = summary.get(metric)
            base = baseline.get(metric) if baseline else None
            row[metric] = value if isinstance(value, (int, float)) and math.isfinite(value) else None
            # Comparing lap times among survivors can reward failed runs.
            comparable = healthy and baseline and baseline.get('completion_rate') == 1.0
            row[metric + '_delta_from_clean'] = (value - base if comparable
                and isinstance(value, (int, float)) and isinstance(base, (int, float))
                and math.isfinite(value) and math.isfinite(base) else None)
        rows.append(row)
    status = report.get('status', 'unknown')
    if 'clean' in issues:
        action = 'Investigate the clean reference/controller failure before interpreting sensor robustness.'
    elif status in ('blocked', 'interrupted'):
        action = 'Inspect the audit error and preserve completed results; finish missing cases before selecting training conditions.'
    elif status != 'completed':
        action = 'Finish the existing sweep. Disturbance results are incomplete; defer controller changes and RL training.'
    elif not baseline or not rows or len(rows) == 1:
        action = 'Run the missing clean/disturbance comparisons before drawing a robustness conclusion.'
    elif issues:
        action = ('Repeat the failing conditions with additional independent seeds and inspect their flight traces. '
                  'For delay failures, check timestamps and delay compensation; for noise failures, investigate estimator filtering and MPC sensitivity. '
                  'Only then select a moderate, recoverable condition for a bounded residual-RL comparison.')
    else:
        action = ('MPC completed all tested conditions. Examine tracking, clearance and lap-time deltas and repeat the most degraded conditions with more seeds. '
                  'Retain MPC as the baseline; this audit does not establish a need for additional RL training.')
    return dict(source_status=status, current_case=report.get('current_case'),
                source_updated_at=report.get('updated_at'), error=report.get('error'),
                cases=rows, failing_cases=issues, next_action=action,
                limitations='Synthetic errors, nominal starts, and few trials. Clean trials are deterministic duplicates; they do not establish statistical reliability. No real-world sensor calibration, wind or opponents.')


def write_assessment(directory, result):
    destination = directory / 'assessment.json'
    temporary = directory / 'assessment.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(destination)
    lines = ['# Sensor audit assessment', '',
             'Audit status: ' + result['source_status'],
             'Current case: ' + str(result['current_case']), '', result['next_action'], '',
             '| Completed case | Trials | Completion | Collisions | Fallbacks | Lap (s) | Tracking (m) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return '-' if value is None else f'{value:.5g}'
    for row in result['cases']:
        lines.append('| ' + ' | '.join([row['case']] + [number(row[key]) for key in (
            'trials', 'completion_rate', 'collision_rate', 'fallbacks',
            'mean_lap_time_s', 'mean_tracking_error_m')]) + ' |')
    lines += ['', 'Metric differences versus clean are in assessment.json. Lap comparisons are withheld for failed cases.',
              '', result['limitations']]
    if result['error']:
        lines += ['', 'Audit error: ' + str(result['error'])]
    temporary = directory / 'assessment.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(directory / 'assessment.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=Path('artifacts/sensor_audit'))
    parser.add_argument('--watch', action='store_true', help='Refresh after each completed case; exit when the audit ends.')
    args = parser.parse_args()
    previous = None
    while True:
        content = (args.audit_dir / 'report.json').read_text(encoding='utf-8')
        report = json.loads(content)
        if content != previous:
            result = assess(report)
            write_assessment(args.audit_dir, result)
            print(f"{result['source_status']} / {result['current_case']}: {result['next_action']}", flush=True)
            previous = content
        if not args.watch or report.get('status') in ('completed', 'blocked', 'interrupted'):
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
