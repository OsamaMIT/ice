# Sensor audit assessment

Audit status: running
Current case: velocity_2cms

Finish the existing sweep. Disturbance results are incomplete; defer controller changes and RL training.

| Completed case | Trials | Completion | Collisions | Fallbacks | Lap (s) | Tracking (m) |
|---|---:|---:|---:|---:|---:|---:|
| clean | 2 | 1 | 0 | 0 | 53.078 | 0.00076126 |
| position_1cm | 2 | 1 | 0 | 0 | 53.087 | 0.0036426 |
| position_3cm | 2 | 1 | 0 | 0 | 53.126 | 0.0080589 |

Metric differences versus clean are in assessment.json. Lap comparisons are withheld for failed cases.

Synthetic errors, nominal starts, and few trials. Clean trials are deterministic duplicates; they do not establish statistical reliability. No real-world sensor calibration, wind or opponents.
