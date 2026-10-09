# Update 306 follow-up diagnostics (2026-09-19)

Source: `checkpoints_motors_spawnfix/checkpoint_000306.pkl` (the latest checkpoint
at diagnosis time). Diagnostics use deterministic policy actions, frozen observation
normalization, zero sensor noise/dropout and corrected spawn geometry on CPU.
They do not train the policy or write source checkpoints.

| Diagnostic | Seed | Gate scale | G1 passes | G2 passes | G6 local active / strict |
| --- | --- | --- | --- | --- | --- |
| Local audit, 8 starts/gate | 10123 | 1.40 | 8/8 | 8/8 | 7/8 active, 6/8 strict |
| Course, 32 starts | 123 | 1.00 | 32/32 | 0/32 | not reached |
| Course, 32 starts | 124 | 1.00 | 32/32 | 0/32 | not reached |

All episodes ended before the diagnostic horizon (6 s local, 8 s course).
Seed 123: 30 misses while targeting G2, two crashes also flagged out of bounds.
Seed 124: all 32 missed G2. Failure-reason counts overlap for crashes/out-of-bounds.
G6 local has one miss; it is no longer the dominant failure seen in update 210.
G5 and G11 local still have weaknesses, so local practice should be retained.
Terminal offsets are sampled at episode end, not interpolated gate-crossing positions;
consult the trajectories before treating those offsets as exact crossing errors.

## Action taken

Increase Phase-A G1 starts from 20% to 40%, retaining 60% local starts, existing
local failure/predecessor weighting, and the 1.40 aperture. This directly increases
exposure to the failing G1-to-G2 transition. It is an intervention to evaluate, not
evidence that training will improve. Existing PPO, rewards and physics remain.
The README provides a separate continuation directory and a 22-million-step target:
the source checkpoint already finished the original 20-million-step budget.
Learning-rate and exploration schedules remain consumed, without a reset.

Qualification now uses rolling audits of one aperture only and cannot advance
Phase A on transitional apertures. Old unlabelled rolling evidence is cleared on
resume; lifetime counts, policy, optimizer, phase and schedules are preserved.
Training, course-evaluation and skill-audit wall times are reported separately.

The longer-credit variant is documented for a subsequent controlled comparison,
not launched concurrently with this reset-mixture intervention. No long training
run was started. These fixed-seed diagnostics establish a reproducible bottleneck,
not a general success-rate estimate across all possible starts.

## Artifacts

- `current_skill/summary.json`, `current_skill/traces.npz`
- `current_course/summary.json`, `current_course/traces.npz`
- `current_course_seed124/summary.json`, `current_course_seed124/traces.npz`
