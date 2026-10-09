# Motor-policy diagnosis ? checkpoint 90

Date: 2026-09-17. Checkpoint: `checkpoints_motors/checkpoint_000090.pkl`
(5,898,240 environment steps). Original training checkpoints were not modified.
All diagnostic rollouts ran in WSL on CPU, with deterministic actions, frozen
normalization, no sensor noise/dropout, and the original 500 Hz first-principles model.

## Findings

1. **G1?G2 is a flight-control failure.** Reproduced official evaluation seed 123
   with 32 environments: 23 passed G1; all 23 then hit the ground while targeting
   G2. The other nine missed G1. No G2 passes. Episodes ended after 1.586?2.176 s.
   Example environment 0 reached 66 degrees tilt at 1.50 s and crashed at 1.818 s,
   still 5.89 m behind G2's plane. The model's maximum collective thrust is 0.48 N
   versus 0.313 N hover thrust, so static hover above about 49.3 degrees tilt is
   impossible even at maximum command. Actual thrust and transient motion also
   matter; this is a physical reference, not a complete causal model.

2. **G5 has a reset bug as well as control failures.** Reproduced the complete
   96-environment skill audit at seed 10123 (eight per gate, 1.8x openings).
   Three G5 starts were already beyond its plane: +0.224, +0.085, and +0.158 m.
   Lateral noise was perpendicular to the approach vector rather than tangent to
   the gate plane. G5's oblique approach makes the distinction significant.
   Seven G5 attempts crashed and one missed the gate; zero passed.

3. **Motor mapping checks passed.** A level zero-action drone held position over
   0.2 s (displacement below 0.0001 m). Independent +0.1 motor pulses produced the
   expected roll/pitch/yaw torque signs for all four motors. RPM bounds are about
   8,398 / 18,968 / 23,216 for minimum / hover / maximum. These checks validate the
   configured simulator model, not an actual A2RL vehicle's calibration.

4. **Credit assignment is shorter in physical time.** At 500 Hz, horizon 256 is
   0.512 s. With gamma 0.999 and lambda 0.99, the untruncated GAE weighting has an
   e-folding time of about 0.181 s (formerly 0.904 s at 100 Hz). Bootstrapping still
   carries information beyond a rollout, so this alone does not prove the cause
   of failed course linking. It is a candidate for a separate controlled experiment.

## Action taken and matched comparison

Changed lateral spawn offsets to use gate-plane right axes. Gate locations,
openings, rewards, motor commands, model weights, and PPO settings were unchanged.
All G5 starts now lie on the approach side. Replaying the same policy and seeds:

| Gate | Original spawns | Corrected spawns |
| --- | ---: | ---: |
| G1 | 7/8 | 7/8 |
| G2 | 7/8 | 8/8 |
| G3 | 8/8 | 8/8 |
| G4 | 8/8 | 8/8 |
| G5 | 0/8 | 5/8 |
| G6 | 7/8 | 5/8 |
| G7 | 8/8 | 7/8 |
| G8 | 8/8 | 8/8 |
| G9 | 8/8 | 8/8 |
| G10 | 7/8 | 7/8 |
| G11 | 8/8 | 8/8 |
| G12 | 8/8 | 8/8 |

Total local successes increased from **84/96 to 87/96**; G5 increased from
**0/8 to 5/8**. G6 and G7 regressed on these changed starting states. This is an
end-to-end reset-distribution comparison with only eight attempts per gate, not
proof of improved policy weights or generalization. No retraining was performed.
G1's spawn geometry is unchanged; the G1?G2 failure remains unresolved.

Checkpoints now record `spawn_geometry_version=2`. Resuming older motor checkpoints
clears old audit/evaluation statistics and sampling priorities, retaining phase,
weights, optimizers, normalization, and schedule counters. This avoids mixing
results from different starting-state distributions. New qualification therefore
needs fresh audit coverage.

## Validation and next run

86 regression tests passed, including oblique spawns for all gates across three
seeds, evidence reset behavior, physics, PPO, and checkpoint round-trips.
Continue from checkpoint 90 into **checkpoints_motors_spawnfix** using the README
command, with evaluation enabled. Preserve the original checkpoints for comparison.
Judge subsequent audits by G5 success and official G2 crossings, not only critic
explained variance or relaxed single-gate success.

If G1?G2 still fails, the next isolated experiment should restore the old physical
credit timescale: 64 environments ? 1024 steps keeps batch size 65,536 and 32
minibatches; gamma = 0.999**0.2 ? 0.99979992, lambda = 0.99**0.2 ? 0.99799195.
Set the potential-shaping gamma to the same gamma. This experiment has **not**
been run or validated as a performance improvement; do not assume it solves the
control problem or change several other hyperparameters simultaneously.

## Artifacts

- [Motor checks](motor_response.json)
- [Original course events](course90/summary.json), [traces](course90/traces.npz)
- [Original skill events](skill90/summary.json), [traces](skill90/traces.npz)
- [Corrected skill events](skill90_fixed/summary.json), [traces](skill90_fixed/traces.npz)

Trace arrays include time, position, velocity, quaternion, angular velocity,
action, actual motor RPM, gate index, gate-plane offsets, tilt, and active mask.
Use the active mask: rows after an episode ends are not performance evidence.

Actual checkpoint-90 resume was also verified on CPU: update 90 and 5,898,240 environment/schedule steps were preserved, and obsolete audit evidence was cleared. No training or checkpoint write occurred during this check.
