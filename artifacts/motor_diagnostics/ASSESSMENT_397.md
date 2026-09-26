# Checkpoint 397: diagnosis and recommended next action

## Evidence

Compared fixed checkpoints 380 and 397 using deterministic CPU course traces,
strict 1.00x gates, frozen normalization and noise-free observations, on seed 124.
Each test used 32 starts and a 16-second horizon; every flight terminated before
the horizon. Seed 123 results below come from the saved training evaluations.

| Checkpoint | Seed | G2 passes | G3 passes | G4 passes | G5 passes |
| --- | --- | --- | --- | --- | --- |
| 380 | 123 | 27/32 | 25/32 | 11/32 | 0/32 |
| 380 | 124 | 28/32 | 27/32 | 13/32 | 0/32 |
| 397 | 123 | 24/32 | 16/32 | 5/32 | 0/32 |
| 397 | 124 | 24/32 | 17/32 | 8/32 | 0/32 |

All counts use starting flights as the denominator. Neither checkpoint completes
the course. These two seeds support preferring 380 as a continuation reference,
but do not establish general performance across all possible starts.

## What fails after G4

G4 is centered at (34,16,1); G5 is at (27,8,1). All 13 checkpoint-380 flights
and all eight checkpoint-397 flights that passed G4 in seed 124 subsequently hit
the ground while targeting G5. Their terminal altitude was below 0.05 m. The
same events also carry out-of-bounds flags because the floor is an arena bound;
these flags are not additional independent failures or evidence of wall strikes.

At checkpoint 397, G4 exit speed was 3.43-7.13 m/s, and X velocity was positive
in all eight cases (2.95-5.59 m/s), opposite the required net X displacement to G5.
The drone fell within 0.23-1.78 seconds after crossing G4. Its closest sampled
distance to G5 was still 7.77-10.77 m. Terminal tilt ranged from 27.6 to 173.8 degrees:
several flights lost attitude severely, although not all failures were inverted.
Offsets at termination are not gate crossing errors; the failures occur before
reaching G5. Positive X velocity at G4 alone is not proof of an invalid trajectory,
but the subsequent drift and ground impact show these policies fail to recover.

Inference: the primary shared bottleneck is planning and executing a viable G4
exit and G4-to-G5 turn while maintaining altitude. Aperture accuracy alone does
not explain the zero G5 passes. Deterministic evaluation also fails, so extra
random action noise during training is not by itself a solution.

## Training exposure

The completed strict-start continuation (updates 381-397, excluding repeated
abandoned resumes) records 155 G4 passes and zero G5 passes in training. Thus
there is no successful G5 transition experience and no later-gate passage practice.
Checkpoint 397's saved local diagnostic passes G5 in 5/8 strict attempts, but these
local evaluations do not update the policy. G11 remains at 2/8 strict local passes.
Several settings changed between earlier runs; this comparison cannot isolate the
causal effect of strict starts, gate narrowing, credit assignment or exploration.

## Recommended action

Preserve both checkpoints and use 380 as the reference for the next controlled
experiment because it reaches the problem corner more reliably on both seeds.
Keep 1.00x gate geometry. The highest-priority intervention is focused G4-to-G5
practice using physically consistent pre-G4 states and velocities, with success
requiring both gates, so the policy learns braking/turn preparation before G4.
Mix that practice with complete flights and enough other local practice to retain
later-gate skills. This recommendation requires relaxing the requested 100% G1
training starts; no reset configuration was changed by this diagnostic task.

If exclusively G1 starts must remain, investigate turn-aware speed/exit-direction
shaping before G4 in a separate experiment, retaining gate-order and physical
passing criteria. Treat that as a reward-design hypothesis, not a validated fix;
a broad speed penalty risks encouraging slow flight instead of a useful turn.
Do not simultaneously raise learning rates or exploration again. Judge the next
intervention by strict G4-to-G5 success, post-G4 survival and G1-to-G4 retention on
both seeds, not just episode reward or critic explained variance.

No training or policy/environment changes were performed. Artifacts:
- course380_seed124/summary.json and traces.npz
- course397_seed124/summary.json and traces.npz
