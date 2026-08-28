# A2RL Racing Drone Training

This package trains an autonomous FPV racing policy with Crazyflow simulation and a plain-JAX PPO implementation. The default setup targets CPU training and the 38 m A2RL arena course.

## System Shape

- Environment: vectorized Crazyflow `Sim` in attitude-control mode.
- Policy action: normalized `[throttle, roll, pitch, yaw_rate]` in `[-1, 1]`.
- Crazyflow command: `[roll, pitch, yaw, collective_thrust]`.
- Actor: gate-conditioned deployment policy using noisy estimator and gate-PnP-compatible features.
- Critic: asymmetric value network using exact, noise-free simulator state during training only.
- Trainer: 256-step rollouts, truncation-correct GAE, clipped PPO, scheduled learning rates and entropy, and KL early stopping.

Crazyflow's attitude controller accepts a yaw angle rather than a yaw-rate target. The adapter therefore integrates the policy's yaw-rate action. The current yaw error is included in the actor observation, and the integrated target is reset to measured yaw at every environment reset. This keeps the controller state observable without changing the deployed action interface.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The optional Warp backend is not required for the configured JAX/Crazyflow CPU path.

## Recommended Staged Training

Start with a two-million-step acceptance stage while keeping every PPO schedule on
the full twenty-million-step clock:

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --num-envs 64 \
  --horizon 256 \
  --minibatches 8 \
  --update-epochs 4 \
  --gamma 0.999 \
  --gae-lambda 0.99 \
  --total-env-steps 2000000 \
  --schedule-env-steps 20000000 \
  --course arena_38m_stacked \
  --physics so_rpy_rotor_drag \
  --sim-hz 500 \
  --control-hz 100
```

After reviewing the acceptance metrics, resume the same schedule:

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --total-env-steps 20000000 \
  --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints/checkpoint_latest.pkl
```

Continue only when PPO values remain finite, at least 1,000 reset events are within
`80% +/- 3%` local starts, linked G12 crossings are present, no recent active-window
gate is below 50%, and the minimum recent rate is moving toward 80%. Also verify
that sampled-action saturation and the sampled/mean action gap trend down with the
scheduled exploration ceiling.

For faster CPU iteration at lower simulation fidelity:

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --num-envs 64 \
  --horizon 256 \
  --minibatches 8 \
  --update-epochs 4 \
  --physics so_rpy_rotor \
  --sim-hz 200 \
  --control-hz 100 \
  --no-evaluation
```

Training logs use an aligned SB3-style table. The `time/` section reports current and run-average FPS, iteration time, elapsed time, and ETA; the remaining sections retain every reward component, PPO diagnostic, curriculum state, and per-gate metric.

Checkpoints default to `checkpoints/`, including atomic numbered saves and `checkpoint_latest.pkl`. Structured update records are appended to `checkpoints/metrics.jsonl`. Use `--no-checkpoint` for disposable benchmark runs.

## Reward V2

`reward_version=v2` is the default. It combines:

- Potential-based progress along a continuous course coordinate, scaled to about `+1.25` shaping return per gate.
- Gate-centering pressure localized near the active gate plane.
- A curriculum-scaled physical time cost, reaching `-0.2 reward/second` in racing phase D.
- Squared action-change cost with a default coefficient of `0.001`, rather than raw throttle or action magnitude cost.
- `+6` gate pass and `+25` true full-course finish rewards.
- `-8` missed-gate and competition-deadline penalties.
- `-12` crash or out-of-bounds penalties.
- A safety-margin penalty only when crossing clearance is below `vehicle_radius + k * position_uncertainty`.

The gate-margin penalty is capped at `6` by default, so a crossing on or outside a
physical inner edge can cancel the `+6` relaxed-window gate reward. Curriculum-scaled
dimensions decide whether Phase-A training advances; physical 1.0x dimensions decide
strict-pass telemetry, clearance, and margin reward.

There is no raw speed reward, exact-center bonus, overlapping distance/lookahead shaping, or stall reward counter in v2. Use `--reward-version v1` to reproduce the legacy reward formula for ablations.

The existing `--time-penalty` option now controls the full phase-D v2 cost in reward per physical second. Phase A starts at zero, phases B and C use configurable fractions, and phase D ramps to the configured value.

The progress potential is rebased by the episode's reset gate, so local starts retain
continuous gate-to-gate shaping without a larger negative offset at later gates.
Forward motion is capped at one segment per transition, while up to half a segment of
backtracking remains visible to the reward instead of being flattened at the segment
start. A local episode that reaches gate 12 terminates successfully but receives no
`+25` finish bonus; that bonus and course-completion credit are reserved for episodes
that began at gate 1. Reward V1 retains its legacy behavior.

## Curriculum

Curriculum advancement is driven by deterministic evaluation, not environment steps
or a selected best rollout. Reset mixtures are sampled from the environments that
reset on each event, with stochastic rounding for small reset batches.

| Phase | Gate-1 starts | Local starts | Gate window | Time cost |
| --- | ---: | ---: | ---: | ---: |
| A: gate skill | 20% | 80% stratified | 1.8x | 0% |
| B: course linking | 50% | 50% stratified | toward 1.3x | 25% |
| C: reliable course | 80% | 20% prioritized | toward 1.0x | 50% |
| D: racing | 80% | 20% prioritized | 1.0x | ramps to 100% |

Phase A qualification uses a separate deterministic, noise-free local skill audit with
eight fixed-seed attempts per runtime gate. The audit uses the active Phase-A opening,
while recording physical strict passes from the same crossings. Qualification uses
the last ten audits, requires at least 32 samples per gate, an approximately 80%
minimum active-window pass rate, and two consecutive qualifying audits. Lifetime
rates remain diagnostics only, so old failures cannot permanently lock the phase.
After coverage, Phase-A direct local starts mix 50% uniform coverage with 50% recent
audit-failure priority. Half of local starts are additionally allocated to predecessor
gates for weak strict-evaluation gates, training the transition into the failure rather
than repeatedly spawning at it. This direct/link blend is configurable. Phase B
additionally requires roughly 50% official full-course
completion and 85% minimum strict gate pass rate. Phase C requires roughly 80%
completion and 90% minimum strict pass rate. Hysteresis and rate-limited
gate-window/time-cost changes prevent rapid transitions.

Prioritized local starts use per-gate failure rates with configurable exponent, epsilon, and a minimum probability floor. Stacked top/bottom openings are separate runtime gates and therefore retain independent statistics and sampling probabilities.

Use `--no-curriculum` for strict gate-1 starts, a 1.0x gate opening, and the full configured time cost.

## Observations

The actor receives only deployment-compatible features:

- Body gyro and specific force.
- Attitude quaternion, body velocity, and body gravity.
- Previous action, course progress, remaining competition time, integrated yaw error, and legacy-v1 stall state.
- Relative gate pose, normal, image-plane bearing, visibility, and distance for the next `N` gates.

Noise is applied in physical units per sensor/feature before normalization. `--sensor-noise-scale` scales all configured noise models; the legacy `--obs-additive-noise-std` spelling is retained as an alias for this scale. PnP dropout masks all PnP-derived geometry. Running normalization is used only for unbounded channels; bounded quaternions, normals, visibility, progress, and controller channels use fixed transforms.

The critic separately receives exact pose, attitude, velocity, angular velocity, acceleration, active and next gate geometry, elapsed time, gate progress, previous action, yaw error, gate-window scale, and reset-start type. Those features never enter the actor network or actor loss.

Normalization statistics are updated during training, frozen during evaluation, and stored in checkpoints.

## Episode Semantics

True terminals are crashes, out-of-bounds failures, missed gates, local segment completion, true course completion, and the real competition deadline set by `--max-episode-time`. Local segment completion is a successful terminal but not a full-course finish. `--artificial-time-limit` is an optional simulator truncation.

GAE bootstraps through truncations from the final pre-reset privileged observation. It does not bootstrap through true terminals, and it never uses an automatically reset initial observation as the final state. The PPO rollout boundary by itself is neither a terminal nor a truncation.

## PPO Defaults

- 64 environments, 256-step rollouts, batch size 16,384.
- 4 epochs and 8 minibatches.
- `gamma=0.999`, `gae_lambda=0.99`.
- Policy and value clipping `0.2`.
- Actor and critic learning rates linearly decay from `3e-4` to `3e-5` over the independent `--schedule-env-steps` budget, default 20 million steps.
- Entropy decays from `0.003` to `0.0003` over 75% of that schedule budget.
- Target KL `0.015`; remaining epochs stop after an over-target epoch.
- Global gradient clipping `0.5`.
- Independent trainable exploration standard deviation for each action, constrained by
  a ceiling that cools from `0.60` to `0.20` over 75% of the schedule and a `0.08`
  floor.
- A small positive-advantage mean-action alignment loss keeps deterministic evaluation
  behavior close to successful sampled behavior.

The CLI exposes rollout length, gamma, lambda, all schedule endpoints, KL target,
clipping, exploration bounds, curriculum thresholds, evaluation cadence, and reward-v2
coefficients for ablations. `--skill-qualification-window` controls the rolling audit
horizon, `--phase-a-priority-mix` controls the uniform/failure-priority blend, and
`--local-link-start-mix` controls predecessor-link replay.

## Evaluation

Strict evaluation always:

- Starts every environment at gate 1.
- Uses the configured fixed evaluation seed.
- Uses a strict 1.0x opening.
- Disables sensor noise and PnP dropout.
- Freezes actor normalization.
- Uses deterministic policy actions.

The Phase-A skill audit is separate from official evaluation. It uses forced local gate
starts at the active Phase-A opening solely for curriculum qualification and never
contributes to gate-1 course-completion statistics. Official evaluation remains
gate-1-only and strict 1.0x.

The chase-video script visualizes a fixed environment index, default `0`; it never chooses the best parallel environment.

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_chase_video.py \
  --checkpoint-dir checkpoints \
  --course arena_38m_stacked \
  --output artifacts/eval_chase_arena_38m_stacked.mp4 \
  --num-eval-envs 32 \
  --visualization-env 0 \
  --seed 123 \
  --device cpu
```

## Checkpoints

Schema-v5 checkpoints include actor and critic parameters, optimizer state, actor
normalization statistics, curriculum phase, official and skill-audit counters, rolling
audit history, latest strict-evaluation gate results and priority state, consumed PPO
schedule steps, evaluation counters,
total environment steps, update count, episode count, RNG state, and a signed course
geometry fingerprint.

The actor observation grew to include remaining time and yaw error, and the critic now has a separate privileged input dimension. Checkpoints from the earlier symmetric 51-dimensional architecture cannot be restored into this version. Reward-v1 ablations should start fresh or use checkpoints created with this observation architecture.

Resume with `--restore-checkpoint`; `--total-env-steps` remains the final stopping
target while `--schedule-env-steps` controls LR and entropy decay independently.
Older checkpoints migrate their saved schedule fraction back to consumed schedule
steps and initialize missing rolling-audit state empty. A checkpoint with a known,
mismatched course fingerprint is rejected. Pre-correction Reward-V2 checkpoints
without a fingerprint remain loadable for evaluation or Reward-V1 ablations, but are
blocked from resuming corrected Reward-V2 training.

## Tests And Benchmark

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/benchmark_cpu_training.py --updates 3
```
