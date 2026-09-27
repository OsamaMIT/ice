# A2RL Racing Drone Training

This package trains an autonomous FPV racing policy with Crazyflow simulation and a plain-JAX PPO implementation. The default setup targets CPU training and the 38 m A2RL arena course.

## System Shape

- Environment: vectorized Crazyflow `Sim` in direct rotor-velocity control mode.
- Policy action: normalized `[motor_1, motor_2, motor_3, motor_4]` in `[-1, 1]`.
- Crazyflow command: `[rpm_1, rpm_2, rpm_3, rpm_4]` in the simulator model's native motor order.
- Actor: gate-conditioned deployment policy using noisy estimator and gate-PnP-compatible features.
- Critic: asymmetric value network using exact, noise-free simulator state during training only.
- Trainer: 256-step rollouts, truncation-correct GAE, clipped PPO, scheduled learning rates and entropy, and KL early stopping.

Each action independently commands one motor: `-1` maps to its minimum RPM,
`0` to hover RPM, and `+1` to maximum RPM, with linear interpolation on either
side of hover. RPM bounds come from the configured model's per-motor thrust limits
and quadratic thrust curve. This bypasses attitude and force/torque controllers;
first-principles physics models the resulting forces, torques, and rotor dynamics.
Resets initialize rotor speeds to hover. The former yaw-error observation channel
is reserved and always zero; actor/critic dimensions remain 54/37.

Direct motor control requires `--physics first_principles`. The default simulation
and action rates are both 500 Hz. A 256-step rollout now covers 0.512 seconds;
discounting and training schedules are still expressed in steps. The action-change
penalty defaults to 0.0002 to preserve its nominal per-second budget at the higher
action rate (previously 0.001 at 100 Hz). Existing attitude-control checkpoints are
incompatible even though both interfaces have four outputs; start fresh in a new
checkpoint directory, such as `--checkpoint-dir checkpoints_motors`.

WSL Ubuntu, with evaluation enabled for curriculum progression:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --checkpoint-dir checkpoints_motors
```


## Install

Run commands from the repository root. Use the examples for your shell: Bash on
Linux, or Command Prompt (`cmd.exe`) on Windows (including a Command Prompt tab in Windows Terminal).
Command Prompt uses a caret (`^`) for line continuation; it must be the last
character on the line, with no trailing spaces.

Linux (Bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows (Command Prompt):

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The Windows examples use the virtual environment's Python directly, so activation
is unnecessary. The setup command assumes Python 3.11 is installed and available
through the `py` launcher.

The optional Warp backend is not required for the configured JAX/Crazyflow CPU path.

## GPU Setup And RTX 5050 Laptop Preset

JAX CUDA runs on Linux; Windows users need WSL2 (listed as experimental by
[JAX](https://docs.jax.dev/en/latest/installation.html)). The native Windows
commands elsewhere in this README are for CPU training. Keep the Windows NVIDIA
driver installed; WSL uses that driver, as described in the
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

If WSL is not installed, run this once in an administrator Command Prompt, then
restart if prompted and complete Ubuntu's first-launch setup:

```bat
wsl --install -d Ubuntu
```

From Command Prompt in the repository root, open Ubuntu at the same location:

```bat
wsl -d Ubuntu
```

Inside Ubuntu, create a separate Linux environment (do not reuse the Windows
`.venv`) and install the project together with CUDA-enabled JAX:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install --upgrade pip
.venv-wsl/bin/python -m pip install -e . "jax[cuda13]"
.venv-wsl/bin/python -m pip check
.venv-wsl/bin/python -c "import jax; print(jax.devices('gpu'))"
exit
```

The final check must list a GPU. CUDA 13 is the current JAX installation path;
see the linked JAX guide for driver requirements. Dependency resolution and GPU
execution must succeed before starting a long run.

Back in Command Prompt, launch the acceptance stage through WSL:

```bat
wsl -d Ubuntu -- .venv-wsl/bin/python -m a2rl_drone_training.train ^
  --profile rtx-5050 ^
  --total-env-steps 2000000 ^
  --schedule-env-steps 20000000 ^
  --course arena_38m_stacked
```

The preset is a starting point for the 8 GB laptop GPU, not a measured optimum:

| Setting | Value | Purpose |
| --- | --- | --- |
| Device | GPU | Require CUDA for simulation and model allocations |
| Parallel environments | 256 | Amortize Python dispatch across more simulated drones |
| Rollout horizon | 256 | Retain the existing temporal rollout length |
| Minibatches | 32 | Keep 2,048 samples per minibatch, matching CPU defaults |
| GPU memory preallocation | 60% | Leave room for the laptop display and other applications |

Explicit flags override the preset, for example `--num-envs 128 --minibatches 16`.
`--gpu-memory-fraction` controls JAX's allocation pool, not a hard limit on total
process GPU memory; see [JAX memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
Physics uses the first-principles motor model at 500 Hz, and PPO retains four epochs and float32
networks. The larger rollout batch changes update frequency per environment step;
evaluate learning quality as well as throughput. Evaluation and checkpoint intervals
are still measured in updates.

Compare 64, 128, 256, and 512 environments on the actual laptop:

```bat
wsl -d Ubuntu -- .venv-wsl/bin/python scripts/benchmark_cpu_training.py ^
  --device gpu ^
  --warmup-updates 2 ^
  --updates 5
```

Despite its historical filename, the benchmark supports both CPU and GPU. It
excludes warmup updates, synchronizes optimizer completion, and disables evaluation
and checkpoints to compare training throughput at the same physics fidelity.
GPU cases use a fixed environment-count sweep; `--num-envs` configures CPU cases.
Use the fastest count that fits in memory, then verify the acceptance metrics with
evaluation enabled. The environment still has a Python rollout loop and synchronizes
on episode resets, so higher GPU utilization and a speedup are not guaranteed.

## Recommended Staged Training

Start with a two-million-step acceptance stage while keeping every PPO schedule on
the full twenty-million-step clock:

Linux (Bash):

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
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500
```
```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --no-evaluation
```

Windows (Command Prompt):

```bat
python -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --gamma 0.999 ^
  --gae-lambda 0.99 ^
  --total-env-steps 2000000 ^
  --schedule-env-steps 20000000 ^
  --course arena_38m_stacked ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500
```

After reviewing the acceptance metrics, resume the same schedule:

Linux (Bash):

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --total-env-steps 20000000 \
  --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints/checkpoint_latest.pkl
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --total-env-steps 20000000 ^
  --schedule-env-steps 20000000 ^
  --restore-checkpoint checkpoints/checkpoint_latest.pkl
```

Continue only when PPO values remain finite, at least 1,000 reset events are within
the configured local-start target (default `60% +/- 3%`), linked G12 crossings are present, no recent active-window
gate is below 50%, and the minimum recent rate is moving toward 80%. Also verify
that sampled-action saturation and the sampled/mean action gap trend down with the
scheduled exploration ceiling.

For CPU training without evaluation (benchmarking only):

Linux (Bash):

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --num-envs 64 \
  --horizon 256 \
  --minibatches 8 \
  --update-epochs 4 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --no-evaluation
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500 ^
  --no-evaluation
```

```bat
python -m a2rl_drone_training.train ^
  --device gpu ^
  --profile rtx-5050 ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500 ^
  --no-evaluation
```

Training logs use compact progress tables and stage messages. The table shows training progress, separate training/evaluation times, basic PPO checks, and clearly labelled gate results. Detailed reward components, PPO diagnostics, and curriculum statistics remain in `metrics.jsonl`.

Tables automatically fit the terminal width, including after resizing the window.
Long labels and values are shortened with `...` to keep columns aligned without
line wrapping. Full diagnostic values remain available in `metrics.jsonl`.

Checkpoints default to `checkpoints/`, including atomic numbered saves and `checkpoint_latest.pkl`. Structured update records are appended to `checkpoints/metrics.jsonl`. Use `--no-checkpoint` for disposable benchmark runs.

## Reward V2

`reward_version=v2` is the default. It combines:

- Potential-based progress along a continuous course coordinate, scaled to about `+1.25` shaping return per gate.
- Gate-centering pressure localized near the active gate plane.
- A curriculum-scaled physical time cost, reaching `-0.2 reward/second` in racing phase D.
- Squared action-change cost with a default coefficient of `0.0002`, rather than raw throttle or action magnitude cost.
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
| A: gate skill | 40% | 60% stratified | 1.4x | 0% |
| B: course linking | 50% | 50% stratified | toward 1.3x | 25% |
| C: reliable course | 80% | 20% prioritized | toward 1.0x | 50% |
| D: racing | 80% | 20% prioritized | 1.0x | ramps to 100% |

Phase A qualification uses a separate deterministic, noise-free local skill audit with
eight fixed-seed attempts per runtime gate. The audit uses the active Phase-A opening,
while recording physical strict passes from the same crossings. Qualification uses
the last ten audits, requires at least 32 samples per gate, an approximately 80%
minimum active-window pass rate, and two consecutive qualifying audits. Lifetime
rates remain diagnostics only, so old failures cannot permanently lock the phase.
Rolling audits are tagged with their actual gate-window scale. A different scale
clears rolling qualification evidence and its streak, retaining lifetime statistics
and historical metrics. Phase A cannot advance while its aperture is still moving
toward the configured target. Older checkpoints without audit-scale metadata start
with empty rolling evidence on resume; policy, optimizer and schedule are retained.
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
- Previous action, course progress, remaining competition time, a reserved zero channel, and legacy-v1 stall state.
- Relative gate pose, normal, image-plane bearing, visibility, and distance for the next `N` gates.

Noise is applied in physical units per sensor/feature before normalization. `--sensor-noise-scale` scales all configured noise models; the legacy `--obs-additive-noise-std` spelling is retained as an alias for this scale. PnP dropout masks all PnP-derived geometry. Running normalization is used only for unbounded channels; bounded quaternions, normals, visibility, progress, and controller channels use fixed transforms.

The critic separately receives exact pose, attitude, velocity, angular velocity, acceleration, active and next gate geometry, elapsed time, gate progress, previous motor action, a reserved zero channel, gate-window scale, and reset-start type. Those features never enter the actor network or actor loss.

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

Linux (Bash):

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

Windows (Command Prompt):

```bat
set "PYTHONPATH=src"
.\.venv\Scripts\python.exe scripts/eval_chase_video.py ^
  --checkpoint-dir checkpoints ^
  --course arena_38m_stacked ^
  --output artifacts/eval_chase_arena_38m_stacked.mp4 ^
  --num-eval-envs 32 ^
  --visualization-env 0 ^
  --seed 123 ^
  --device cpu
```

## Checkpoints

Schema-v6 checkpoints include actor and critic parameters, optimizer state, actor
normalization statistics, curriculum phase, official and skill-audit counters, rolling
audit history, latest strict-evaluation gate results and priority state, consumed PPO
schedule steps, evaluation counters,
total environment steps, update count, episode count, RNG state, and a signed course
geometry fingerprint.

Schema-v6 also records the action-space identifier
`motor_rpm_hover_centered_v1`. Training and video evaluation reject checkpoints
without that identifier, including all older attitude-control checkpoints. Motor
checkpoints restore through `--restore-checkpoint`; the total-step target and the
schedule-step budget remain independent. Course-fingerprint checks still apply.

## Tests And Benchmark

Linux (Bash):

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/benchmark_cpu_training.py --updates 3
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m pip install pytest
set "PYTHONPATH=src"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark_cpu_training.py --updates 3
```

In Command Prompt, `set "PYTHONPATH=src"` applies to subsequent commands in the current
terminal session.

## Motor Policy Diagnostics And Corrected Spawns

Lateral reset offsets now lie in the gate plane. Previously, offsets perpendicular
only to the incoming path could place a drone beyond an oblique gate before its
attempt began. On checkpoint 90, three of eight gate-5 audit starts had this issue.
With the same policy and seeds, corrected spawns improved gate-5 local passes from
0/8 to 5/8. This changes reset geometry, not gate positions or pass criteria.

On resume from an older motor checkpoint, the trainer clears audit/evaluation
statistics and sampling priorities collected with the old reset geometry. Policy,
optimizer, normalization, phase, environment steps, and schedules are preserved.
Use a separate output directory to retain the original experiment:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --total-env-steps 20000000 \
  --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints_motors/checkpoint_000090.pkl \
  --checkpoint-dir checkpoints_motors_spawnfix
```

Reproduce the deterministic diagnostics in WSL (CPU, frozen normalization,
noise/dropout disabled). These commands save traces and summaries without training:

```bash
./.venv/bin/python3.13 scripts/check_motor_response.py
./.venv/bin/python3.13 scripts/diagnose_motor_policy.py \
  --checkpoint checkpoints_motors/checkpoint_000090.pkl \
  --scenario course --output artifacts/motor_diagnostics/course90_replay
./.venv/bin/python3.13 scripts/diagnose_motor_policy.py \
  --checkpoint checkpoints_motors/checkpoint_000090.pkl \
  --scenario skill --seconds 4 --output artifacts/motor_diagnostics/skill90_replay
```

For a baseline comparison, add `--legacy-spawns` and use a different output folder.
The legacy option exists only in the diagnostic script. See
[the diagnostic report](artifacts/motor_diagnostics/REPORT.md) for results and the
remaining G1-to-G2 control failure. The spawn fix does not establish full-course
success; evaluation must remain enabled during further training.


## Follow-up after the 1.40x run

The Phase-A default now allocates 40% of resets to G1 and 60% to local starts.
This increases practice on G1-to-G2 transitions while retaining local gate coverage.
The update-306 diagnostics found G6 passing 7/8 local attempts (6/8 strict), but
full-course evaluation passed G1 in 32/32 attempts and G2 in 0/32 at seed 123.
This is a sampling intervention to evaluate, not a demonstrated performance gain.

Training tables and JSON metrics now separate training, course evaluation, and
skill-audit seconds. In JSON metrics, `training_env_steps_per_second` excludes
evaluations; `env_steps_per_second` includes them. Initial compilation is included in the relevant timer. Checkpoint writing and
log output are outside these update timers.

Re-run read-only diagnostics against a fixed checkpoint in WSL:

```bash
./.venv/bin/python3.13 scripts/diagnose_motor_policy.py \
  --checkpoint checkpoints_motors_spawnfix/checkpoint_000306.pkl \
  --scenario skill --output artifacts/motor_diagnostics/skill306 --seconds 6
./.venv/bin/python3.13 scripts/diagnose_motor_policy.py \
  --checkpoint checkpoints_motors_spawnfix/checkpoint_000306.pkl \
  --scenario course --output artifacts/motor_diagnostics/course306 --seconds 8
```

Skill diagnostics use the checkpoint's saved aperture by default; use
`--gate-window-scale 1.40` to override it. Course diagnostics always use strict
1.0x gates. `summary.json` contains pass/failure events and gate totals; `traces.npz`
contains positions, velocity, attitude, angular velocity, motor actions and RPM.
Events can have more than one failure reason; do not add overlapping reason counts.
Repeat course diagnostics with `--seed 124` and a different output directory to
check whether a finding persists across starts.

Update 306 has already reached the original 20-million-step target. To evaluate the
new reset mixture, use a separate output directory and a higher total-step target:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles --sim-hz 500 --control-hz 500 \
  --phase-a-window-scale 1.40 --phase-a-gate1-fraction 0.40 \
  --total-env-steps 22000000 --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints_motors_spawnfix/checkpoint_000306.pkl \
  --checkpoint-dir checkpoints_motors_link_practice
```

The existing schedule remains consumed: learning rates and exploration stay at
their scheduled final values. This does not restart annealing. Compare strict G2
passes and local per-gate audits against the fixed source checkpoint before
adopting the change for a longer run.

If linking remains weak, test longer credit assignment separately from the reset
mixture change. A candidate uses `--num-envs 64 --horizon 1024 --minibatches 32`
(the same 65,536 transitions/update as 256 x 256), `--gamma 0.99979992`,
`--gae-lambda 0.99799195`, and `--potential-gamma 0.99979992`. At 500 Hz this gives
2.048-second rollouts and a roughly 0.9-second GAE decay time. These are experimental
settings, not new defaults or a validated improvement. Start both comparisons from
the same fixed checkpoint, keep the reset mixture and update budget equal, and use
separate checkpoint directories. No long training run is launched by diagnostics.


## Longer credit with increased exploration

Use `--profile rtx-5050-long-credit` for the experimental continuation from update
367. It sets 64 environments, a 1024-step (2.048-second) horizon, 32 minibatches,
gamma/potential gamma 0.99979992 and GAE lambda 0.99799195. The batch remains
65,536 transitions; GAE decay time is approximately 0.9 seconds.

The profile also holds the exploration standard-deviation ceiling at 0.35, raises
the floor to 0.25, and holds the entropy coefficient at 0.001. These apply even
when the checkpoint's schedule is exhausted. The floor actually increases resumed
standard deviations below 0.25; just increasing a ceiling would not do so.
These standard deviations are in the policy's pre-tanh action distribution,
not percentages of motor RPM. Learning-rate schedules are preserved.

Run a monitored two-million-step continuation, retaining evaluation:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050-long-credit \
  --physics first_principles \
  --sim-hz 500 --control-hz 500 \
  --phase-a-window-scale 1.40 --phase-a-gate1-fraction 0.40 \
  --total-env-steps 26000000 --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints_motors_link_practice/checkpoint_000367.pkl \
  --checkpoint-dir checkpoints_motors_long_credit
```

This jointly tests longer credit and increased exploration; improvement cannot be
attributed to either change alone. Compare strict course G3/G4/G5 passes and local
G5/G11 accuracy against checkpoint 367. More exploration can initially increase
crashes. Keep this run separate from the source checkpoints. Explicit CLI flags
override profile defaults. This profile does not itself start training.


### Current long-credit setup: initial starts and strict gates

The `rtx-5050-long-credit` profile now enables `--strict-course-training`:
all training resets start at G1 and all openings are 1.00x, immediately on resume
and in every curriculum phase. This overrides phase-specific start fractions and
window scales. Normal G1 spawn randomization remains; this does not fix every reset
to identical coordinates. Local skill audits remain diagnostics only. The existing
phase-dependent time-cost schedule is retained. Rolling audits from larger openings
are cleared, while lifetime counts and learned policy parameters are retained.

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050-long-credit \
  --strict-course-training \
  --physics first_principles --sim-hz 500 --control-hz 500 \
  --total-env-steps 26000000 --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints_motors_long_credit/checkpoint_000380.pkl \
  --checkpoint-dir checkpoints_motors_strict_start
```

This replaces the earlier mixed-start, 1.40x long-credit command. Longer credit and
increased exploration remain enabled. Restart training with the new configuration
to apply it; editing the code does not change an already-running process.


### Reading training progress

The terminal now prints short stage messages for each update: collecting flight
experience, learning, course evaluation, gate-skill evaluation, and checkpoint saved.
The first update can take longer while JAX compiles. A quiet evaluation stage means
that evaluation has started; it is not proof that the process is still running.

`--log-interval 5` prints a compact results table every five updates (also the first
and final updates). Other updates print a one-line completion message. The table
shows steps, estimated remaining time, training/evaluation time, gate scale, G1
start target, reward and basic PPO checks. Course results identify their evaluation
update and show **passes / flights that reached that gate**, not all starting
flights. Unreached gates are not counted as failed attempts. The local-skill row
identifies the weakest strict gate. Detailed diagnostic fields remain in
`metrics.jsonl`; finite PPO values alone do not establish successful racing.


## Targeted G3-to-G5 practice (current experiment)

Use `rtx-5050-corner` to keep longer credit assignment and the same exploration,
while replacing the strict-start-only reset policy with this expected mixture:

| Training starts | Share |
| --- | ---: |
| Full course from G1 | 40% |
| Recorded pre-G3 states, complete G3, G4 then G5 | 40% |
| Other local gates, excluding G3 | 20% |

The drone must fly through G3 and the entire G3-to-G4 leg before attempting the
G4-to-G5 turn. G3 and G4 are about 9.4 metres apart, providing a substantial
lead-in instead of dropping the drone directly into the final turn.

All gates are **1.00x immediately on resume**, in every curriculum phase. The
G3-focused episodes end only after G5 is passed (or an ordinary failure), do not
count as full-course finishes, and receive no full-course finish bonus under
Reward V2. Other local starts retain their existing remaining-course objective.
Full-course evaluation still starts from G1; skill audits do not use the reset bank.
The usual phase-dependent time cost is retained. Do not combine this profile with
`--strict-course-training` or `--no-curriculum`.

The reset bank contains 160 pre-G3 states from 28 checkpoint-380 flights, seed 124:
1.5-4 metres before the G3 plane, at approximately 4.3-5.5 m/s. It restores position,
velocity, quaternion, angular velocity, rotor RPM and previous motor action from
those recorded flights. Sensor/acceleration history starts afresh on reset.
The finite bank can overfit, so judge progress with full-course evaluation and
additional seeds, not only targeted practice completions.

Run from the repository root in WSL:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050-corner \
  --physics first_principles --sim-hz 500 --control-hz 500 \
  --log-interval 5 --checkpoint-interval 5 \
  --total-env-steps 27000000 --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints_motors_long_credit/checkpoint_000380.pkl \
  --checkpoint-dir checkpoints_motors_corner_practice
```

This is about 2.1 million additional steps from checkpoint 380, in a separate
output directory. Learning rates and exploration remain at their previous scheduled
values; this does not restart annealing. The compact log adds G3-to-G5 practice
resets and completed segments for the logged rollout. JSON records include
`corner_reset_count` and `corner_complete_count` on every update. They are counts,
not a success fraction: an episode may start and finish in different rollouts.

The profile loads `artifacts/motor_diagnostics/g3_approach_bank.npz`; retain that
file when moving the run. To reproduce it from the saved diagnostic trace:

```bash
./.venv/bin/python3.13 scripts/build_corner_reset_bank.py \
  --traces artifacts/motor_diagnostics/course380_seed124/traces.npz \
  --source-update 380 --source-seed 124 \
  --output artifacts/motor_diagnostics/g3_approach_bank.npz
```

Stop and assess after the trial: look for strict G5 passes, survival after G4,
preservation of G1-to-G4 performance, and local G11 accuracy. No improvement has
yet been established for this new curriculum.

## Hierarchical trajectory / residual RL / MPC controller

The existing `--controller direct_motor` mode remains the default. The optional
`residual_mpc` mode separates an offline minimum-time reference, a four-output
residual policy, and a quaternion nonlinear MPC flight controller. It uses the
same first-principles drone model and native motor ordering as Crazyflow.

Install the offline optimizer and optional report plotting dependencies:

```bash
pip install -e '.[planning,visualization]'
```

Generate a reference before starting a residual run:

```bash
a2rl-drone-plan --course arena_38m_stacked \
  --output artifacts/arena_reference.npz
```

The planner jointly optimizes motor commands, states, gate-crossing locations,
and segment durations with CasADi/IPOPT multiple shooting. It includes rotor
response, drag, thrust/torque curves, inertia, and normalized `xyzw` attitude.
Gate openings are reduced by the vehicle radius (0.15 m by default) and a 0.10 m
tracking margin. Gate frames use the union of outer rectangles minus the union
of openings for each logical gate; frame depth defaults to 0.10 m. These are
simulation geometry assumptions, not measured competition-frame specifications.
Stacked-gate initialization uses explicit exit and re-approach waypoints with
continuous acceleration and jerk. A feasibility solve supplies the warm start
for minimum-time optimization. Separating planes around frame solids define a
conservative clearance corridor along this route. The reference starts at rest
at the course's nominal start. The final crossing ends the timed run; this mode currently
supports one course run, not continuous flying laps.

Every solution is replayed densely at **at least 500 Hz** to check numerical
integration defects, frame clearance, bounds, and crossing geometry. Invalid
solutions trigger mesh refinement and are rejected if validation still fails.
This is a numerical local optimization and empirical feasibility check, not a
global-optimality or formal safety certificate. The NPZ includes reference
states, commands, timing, progress, gate events, and fingerprints for the model,
course, planner settings, and artifact contents. An incompatible or unvalidated
reference is rejected on load.
An iteration-limited optimization candidate is usable only if it passes the same
dense feasibility checks. Artifacts explicitly record solver status, iteration
count, and whether the optimization converged; a feasible candidate is not a
claim of optimality.

Train the residual policy, using the GPU setup described above:

```bash
a2rl-drone-train --controller residual_mpc \
  --reference-path artifacts/arena_reference.npz \
  --device gpu --num-envs 16 --physics first_principles --sim-hz 500 \
  --policy-hz 20 --mpc-hz 100 --mpc-horizon 0.5 \
  --mpc-prediction-dt 0.02 --mpc-iterations 5 \
  --checkpoint-dir checkpoints_residual
```

Before collecting training experience, the trainer runs a deterministic,
noise-free, zero-offset flight. Training refuses to start if that flight does
not complete the course. Planning feasibility alone does not imply that the
flight controller can track an aggressive minimum-time solution. Use
`--max-speed` when planning to constrain the reference if necessary, and set
`--max-episode-time` to allow sufficient time for the planned course.

The policy outputs `[offset_x, offset_y, offset_z, speed_offset]` in `[-1, 1]`.
Defaults map these to world-frame position offsets of ±0.5 m per axis and a speed
change of ±30%. A critically damped 0.2 s filter supplies position, velocity,
acceleration, and speed derivatives for a consistent MPC reference. Progress is
projected only onto the active gate segment, including stacked turns. All-zero
residuals reproduce the planned reference. RL may explore outside the planner's
tracking margin; there is no runtime gate-corridor safety shield. Physical motor
bounds are always enforced, and swept vehicle/frame collisions terminate the
episode before a simultaneous crossing can earn a gate reward.

The MPC uses batched JAX iLQR, a 0.5 s prediction horizon, 0.02 s prediction
spacing, and five iterations by default. Its model includes normalized rotor
speeds and uses a three-component quaternion attitude error in its cost. It
warm-starts the previous solution and checks numerical validity. A failed solve
may reuse the previous valid sequence for one controller interval; subsequent
failures use a bounded geometric stabilization fallback. Fallback counts and
solve latency are reported. Control frequencies are simulated frequencies;
real-time wall-clock performance must be measured on the target machine.

Actor observations add tracking errors, three reference lookahead positions,
nominal speed, previous residual actions, and the offset filter state and derivative
so the controller memory is observable. The critic remains separate.
Initial control uses exact simulator state. `--estimation-noise-std` adds position
and velocity estimation noise for robustness experiments; it does not implement
a sensor-fusion estimator. Spawn perturbations are configured separately with
`--spawn-position-std` and `--spawn-velocity-std`.

Each policy action spans up to 25 physics steps. Rewards accumulate only through
the first terminal/truncation event. Terminal observations are captured before
reset; time-limit bootstrapping uses the actual partial-action duration.
`--discount-per-second` and `--gae-lambda-per-second` replace per-step discount
settings in this mode. PPO budgets and schedules count **policy transitions**;
`physics_steps` is recorded independently. Gate openings stay at 1.0×; the
original gate-window curriculum and local reset bank do not apply. New
checkpoints identify the controller, observation schema, and reference artifact,
so a four-motor policy cannot be mistaken for a four-residual policy.

Evaluate baseline stability or compare a trained checkpoint on paired held-out
trials (100 by default):

```bash
a2rl-drone-evaluate --reference-path artifacts/arena_reference.npz \
  --device gpu --trials 100 --num-envs 16 --plot \
  --output-dir artifacts/mpc_baseline

a2rl-drone-evaluate --reference-path artifacts/arena_reference.npz \
  --checkpoint checkpoints_residual/checkpoint_latest.pkl \
  --device gpu --trials 100 --num-envs 16 --plot \
  --output-dir artifacts/residual_comparison
```

Both commands run exact-state and noisy-state suites with shared spawn seeds.
Keep the batch size fixed across comparisons. Reports include failures, lap-time
distributions, frame clearance, tracking errors, fallback counts, latency,
throughput, raw NPZ trials, and optional trajectory/speed/clearance plots. Learned
acceptance requires at least 100 trials, at least 95% completion, and a positive
95% bootstrap confidence interval for mean lap-time improvement on paired
successful trials in both suites. Failures are listed separately. A baseline-only
report cannot claim learned-policy acceptance. No pretrained residual policy or
claim of improved A2RL lap time is included with this implementation.
Evaluation retains the configured episode deadline; `--max-episode-time` is an
explicit override for controller experiments, not an automatic extension.

Implementation validation (CPU, September 2026): the repository suite passes
112 tests, with 19 focused controller tests also passing after the final
start-at-rest correction. A deterministic two-gate straight-course flight from
rest completed in 0.982 simulated seconds with no frame collision or MPC
fallback. Its minimum vehicle/frame clearance was 0.0719 m and mean position
tracking error was 0.0185 m. Median MPC solve latency was 30.9 ms on this CPU;
this does **not** meet the configured 10 ms controller interval in wall time.
The measurement is a smoke test, not a reliability estimate. Stacked two-gate
trajectory optimization also passes dense validation.

Full-arena zero-offset completion has **not** been established. The full-arena
planning experiment required further refinement after failing dense validation;
no accepted full-course flight result or trained residual policy is supplied.
GPU throughput, the 100-trial held-out comparison, and noisy-state acceptance
remain to be measured. Training stays disabled unless the runtime baseline
check passes for the supplied reference.
