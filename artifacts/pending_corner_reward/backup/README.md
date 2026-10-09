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

Generate a reference before starting a residual run. For the first controller
check, use a feasible reference with a conservative speed cap:

```bash
a2rl-drone-plan --course arena_38m_stacked \
  --max-speed 3 --nodes-per-segment 24 --max-refinements 1 \
  --feasibility-only --output artifacts/arena_reference.npz
```

From Windows PowerShell, run the WSL command on one line:

```powershell
wsl -d Ubuntu --cd "/mnt/c/Users/abdul/Downloads/not ice dont open/ice-main" -- env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/a2rl-drone-plan --course arena_38m_stacked --max-speed 3 --nodes-per-segment 24 --max-refinements 1 --feasibility-only --output artifacts/arena_reference.npz
```

The planner uses CPU execution. This command limits refinement to 576 intervals;
it can still fail validation, in which case no new accepted reference is saved.
`--feasibility-only` skips minimum-time optimization. Dense validation thresholds
are unchanged. A validated feasible candidate is saved immediately as
`artifacts/arena_reference_feasible.npz`, before any optional speed optimization.
Omit `--feasibility-only` to attempt a faster trajectory; failed or slower speed
candidates retain the validated feasible reference. Check the artifact's
`objective` and `solver_converged` metadata before interpreting it as optimized.
The zero-offset MPC flight must still pass before residual training can begin.

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

Check nominal MPC tracking before training (one deterministic flight, from PowerShell):

```powershell
wsl -d Ubuntu --cd "/mnt/c/Users/abdul/Downloads/not ice dont open/ice-main" -- env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/a2rl-drone-evaluate --reference-path artifacts/arena_reference.npz --device gpu --num-envs 1 --trials 1 --exact-state-only --spawn-position-std 0 --spawn-velocity-std 0 --max-episode-time 120 --output-dir artifacts/feasibility_baseline_optimized
```

Evaluation prints progress every simulated second (`--progress-interval`) and
saves partial diagnostics every five simulated seconds (`--save-interval`).
Both also trigger after 30 wall-clock seconds at the next completed policy step;
compilation or a blocked step can delay that heartbeat. The live directory
`artifacts/feasibility_baseline_optimized/exact_state_baseline_live` contains:

- `progress.json`: run status, current gate, collisions, tracking error, fallbacks,
  and cumulative MPC, physics, and other wall time.
- `trace.npz`: a labeled `trace` array with trial IDs, times, position, shifted
  reference target, normalized motor commands, and flight metrics, sampled at
  policy steps. Completed worlds stop contributing trace rows.

Files are replaced atomically; Ctrl+C saves the last completed observations.
These files are diagnostics, not resumable training checkpoints. `completed`
means the evaluation ended; check each trial's `success` and the final
`report.json` completion rate to determine whether the controller passed.
Residual control now compiles gate/reward calculations, actor/critic
observation builders, and physics-tick bookkeeping into JAX kernels. State, aperture scale, and time-cost
scale remain explicit inputs; physics and controller rates are unchanged.
Live timing breaks `other` into gate/reward, observation, reference preparation,
and bookkeeping stages, plus unclassified time. These detail fields are subsets
of `other`, not additional time to add to the total.

To compare warmed eager and compiled paths on a short deterministic flight:

```bash
./.venv/bin/python scripts/benchmark_residual_step.py --device gpu --steps 4
```

The benchmark checks states, observations, rewards, terminal flags, and gate
pass events against the eager path before writing
`artifacts/residual_step_benchmark.json`. Its timings include diagnostic
synchronization but exclude the warmup step. A short-flight equivalence check
does not replace full-course baseline validation.

Timing synchronizes GPU work and includes compilation; `other` includes setup,
reference calculations, rewards, logging, and remaining processing. Profiling
adds overhead and does not measure real-time deployment performance. Training's
mandatory baseline also saves diagnostics under the checkpoint directory's
`baseline_diagnostics` folder; normal PPO rollouts do not enable this profiling.

To run the qualification sequence and a small PPO pilot automatically:

```powershell
wsl -d Ubuntu --cd "/mnt/c/Users/abdul/Downloads/not ice dont open/ice-main" -- ./.venv/bin/python scripts/run_staged_validation.py --output-dir artifacts/staged_validation
```

Use a new output directory for each run. The sequence requires one nominal
12-gate finish, then four flights with 0.02 m position and 0.05 m/s velocity
spawn standard deviations. Every flight must complete all gates without
collisions or MPC fallbacks, with finite flight metrics. A failed check stops
the sequence and writes the reason to `status.json`; no later stage starts.
Four disturbed flights are a preliminary check, not a statistical reliability
claim or a test of sensor noise and wind.

After both checks pass, it starts a fresh residual-PPO pilot: four environments,
a 64-step rollout, and 1,024 policy transitions (four updates), saving every
update under `pilot_checkpoints`. The trainer retains its own mandatory nominal
baseline check. The pilot verifies that training and checkpointing work; it is
not enough training to establish a performance improvement. Stage logs and
flight traces remain under the chosen output directory.

For an already running nominal WSL evaluator, `--baseline-pid PID --baseline-dir
DIRECTORY` attaches to that exact process after checking its command, reference,
and output directory. It rejects existing completed results and changed
references. Do not launch a second runner while the active sequence is waiting.

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


### Sensor-model audit and MPC robustness sweep

The residual controller now uses one cached position/velocity estimate per MPC
tick for both policy observations and MPC inputs. Position-derived gate features
also use that estimate. Noise has separate metre and metre/second scales. Delay
is quantized to MPC ticks (10 ms at 100 Hz), bias is fixed per axis, and a dropped
sample holds the previous position/velocity estimate. Reset histories contain
initial state, never a previous episode. The legacy `estimation_noise_std` is a
fallback for either scale when no separate scale is supplied.

This remains a synthetic estimator-error model: attitude, angular rates,
acceleration and rotor states are exact; the gate map/index is known. It does
not implement camera/IMU fusion, attitude noise, timestamp jitter, wind, or
opponents. The privileged training critic still receives simulation truth.
Do not interpret these stress levels as measured race-sensor specifications.

Run a one-factor MPC-only sweep from PowerShell:

```powershell
wsl -d Ubuntu --cd "/mnt/c/Users/abdul/Downloads/not ice dont open/ice-main" -- env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python scripts/audit_sensor_robustness.py
```

The default sweep uses two seeded trials per condition: clean, position noise
of 1/3 cm, velocity noise of 2/5 cm/s, and latency of 10/30 ms. Starts are nominal
and other disturbances are zero, isolating one factor at a time. Optional cases
include `dropout_5pct` and `position_bias_2cm`. The clean baseline must pass first.
Reports and traces are saved after every condition under `artifacts/sensor_audit`.
This is a preliminary sensitivity test, not a reliability certification. No RL
training starts automatically. Use a fresh `--output-dir` for another sweep.

Evaluation CLI options `--position-noise-std`, `--velocity-noise-std`,
`--sensor-delay`, `--sensor-dropout`, `--position-bias X Y Z` and
`--velocity-bias X Y Z` apply to its noisy-state suite; the exact-state suite
explicitly zeroes them. Training uses `--estimator-position-noise-std` and
`--estimator-velocity-noise-std` to distinguish these from its existing
observation-noise flags, plus `--sensor-delay` and `--sensor-dropout`. Its
mandatory nominal baseline explicitly clears all sensor disturbances.

To summarize completed audit cases without interrupting the sweep, run in WSL:

```bash
./.venv/bin/python scripts/assess_sensor_audit.py --watch
```

This refreshes `artifacts/sensor_audit/assessment.md` and `assessment.json` after
each case and exits when the audit ends. It reports metric differences from the
clean baseline, withholds lap comparisons for failed cases, and recommends
follow-up diagnostics without automatically changing the controller or starting
training. The source audit status is reported as recorded; this is not a process
liveness monitor. Two identical clean flights are deterministic repeats, not
independent evidence of reliability.

## Uncapped-speed residual PPO curriculum

Start in WSL from the project directory:

```bash
./.venv/bin/python scripts/run_speed_curriculum.py --budget-hours 24
```

Resume the same configuration and artifacts after an interruption:

```bash
./.venv/bin/python scripts/run_speed_curriculum.py --budget-hours 24 --resume
```

The budget is cumulative active execution time, including planning, qualification,
training and evaluation; restarting does not grant another 24 hours. Wall time
while the runner is stopped is excluded. The runner kills child process groups
at the deadline; a partly collected rollout is discarded and the last completed
update remains available. Budget exhaustion does not count as successful flight.

Stages are **6, 12, 24 m/s, then uncapped**. `a2rl-drone-plan --max-speed none`
removes only the planner's absolute translational-speed constraint, recording
JSON null in reference metadata. Numeric caps must be positive and finite.
`--warm-start PATH` resamples a validated reference with matching course and
vehicle fingerprints. Motor/RPM, body-rate, boundary, gate and dense dynamics
checks remain enforced. Caps constrain the plan, not actual flight speed.
Equal achieved peak speed at successive stages is allowed.

Each stage passes one nominal and four perturbed-start MPC flights, then trains
fresh residual PPO in 32-update blocks, at most four blocks. Each block contains
32,768 policy transitions (four worlds, 256-step rollouts). Policy/MPC/physics
rates remain 20/100/500 Hz, with bounded corrections and 1 cm position noise.
Checkpoints are written every update. The learning schedule spans 128 updates
and remains fixed across blocks. Policies are resumed only on their original
reference; a new reference starts a fresh policy.

After each block, eight held-out paired flights under exact-state and 1 cm noise
must all finish with no collisions or fallbacks. Mean RL lap time must be at most
2% slower than MPC in each suite. Screening seeds change between blocks. A failed
qualification or four unsuccessful blocks stops the pipeline without weakening
constraints. The first qualifying block is retained for progression.

Final acceptance requires 100 distinct, previously unused seeds **per condition**:
exact state, position noise 1/3 cm, velocity noise 2/5 cm/s, and delay 10/30 ms.
Errors are isolated; start perturbations remain 2 cm and 5 cm/s. Baseline and RL
use identical seeds. All 100 flights for both controllers must pass every gate
without collisions or fallbacks; RL's mean lap must be within 2% of MPC. Each
completed pair is persisted with input/output hashes. One worker per condition
reuses its loaded policy and compiled environment across all 100 seeds, allowing
resume without rerunning completed pairs. This can exceed 24 hours; incomplete
validation remains incomplete. The existing evaluator's separate faster-than-MPC
`accepted` field is not the curriculum's reliability criterion.

Follow `artifacts/uncapped_curriculum/status.json`, its `current_log`, and
`final_report.json`. Status includes the current child PID, heartbeat, updates,
remaining budget and completed final pairs. A stale heartbeat is not proof that
a process is alive. Reference/output hashes, source code and configuration are
checked on resume. Older `artifacts/speed_curriculum` results are retained as
historical artifacts; the new runner uses a separate output directory.

Final metrics include sampled peak speed (20 Hz), lap time, tracking error,
clearance and MPC latency. Only a complete passing run is labeled
`uncapped simulation accepted`. This does not establish real-time performance or
real A2RL hardware capability; the vehicle and synthetic errors remain uncalibrated.

### Rollout performance

Live flight progress does not enable synchronized component profiling by default.
Use `a2rl-drone-evaluate --profile-timing ...` only when investigating detailed
MPC/physics/observation timings. Lap, clearance, tracking and controller latency
metrics remain available in normal evaluation. Profiling does not change physics.

The final validation worker (`scripts/evaluate_seeded_condition.py`) keeps one
policy and one environment alive per condition. Every flight resets MPC, sensor
history, RNG state, counters and episode state. Baseline/RL pairs keep the same
individual seeds; no conditions or required trials are removed. Existing pair
markers are checked before reuse. `progress.json` reports completed pairs while
the worker runs; the curriculum supervisor still enforces its shared deadline.

Benchmark warmed rollout throughput on the GPU:

```bash
./.venv/bin/python scripts/benchmark_rollout_throughput.py --reference-path artifacts/uncapped_curriculum/6mps/reference.npz
```

This compares 4/8/16 environments and verifies equivalent profiled/unprofiled
trajectories. Results go to `artifacts/rollout_throughput.json`. It measures rollout
throughput, not PPO update performance or learning quality; training retains its
four-world, 256-step batch configuration until a separate training comparison
justifies changing that configuration.

## Current strategy: optimized trajectory with progressive RL authority

The trajectory-first curriculum replaces the absolute-speed ladder for the active
experiment. It optimizes one reference using `--max-speed none --curvature-seed`,
then retains that reference and carries the learned checkpoint across RL stages:

| RL stage | Positive local speed correction | Slowing allowance | Policy std bounds | Entropy coefficient |
|---|---:|---:|---:|---:|
| 1 | up to +10% | up to -30% | 0.20-0.35 | 0.003 |
| 2 | up to +20% | up to -30% | 0.35-0.50 | 0.005 |
| 3 | up to +30% | up to -30% | 0.50-0.65 | 0.008 |

These are reference-phase speed corrections, not an absolute physical velocity
limit. Position-offset derivatives can also contribute velocity. Zero RL action
still reproduces the optimized reference. The optional
`--residual-speedup-fraction` changes only positive speed-action scaling; the
existing `--residual-speed-fraction` retains negative scaling and is the default
positive scaling when no separate allowance is provided.

The curvature seed uses the existing gate-aware spline geometry, estimates
curvature and applies forward acceleration/backward braking passes. Its initial
acceleration budget is 25% of the model's estimated lateral thrust reserve. It
starts at rest; this does not launch the drone with an artificial initial speed.
This is an initial guess, not proof of feasibility or a globally optimal lap.
The nonlinear optimizer and dense validator retain the physical constraints.
`--curvature-seed` and `--warm-start` are mutually exclusive to avoid silently
ignoring one initialization strategy.

Each stage trains in 32-update blocks, up to four blocks. Higher speed authority,
exploration bounds and entropy incentives are enabled only after all eight paired
screening flights in both clean/1 cm noise suites complete without collisions or
fallbacks and meet the existing 2% lap-time criterion. The PPO schedule spans the
maximum 384 updates; checkpoints keep the same reference, optimizer and observation
normalization across stages. Final 100-seed tests in all seven conditions remain.

Start the current curriculum in WSL:

```bash
./.venv/bin/python scripts/run_residual_curriculum.py --budget-hours 24
```

Resume it with the same command plus `--resume`. Status and logs now live under
`artifacts/residual_curriculum`. By default, the runner carries elapsed execution
time from the stopped `artifacts/uncapped_curriculum/status.json`; restarting the
strategy does not grant a fresh 24 hours. Older run artifacts remain preserved.


### Recovery after uncapped MPC floor failure

The first uncapped reference passed dense dynamics checks but failed nominal MPC
flight after four gates. The drone crossed the minimum center height of 0.20 m
at 4.526 s; heavy motor saturation accompanied the tracking failure.

The recovery reserves motor authority for MPC: `--motor-command-limit 0.8`
limits the planner's normalized upper motor command and corresponding rotor RPM.
MPC retains the physical motor range. This is not a translational speed cap or
an 80% thrust limit. It is a candidate tracking remedy, subject to the same
nominal and perturbed-flight qualification before PPO starts.

Run in WSL from the project directory (add `--resume` only for an existing run):

```bash
./.venv/bin/python scripts/run_residual_curriculum.py \
  --output-dir artifacts/residual_headroom \
  --budget-hours 24 \
  --carry-budget-from artifacts/residual_curriculum/status.json \
  --warm-start artifacts/residual_curriculum/reference.npz \
  --motor-command-limit 0.8
```

This preserves the failed run and carries its elapsed execution time into the
24-hour budget. Monitor `artifacts/residual_headroom/status.json` and its current
log. Evaluation reports include crash, boundary, missed-gate, and deadline counts.


### RL-first speed curriculum: 12, 24, then uncapped

The requested RL-first run starts PPO without requiring the preliminary MPC
flights to pass. It uses the headroom reference as the planning warm start,
retains the planner motor-command limit of 0.8, and trains with perturbed starts
and 1 cm position noise. Each stage is replanned with its requested speed cap;
these are planner speed limits, not hard limits on disturbed simulated velocity.
New references use fresh policies. Training runs in 32-update blocks (four at
most per stage), saving every update. Paired screening and final acceptance
requirements remain unchanged; starting training does not mean qualification.

```bash
./.venv/bin/python scripts/run_speed_curriculum.py \
  --output-dir artifacts/rl_speed_curriculum \
  --previous-reference artifacts/residual_headroom/reference.npz \
  --caps 12 24 none \
  --motor-command-limit 0.8 \
  --train-before-qualification \
  --budget-hours 24 \
  --carry-budget-from artifacts/residual_headroom/status.json
```

Add `--resume` to this same command after interruption. Monitor
`artifacts/rl_speed_curriculum/status.json`. The previous execution time counts
against the original 24-hour budget.


For earlier promotion checks, add `--screen-every 16` to the RL-first speed
curriculum command. This tests every 16 updates instead of 32, retaining the
128-update maximum per reference and the unchanged PPO learning-rate schedule.
A stage advances immediately after both eight-pair screening suites pass: every
flight completes without collisions or fallbacks, and mean RL lap time is at
most 2% slower than MPC. Training reward alone cannot trigger promotion. More
frequent tests may cost extra time when the policy is not yet ready. Final
uncapped acceptance remains unchanged.


### Recovery-only RL assistant

`--recovery-only` on the speed curriculum trains a fresh residual policy that
activates only when the sensor-estimated position differs from the original
reference by more than 5 cm, or velocity differs by more than 0.5 m/s. Authority
ramps smoothly to full at 15 cm / 1.5 m/s. It shapes a short local MPC reference
using bounded 10 cm offsets and up to 15% slowdown, with no positive speed boost.
This is local reference correction, not a separately collision-validated path
planner. Existing smoothing returns corrections toward zero after recovery.

Training uses 8 cm position and 0.15 m/s velocity spawn perturbations, plus 1 cm
position measurement noise, to expose recoverable deviations. Screening retains
its original held-out distribution and pass criteria. Every 16 updates it tests
for promotion, continuing on failure up to 128 updates per stage and within the
remaining original 24-hour budget. Failure never authorizes higher speed.

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ./.venv/bin/python scripts/run_speed_curriculum.py \
  --output-dir artifacts/recovery_assistant \
  --previous-reference artifacts/rl_speed_curriculum/12mps/reference.npz \
  --caps 12 24 none --motor-command-limit 0.8 \
  --train-before-qualification --recovery-only --screen-every 16 \
  --budget-hours 24 --carry-budget-from artifacts/rl_speed_curriculum/status.json
```

The coordinator uses CPU for checkpoint inspection; its training and evaluation
workers select GPU explicitly. Monitor `artifacts/recovery_assistant/status.json`.


Recovery mode now enables `--residual-recovery-critical-areas`: activation thresholds
blend from 5 cm / 0.5 m/s on straights down to 2 cm / 0.2 m/s at gate crossings
and tight turns. Gate sensitivity spans half a second on either side of the
reference crossing; turn sensitivity looks half a second ahead, increasing for
curvature from 0.15 to 0.5 per metre. Exactly on-path motion still receives no
RL correction. Correction bounds, smoothing, and pass-only promotion remain
unchanged. These thresholds are simulation tuning values, not validated hardware
limits. The updated deployment is in `artifacts/recovery_sensitive`.


### Development robustness evaluation

Use `--validation-trials 20` on the curriculum command for 20 paired seeds in
each of the seven existing conditions (140 pairs / 280 flights). Completed seed
pairs are hash-checked and reused. Completion, collision, fallback and 2% mean
lap-time criteria are unchanged. Success is labeled `development validation
passed`; `accepted` remains false because the 100-seed-per-condition full
acceptance requirement has not been met. The active recovery-sensitive run was
migrated to this development mode; include `--validation-trials 20 --resume`
with its original arguments when resuming.


### Faster uncapped trajectory experiment

The previous uncapped reference was a validated feasibility fallback: its
minimum-time candidate failed dense validation. The planner now uses the
remaining mesh refinement for a rejected speed candidate, while preserving the
fastest valid fallback. The same 0.8 planner motor-command bound, 10 cm tracking
margin, collision checks, and dense validation tolerances remain in force.

`./.venv/bin/python scripts/optimize_faster_reference.py` runs a fresh uncapped
solve and compares lap duration and achieved peak speed against the existing
reference. A faster validated result proceeds automatically to one nominal and
four perturbed-start MPC qualification flights. It does not substitute an old RL
checkpoint for a changed reference. The experiment carries the remaining budget
from the interrupted development validation; status and comparison reports are
in `artifacts/faster_reference`. Use `--resume` after interruption.


### Measured-risk racing search

Run `JAX_PLATFORMS=cpu ./.venv/bin/python scripts/run_racing_optimization.py`
from WSL (add `--resume` after interruption). The existing evaluations stay stopped;
completed trials and the incumbent uncapped reference/checkpoint are preserved.
The runner carries the remaining original 24-hour budget from
`artifacts/faster_reference/status.json`; it reserves 40% of that remaining time
for development validation and stops starting search work after the other 60%.
Already running work remains subject to the shared hard deadline.

The default `--risk-schedule` is `0.10:0.80,0.075:0.80,0.075:0.85,0.05:0.90`
(extra margin in metres, normalized planner motor-command limit). Every setting
tests the current-reference seed and gate offsets at +50% and -50% of usable
half-width. The physical gate geometry and all numerical tolerances are unchanged.
`--incumbent-reference` and `--incumbent-checkpoint` select the fixed comparison
controller. MPC-only failures are diagnostics; explicit
`--allow-mpc-baseline-failure` permits recovery RL training against valid references.

Fresh policies train in 16-update blocks, up to 128 updates per candidate. Eight
paired seeds in exact state and 1 cm position noise must all finish without
collisions or fallbacks; candidate mean lap time must improve by at least 1% in
each suite against incumbent RL-MPC. The best qualifying score is the smallest
worst-suite lap-time ratio. MPC-only failures never substitute for an RL pass.
Development validation uses 20 new paired seeds per condition in seven conditions,
with no more than 2% lap-time regression. Full uncapped acceptance is not claimed.

Artifacts are in `artifacts/racing_optimization`. Planner `--diagnostics-dir`
saves every returned solver candidate and reports failed constraints, measured
values, limits, times, and gate indices. Invalid/raw NPZ files cannot be loaded
for training. `--gate-offset-fraction` changes only the seed, not gate apertures.
Old paths remain the fallback; a new path is not a proven real racing advantage.


### Faster RL paths around the 13.82-second reference (currently paused)

`--reference-only` on `run_racing_optimization.py` skips planning and trains a fresh
policy against the unchanged incumbent reference. It enables
`--residual-racing-objective`: +20 per clean gate (up to +2 for a faster segment),
+200 for clean completion plus a bounded lap-time adjustment (-50 to +20),
-2 per simulated second, -500 for crashes/out-of-bounds, and -400 for missed
gates/deadlines. Existing potential progress and action-change terms remain.
Speed rewards occur at clean gate/finish events, so raw high speed, circling or
crashing after a speed burst is not itself rewarded.

On-track spatial exploration starts at 25% of the per-axis 10 cm bound; recovery
can use the full bound. With `--residual-recovery-braking`, positive reference speed requests are limited to 15%,
reduced near gates/turns, and blocked when tracking exceeds the activation
threshold or measured physical frame clearance is <=5 cm. Slowdown is capped at
5% and is enabled only when recovery is active or frame clearance is <=5 cm.
Clean tracking suppresses negative speed requests. These percentages scale
reference speed; they are not guarantees of actual flight speed. These action bounds do not guarantee collision avoidance.
Paired flight screening against the retained controller remains mandatory;
failed or slower policies never replace it. Old policies cannot resume with
the new reward/action mode.

When explicitly resumed, start a NEW experiment with:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false ./.venv/bin/python scripts/run_racing_optimization.py \
  --reference-only --output-dir artifacts/reference_racing_policy \
  --carry-budget-from artifacts/racing_optimization/status.json
```

No process is started by this configuration change. Existing runs remain paused.
