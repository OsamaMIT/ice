# A2RL Racing Drone Training

This package scaffolds an autonomous FPV racing drone trainer using JAX for the policy, critic, optimizer, and PPO math, with Crazyflow as the drone simulator.

## System Shape

- Environment: Crazyflow `Sim` in attitude-control mode.
- Policy action: normalized FPV sticks `[throttle, roll, pitch, yaw_rate]` in `[-1, 1]`.
- Crazyflow command: `[roll, pitch, yaw, collective_thrust]`, with yaw integrated from the yaw-rate stick.
- Observation: a compact estimator state that can be produced from IMU plus camera PNP over segmented gate masks.
- Actor: G&CNet-style gate-conditioned actor with separate state and gate encoders.
- Critic: value network trained with PPO clipped value loss.
- Trainer: vectorized rollout collection, GAE, squashed-Gaussian PPO updates.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Crazyflow pulls in MuJoCo, JAX, and its drone model/controller packages. For GPU training, install the JAX CUDA wheel matching your system before running training.

## Train

```bash
a2rl-drone-train \
  --num-envs 128 \
  --total-env-steps 2000000 \
  --horizon 128 \
  --course arena_38m_stacked \
  --device cpu \
  --physics so_rpy_rotor_drag \
  --checkpoint-dir checkpoints
```

The default course is `arena_38m_stacked`: a 38 m x 38 m arena with 12 runtime gates, including stacked top/bottom openings for logical gates 7 and 10. A smaller `compact_slalom` course remains available for fast debugging. Replace either by passing a custom `GateCourse` to `PPOTrainer`.

Training starts with two curricula by default. The gate-window curriculum scales the 1.5 m x 1.5 m pass opening up to the 2.7 m outer frame and anneals back to the strict opening. The start-gate curriculum resets episodes before random gates early in training, then anneals back to full-course starts. Disable them with `--no-curriculum` and `--no-start-gate-curriculum` for strict full-course runs.

The arena reward includes time pressure, early gate-pass bonuses, lookahead progress, centered crossing bonuses, and stall penalties so the policy is pushed toward chaining gates instead of hovering near safe partial-course behavior. Training logs include `speed`, `fwd_speed`, `ttg`, and `stall` to diagnose whether the policy is moving usefully toward the next gate.

The default episode horizon is 24 seconds. Tune it with `--max-episode-time`.

Resume from a checkpoint with `--restore-checkpoint`. `--total-env-steps` is treated as the final target, including restored steps. For example, restoring a 6M-step checkpoint with `--total-env-steps 12000000` runs roughly 6M additional steps:

```bash
a2rl-drone-train \
  --course arena_38m_stacked \
  --num-envs 128 \
  --total-env-steps 12000000 \
  --horizon 128 \
  --physics so_rpy_rotor_drag \
  --restore-checkpoint checkpoints_arena_start_gate_curriculum/checkpoint_000366.pkl \
  --checkpoint-dir checkpoints_arena_resume
```

## Observation Layout

For each environment, the observation vector is:

- IMU/state-estimator core: body gyro, body specific force, attitude quaternion, body velocity, body gravity, previous action, course progress, remaining episode fraction.
- Gate context for the next `N` gates: relative gate center in body/camera frame, gate normal in body frame, image-plane bearing, mask/PNP visibility, and distance.

This mirrors a real stack where IMU provides angular velocity and specific force, the state estimator provides attitude and velocity, and gate masks feed PNP to recover relative gate pose.

## Notes

The trainer is intentionally plain JAX instead of a large RL framework. That keeps the actor/critic math explicit and makes it easier to adapt the observation layout, reward terms, or action adapter to a real drone control stack.


## Eval
```
PYTHONPATH=src .venv/bin/python scripts/eval_chase_video.py \
  --checkpoint-dir checkpoints_arena_path_alignment \
  --course arena_38m_stacked \
  --output artifacts/eval_chase_arena_38m_stacked.mp4 \
  --num-eval-envs 64 \
  --seed 123 \
  --max-episode-time 24 \
  --device cpu \
  --fps 20 \
  --stride 5 \
  --view-radius 8
```