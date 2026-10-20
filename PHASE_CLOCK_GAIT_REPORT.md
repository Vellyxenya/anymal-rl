# Phase-Conditioned Gait Experiment

## Executive summary

The existing rough-terrain policy walked reliably, but its touchdown timing retained a measurable phase error.
Changing only the scalar contact-duration reward moved errors between leg pairs and an event-only touchdown-phase
reward made the aggregate phase MAE worse. The successful intervention was to make the desired gait phase
observable to the policy and provide dense feedback for following its contact schedule.

After 100 PPO fine-tuning iterations, the selected phase-clock policy reduced touchdown phase MAE by **41.7%**,
reduced the composite trot-deviation index by **37.7%**, and reduced stride-period variability by **95.5%**.
The controlled evaluation had zero falls and essentially unchanged velocity tracking. On the broader generated
rough-terrain evaluation, completion decreased slightly from 98.4% to 97.7%, exposing a small regularity-versus-
adaptability tradeoff rather than hiding it.

## Problem and previous negative result

The reference policy was:

`logs/rsl_rl/anymal_training_rough_direct/2026-09-13_11-08-00_gait_timing_scale035_error020_500/model_2996.pt`

Its controlled touchdown phase MAE was 0.02881 cycles. The original gait reward compares contact and air
durations between feet, but it does not give the policy a persistent cyclic reference. An attempted reward based
only on touchdown events worsened phase MAE to 0.04366 cycles. Its feedback was sparse, and the policy could not
observe the phase against which those events were judged.

## Implemented design

The phase-clock task is registered separately as:

`AnymalTraining-Rough-PhaseClock-Direct-v0`

This preserves the original flat and rough tasks. The new task makes three deliberate changes:

1. A 2.5 Hz clock advances with episode time. This corresponds to a 0.4 s stride, close to the reference
   policy's measured 0.39 s stride.
2. The policy receives `sin(2*pi*phase)` and `cos(2*pi*phase)`, increasing the observation dimension from 235
   to 237 without introducing a discontinuity at the cycle boundary.
3. A dense contact-schedule reward encourages LF+RH and RF+LH to form alternating diagonal pairs with a 0.55
   duty factor. It is active only for straight forward commands (`vx > 0.3 m/s`, `|vy| < 0.1 m/s`, and
   `|yaw rate| < 0.1 rad/s`), leaving turning, standing, and lateral behavior unconstrained by the clock.

The contact-schedule reward scale is 0.75. All established rough-terrain locomotion rewards remain present.

## Checkpoint migration

The new observations change only the first network layer. The migration utility appends two zero columns to
the actor and critic first-layer weights and to their Adam first/second moments. Consequently, the two new
inputs initially contribute exactly zero and the migrated policy starts with behavior equivalent to the
reference checkpoint. All remaining parameters and optimizer state are retained.

Migration command:

```powershell
isaaclab -p .\scripts\rsl_rl\migrate_phase_clock_checkpoint.py `
  ".\logs\rsl_rl\anymal_training_rough_direct\2026-09-13_11-08-00_gait_timing_scale035_error020_500\model_2996.pt" `
  ".\checkpoints\phase_clock\baseline_model_2996_obs237.pt"
```

## Training setup

- Starting point: migrated reference `model_2996.pt`
- Environment: 4,096 parallel generated rough-terrain environments
- PPO rollout: 24 policy steps per environment
- Actor/critic: `[512, 256, 128]`, ELU activation
- Physics: 200 Hz
- Policy control: 50 Hz (`decimation=4`)
- Seed: 42
- Selected checkpoint: iteration 3100, after 104 logged iterations / approximately 100 intended updates
- Selection rule: earliest checkpoint showing a clear controlled gait improvement without meaningful tracking loss

Training command:

```powershell
isaaclab -p .\scripts\rsl_rl\train.py `
  --task AnymalTraining-Rough-PhaseClock-Direct-v0 `
  --run_name phase_clock_contact_schedule_500 `
  --resume `
  --checkpoint ".\checkpoints\phase_clock\baseline_model_2996_obs237.pt" `
  --max_iterations 500 `
  --viz none
```

Training was deliberately early-stopped after evaluating `model_3100.pt`; continuing to an arbitrary iteration
budget could degrade the gait after the target metric had already improved.

## Controlled gait result

Protocol: 64 paired trials, seed 42, flat controlled terrain, fixed 1.0 m/s forward command, 2 s warm-up, and
10 s measurement. Confidence intervals use 20,000 paired bootstrap resamples.

| Metric | Reference | Phase clock | Change | Paired 95% CI |
|---|---:|---:|---:|---:|
| Touchdown phase MAE [cycles] ↓ | 0.02881 | **0.01680** | **-41.7%** | [-0.01221, -0.01182] |
| Trot deviation index ↓ | 0.06056 | **0.03774** | **-37.7%** | [-0.02312, -0.02250] |
| Duty-factor CV ↓ | 0.10127 | **0.08262** | **-18.4%** | [-0.01908, -0.01821] |
| Duty-factor range ↓ | 0.14441 | **0.10078** | **-30.2%** | [-0.04428, -0.04297] |
| Stride-period CV ↓ | 0.02573 | **0.00116** | **-95.5%** | [-0.02508, -0.02398] |
| LF-RH agreement ↑ | 0.86344 | **0.90075** | +0.03731 | [0.03650, 0.03813] |
| RF-LH agreement ↑ | 0.88063 | **0.89959** | +0.01897 | [0.01816, 0.01981] |
| Diagonal opposition ↑ | 0.98541 | **0.99913** | +0.01372 | [0.01300, 0.01444] |
| Velocity tracking MAE [m/s] ↓ | **0.03231** | 0.03251 | +0.00020 | [0.00014, 0.00026] |
| Fall rate ↓ | 0% | **0%** | 0 | — |

The 0.00020 m/s tracking change is practically negligible, while every primary gait metric improved and its
paired confidence interval excludes zero.

## Generated rough-terrain check

Protocol: 256 environments, one episode each, seed 42, identical evaluator settings for both checkpoints.

| Metric | Reference | Phase clock | Change |
|---|---:|---:|---:|
| Completion ↑ | **98.4%** | 97.7% | -0.7 points |
| Fall rate ↓ | **1.6%** | 2.3% | +0.7 points |
| Planar velocity error ↓ | **0.062 m/s** | 0.070 m/s | +0.008 m/s |
| Yaw-rate error ↓ | 0.065 rad/s | 0.065 rad/s | unchanged |

This modest degradation is the expected cost of imposing a more regular periodic gait on terrain where adaptive
foot timing can be useful. Because the phase reward is gated to straight forward motion, other command modes
remain free of the prescribed schedule.

## Selected model and reproducibility

Selected model:

`checkpoints/phase_clock/model_3100_selected.pt`

SHA-256:

`A32C4D11CE404C04675A0C16A419AFB0A59EA580430B523CF3390D6B863007FD`

Controlled evaluation:

```powershell
isaaclab -p .\scripts\rsl_rl\evaluate_gait.py `
  --task AnymalTraining-Rough-PhaseClock-Direct-v0 `
  --checkpoint ".\checkpoints\phase_clock\model_3100_selected.pt" `
  --num_envs 64 --seed 42 --warmup 2 --duration 10 --command_vx 1.0 `
  --output_dir ".\gait_evaluation\phase_clock_3100" --viz none
```

Play interactively:

```powershell
isaaclab -p .\scripts\rsl_rl\play.py `
  --task AnymalTraining-Rough-PhaseClock-Direct-v0 `
  --checkpoint ".\checkpoints\phase_clock\model_3100_selected.pt" `
  --num_envs 1 --viz kit
```

Raw controlled results are in `gait_evaluation/phase_clock_3100`; the paired comparison is in
`gait_evaluation/comparison_best_vs_phase_clock_3100.json`. Matched rough-terrain results are in
`evaluations/baseline_model_2996_rough_matched` and `evaluations/phase_clock_3100_rough`.

## Defensible conclusion

The experiment demonstrates that the residual gait defect was not best solved by repeatedly tuning an indirect
scalar timing reward. Providing an observable phase reference plus dense schedule feedback directly addressed
the measured failure: phase MAE fell by 41.7% with broad improvements in symmetry and periodicity. The broader
terrain check also revealed the genuine engineering tradeoff—the more regular gait was marginally less adaptive
on generated rough terrain—which should guide future work toward terrain-dependent phase flexibility rather than
claiming an unconditional improvement.
