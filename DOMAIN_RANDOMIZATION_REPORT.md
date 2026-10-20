# ANYmal C Rough-Terrain Locomotion and Domain Randomization Report

## Executive summary

This project trains an ANYmal C quadruped in Isaac Lab to follow planar velocity commands while traversing
flat and procedurally generated rough terrain. The final policy can stand, walk forward and backward, move
laterally, turn in place, combine translation with turning, and traverse slopes, stairs, boxes, and random
rough surfaces.

The main experiment tests whether domain randomization (DR) improves robustness to physics conditions that
were not encountered during training. To make the comparison fair, two policies were fine-tuned from the
same checkpoint for the same additional 500 PPO iterations:

- **Nominal control:** continued training with fixed nominal physics.
- **DR policy:** continued training while friction, base mass, center of mass, actuator gains, reset state,
  and external disturbances were randomized.

Both policies were evaluated with the same commands, terrain distribution, environment count, and seed on
nominal physics and on a held-out out-of-distribution (OOD) physics stress test.

The DR policy improved nominal completion from **96.5% to 99.2%** and held-out OOD completion from **85.2%
to 89.1%**. Under OOD physics, it improved survival particularly for straight walking and combined walking
and turning, but produced slightly worse planar tracking error and worse in-place-turning survival. The
result therefore supports a nuanced conclusion: DR improved overall robustness without reducing nominal
performance, but it introduced behavior-specific tradeoffs.

## System and task setup

### Simulator and robot

| Item | Configuration |
|---|---|
| Framework | Isaac Lab with RSL-RL PPO |
| Robot | ANYmal C |
| Parallel training environments | 4,096 |
| Physics time step | 0.005 s (200 Hz) |
| Control decimation | 4 |
| Policy/control rate | 50 Hz |
| Episode duration | 20 s |
| Action dimension | 12 |
| Rough-task observation dimension | 235 |

The 12 actions are joint-position offsets from ANYmal's default standing pose:

```text
joint target = default joint position + 0.5 rad * policy action
```

The rough-task observation vector contains:

- Base linear velocity in the body frame: 3 values
- Base angular velocity in the body frame: 3 values
- Projected gravity in the body frame: 3 values
- Commanded forward/lateral velocity and yaw rate: 3 values
- Joint-position offsets: 12 values
- Joint velocities: 12 values
- Terrain height scan: 187 values
- Previous actions: 12 values

Total: `3 + 3 + 3 + 3 + 12 + 12 + 187 + 12 = 235` observations.

### Sensors

The task uses:

- A contact sensor covering the robot's rigid bodies, with foot air-time tracking.
- A base-mounted ray-caster height scanner.
- A `1.6 m × 1.0 m` scanner grid at `0.1 m` resolution.

The height scan gives the policy local terrain geometry in the robot's yaw-aligned frame. Contact data is
used for termination and undesired thigh/shank collision penalties.

### Terrain

The rough task uses Isaac Lab's procedural rough-terrain generator with difficulty levels 0 through 9.
Every level remains represented during training rather than relying on promotion and demotion. Terrain
families are:

- `pyramid_stairs`
- `pyramid_stairs_inv`
- `boxes`
- `random_rough`
- `hf_pyramid_slope`
- `hf_pyramid_slope_inv`

Episodes terminate on physical base contact. Fixed base-height and tilt termination are not used for the
rough task because crouching and pitching can be valid stair-climbing behavior.

## Command distribution

Commands are body-frame `[v_x, v_y, yaw_rate]` targets sampled from:

| Command | Range |
|---|---:|
| Forward velocity | -1.0 to 2.0 m/s |
| Lateral velocity | -0.5 to 0.5 m/s |
| Yaw rate | -1.0 to 1.0 rad/s |

One command is held for the complete 20-second rough-terrain episode. The command modes are deliberately
sampled as:

| Mode | Probability | Purpose |
|---|---:|---|
| Standing | 10% | Stability at zero command |
| Pure turning | 20% | Explicitly train in-place yaw control |
| Straight translation | 25% | Penalize unnecessary turning while traversing terrain |
| General combined motion | 45% | Train simultaneous translation and yaw tracking |

Pure-turn commands have at least `0.3 rad/s` yaw magnitude. Explicit straight and turning modes were added
because independently sampling continuous command components makes exact zero-yaw translation and exact
zero-translation turning extremely unlikely.

## Rough-terrain reward design

The final rough task is traversal-focused. Rewards that strongly enforce a nominal standing posture were
disabled because they can discourage useful crouching, pitching, and irregular foot placement on stairs.

| Term | Scale | Intent |
|---|---:|---|
| Planar velocity tracking | +2.0 | Track commanded `v_x` and `v_y` |
| Yaw-rate tracking | +1.0 | Track commanded turning rate |
| Vertical velocity | -1.0 | Discourage bouncing |
| Roll/pitch angular velocity | -0.2 | Discourage unstable body rotation |
| Action rate | -0.05 | Encourage smoother commands |
| Joint torque | -2.0e-5 | Reduce excessive effort |
| Joint acceleration | -2.5e-7 | Reduce violent joint motion |
| Thigh/shank contacts | -1.0 | Penalize collisions with obstacles |
| Termination | -2.0 | Penalize falling/base contact |

Base-height, upright-pose, joint-pose, fixed gait, foot-clearance, and feet-slide shaping have zero weight in
the final rough configuration. This follows the principle of rewarding the task outcome while avoiding
constraints that prevent valid terrain-adaptive motion.

## PPO configuration

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Rollout length | 24 steps per environment |
| Actor hidden layers | `[512, 256, 128]` |
| Critic hidden layers | `[512, 256, 128]` |
| Activation | ELU |
| Observation normalization | Disabled |
| Initial action standard deviation | 1.0 |
| Learning rate | 1.0e-3, adaptive schedule |
| Discount factor (`gamma`) | 0.99 |
| GAE parameter (`lambda`) | 0.95 |
| PPO clip parameter | 0.2 |
| Value-loss coefficient | 1.0 |
| Entropy coefficient | 0.005 |
| Learning epochs | 5 |
| Mini-batches | 4 |
| Desired KL divergence | 0.01 |
| Maximum gradient norm | 1.0 |

## What domain randomization means

Domain randomization trains one policy across many simulated versions of the system rather than one exact
model. Each version varies physical parameters that may be uncertain or different in a real deployment.
The objective is to prevent the policy from depending on a narrow simulator-specific solution.

In this project, every parallel environment receives a stable but different physical identity at simulator
startup. This produces thousands of simultaneous variations during training. Reset noise and pushes also
expose the policy to different states and disturbances.

Domain randomization is commonly used for sim-to-real transfer, but a physical robot is not required to test
its underlying principle. Robustness can be evaluated in simulation by testing on physics parameters outside
the training distribution.

## Domain-randomized training distribution

The DR task is registered as:

```text
AnymalTraining-Rough-DR-Direct-v0
```

It retains the same observations, actions, rewards, terrain, commands, and policy architecture as the nominal
rough task. Only the physical and disturbance distributions differ.

### Startup physics randomization

| Parameter | DR training range | Distribution |
|---|---:|---|
| Static friction | 0.60 to 1.30 | Bucketed uniform |
| Dynamic friction | 0.50 to 1.20 | Bucketed uniform |
| Restitution | 0.00 to 0.05 | Bucketed uniform |
| Base mass multiplier | 0.85 to 1.15 | Log-uniform |
| Base CoM offset X | -0.03 to +0.03 m | Uniform |
| Base CoM offset Y | -0.03 to +0.03 m | Uniform |
| Base CoM offset Z | -0.015 to +0.015 m | Uniform |
| Actuator stiffness multiplier | 0.90 to 1.10 | Log-uniform |
| Actuator damping multiplier | 0.90 to 1.10 | Log-uniform |

### Reset and disturbance randomization

| Parameter | DR training range |
|---|---:|
| Initial joint-position noise | ±0.05 rad |
| Initial joint-velocity noise | ±0.05 rad/s |
| Initial base roll/pitch | ±0.035 rad (about 2°) |
| Initial base linear velocity | ±0.05 m/s |
| Initial base angular velocity | ±0.05 rad/s |
| Push interval | 5 to 10 s |
| Maximum horizontal push velocity | 0.2 m/s |

“DR” in the reported experiment refers to this complete bundle. The experiment does not isolate the causal
effect of each individual randomized parameter.

## Held-out OOD evaluation distribution

The evaluation-only task is:

```text
AnymalTraining-Rough-OOD-Direct-v0
```

It was never used for policy training. Every tested physics range lies outside the corresponding DR training
support:

| Parameter | DR training support | Held-out OOD range |
|---|---:|---:|
| Static friction | 0.60 to 1.30 | 0.35 to 0.55 |
| Dynamic friction | 0.50 to 1.20 | 0.30 to 0.45 |
| Restitution | 0.00 to 0.05 | 0.10 to 0.20 |
| Base mass multiplier | 0.85 to 1.15 | 1.20 to 1.35 |
| Base CoM offset X | -0.03 to +0.03 m | +0.04 to +0.06 m |
| Base CoM offset Y | -0.03 to +0.03 m | -0.06 to -0.04 m |
| Base CoM offset Z | -0.015 to +0.015 m | +0.02 to +0.03 m |
| Stiffness multiplier | 0.90 to 1.10 | 0.75 to 0.88 |
| Damping multiplier | 0.90 to 1.10 | 1.12 to 1.25 |

This is intentionally a difficult compound stress test: the robot is simultaneously heavier, less stiff,
more damped, more asymmetric, bouncier, and placed on lower-friction contacts. It measures extrapolation,
not interpolation inside the training distribution.

Evaluation disables pushes for all policies and uses the nominal rough-task reset state. The OOD comparison
therefore isolates held-out physics rather than combining physics changes with random external disturbances.

## Experimental design

### Initial checkpoint

The shared starting checkpoint was:

```text
logs/rsl_rl/anymal_training_rough_direct/2026-08-31_09-53-02/model_1998.pt
```

### Matched fine-tuning branches

Both branches started from `model_1998.pt` and received exactly 500 additional PPO iterations, ending at
`model_2497.pt`:

| Branch | Training task | Run name | Added iterations |
|---|---|---|---:|
| Nominal control | `AnymalTraining-Rough-Direct-v0` | `nominal_control_500` | 500 |
| Domain randomized | `AnymalTraining-Rough-DR-Direct-v0` | `dr_finetune_v1` | 500 |

This matched control removes the major confound in the initial comparison, where only the DR policy had
received an additional 500 iterations.

### Evaluation protocol

Each final checkpoint was evaluated using:

- 256 parallel environments
- One completed episode per environment
- Seed 42
- 20-second maximum episode duration
- The same terrain generator and levels 0 through 9
- Four fixed command cases, evenly assigned across environments:
  - Standing: `[0.0, 0.0, 0.0]`
  - Straight: `[1.0, 0.0, 0.0]`
  - Turn left: `[0.0, 0.0, 0.7]`
  - Combined: `[1.0, 0.0, 0.5]`

Reported metrics are:

- **Completion rate:** fraction of episodes reaching the time limit without base-contact termination.
- **Fall rate:** fraction terminating through base contact.
- **Planar tracking error:** mean norm of commanded minus measured body-frame XY velocity.
- **Yaw tracking error:** mean absolute commanded minus measured body-frame yaw rate.

Raw episode records and grouped summaries are stored as CSV and JSON beside each checkpoint.

## Results

### Primary matched comparison

| Training policy | Evaluation physics | Completion | Fall rate | Planar error | Yaw error |
|---|---|---:|---:|---:|---:|
| Nominal control | Nominal | 96.5% | 3.5% | 0.072 m/s | 0.072 rad/s |
| DR policy | Nominal | **99.2%** | **0.8%** | **0.067 m/s** | **0.064 rad/s** |
| Nominal control | Held-out OOD | 85.2% | 14.8% | **0.155 m/s** | 0.131 rad/s |
| DR policy | Held-out OOD | **89.1%** | **10.9%** | 0.175 m/s | **0.126 rad/s** |

On nominal physics, the DR policy improved completion by **2.7 percentage points** and slightly improved both
tracking metrics. On held-out OOD physics, it improved completion by **3.9 percentage points** and reduced
the fall rate by the same amount. OOD yaw tracking improved slightly, while planar tracking worsened by
`0.020 m/s`.

### Results by command mode

#### Nominal physics

| Command | Control completion | DR completion | Control planar error | DR planar error | Control yaw error | DR yaw error |
|---|---:|---:|---:|---:|---:|---:|
| Standing | 100.0% | 100.0% | 0.004 | 0.005 | 0.016 | **0.013** |
| Straight | 96.9% | **100.0%** | **0.101** | 0.104 | **0.080** | 0.082 |
| Turn left | 98.4% | **100.0%** | 0.049 | **0.048** | 0.075 | **0.068** |
| Combined | 90.6% | **96.9%** | 0.133 | **0.112** | 0.116 | **0.094** |

#### Held-out OOD physics

| Command | Control completion | DR completion | Control planar error | DR planar error | Control yaw error | DR yaw error |
|---|---:|---:|---:|---:|---:|---:|
| Standing | 100.0% | 100.0% | **0.006** | 0.020 | **0.021** | 0.033 |
| Straight | 70.3% | **79.7%** | **0.274** | 0.285 | 0.176 | **0.128** |
| Turn left | **98.4%** | 90.6% | **0.070** | 0.077 | **0.106** | 0.134 |
| Combined | 71.9% | **85.9%** | **0.272** | 0.319 | 0.222 | **0.209** |

The main OOD robustness benefit appears in translational locomotion:

- Straight-walking completion improved by **9.4 percentage points**.
- Combined walking-and-turning completion improved by **14.0 percentage points**.
- Straight-command yaw error improved from `0.176` to `0.128 rad/s`.

The main tradeoff is isolated OOD turning, where completion decreased from 98.4% to 90.6%. The DR policy
also prioritizes survival over precise planar tracking in the hardest OOD locomotion cases.

## Interpretation

### Why could DR improve nominal performance?

It is reasonable to expect randomization to reduce peak nominal performance because the policy must solve a
broader problem. That did not happen in this run. Plausible explanations include:

- Randomization acted as regularization and discouraged brittle simulator-specific behaviors.
- Reset variation and pushes exposed recovery states missing from nominal training.
- The broader experience distribution led PPO to a better local optimum.
- Stochastic variation between two single training branches may account for part of the difference.

The matched iteration budget rules out “the DR policy simply trained 500 iterations longer” as the full
explanation. It does not prove that DR itself caused every nominal improvement because the experiment has one
training seed per branch.

### What the results support

The results support the following claims:

1. The learned controller operates across all rough-terrain families and levels in the evaluation set.
2. Moderate DR fine-tuning did not damage nominal performance in this experiment.
3. DR improved overall survival under a compound physics distribution outside its training support.
4. The robustness improvement is behavior-dependent rather than universal.
5. DR particularly benefited forward and combined locomotion under OOD dynamics.

The results do **not** establish real-world transfer because no hardware experiment was performed.

## Limitations

- Each training branch was run once. Multiple training seeds are required for confidence intervals and a
  causal claim about DR.
- Evaluation used one seed and 256 episodes. More evaluation seeds would reduce sampling uncertainty.
- Only leftward turning was explicitly evaluated; rightward turning should be included in a complete test.
- The OOD task changes several parameters simultaneously and represents one difficult corner of the possible
  physics space. It does not identify which parameter causes each failure.
- DR is a bundle of physics variation, reset noise, and pushes. An ablation study would be needed to quantify
  each component.
- No observation noise, sensor latency, actuator latency, terrain perception error, or communication delay was
  randomized.
- Simulation robustness is evidence of reduced model sensitivity, not proof of sim-to-real transfer.

## Reproduction commands

Run commands from the `AnymalTraining` repository root.

### Nominal control fine-tune

```powershell
isaaclab -p .\scripts\rsl_rl\train.py `
  --task AnymalTraining-Rough-Direct-v0 `
  --run_name nominal_control_500 `
  --resume `
  --checkpoint "C:\path\to\model_1998.pt" `
  --max_iterations 500 `
  --viz none
```

### DR fine-tune

```powershell
isaaclab -p .\scripts\rsl_rl\train.py `
  --task AnymalTraining-Rough-DR-Direct-v0 `
  --run_name dr_finetune_v1 `
  --resume `
  --checkpoint "C:\path\to\model_1998.pt" `
  --max_iterations 500 `
  --viz none
```

### Nominal evaluation

```powershell
isaaclab -p .\scripts\rsl_rl\evaluate.py `
  --task AnymalTraining-Rough-Direct-v0 `
  --checkpoint "C:\path\to\model_2497.pt" `
  --num_envs 256 `
  --episodes_per_env 1 `
  --seed 42 `
  --viz none
```

### Held-out OOD evaluation

```powershell
isaaclab -p .\scripts\rsl_rl\evaluate.py `
  --task AnymalTraining-Rough-OOD-Direct-v0 `
  --checkpoint "C:\path\to\model_2497.pt" `
  --num_envs 256 `
  --episodes_per_env 1 `
  --seed 42 `
  --viz none
```

## Artifact locations

### Final checkpoints

- Nominal control: `logs/rsl_rl/anymal_training_rough_direct/2026-08-31_15-54-40_nominal_control_500/model_2497.pt`
- DR policy: `logs/rsl_rl/anymal_training_rough_dr_direct/2026-08-31_12-44-38_dr_finetune_v1/model_2497.pt`

### Evaluation summaries

- Nominal control on nominal physics:
  `logs/rsl_rl/anymal_training_rough_direct/2026-08-31_15-54-40_nominal_control_500/evaluations/2026-08-31_16-34-13/summary.json`
- Nominal control on OOD physics:
  `logs/rsl_rl/anymal_training_rough_direct/2026-08-31_15-54-40_nominal_control_500/evaluations/2026-08-31_16-35-51/summary.json`
- DR policy on nominal physics:
  `logs/rsl_rl/anymal_training_rough_dr_direct/2026-08-31_12-44-38_dr_finetune_v1/evaluations/2026-08-31_15-19-38/summary.json`
- DR policy on OOD physics:
  `logs/rsl_rl/anymal_training_rough_dr_direct/2026-08-31_12-44-38_dr_finetune_v1/evaluations/2026-08-31_16-37-18/summary.json`

## Recommended follow-up work

1. Repeat both 500-iteration branches with at least three training seeds.
2. Evaluate every checkpoint with several evaluation seeds and report means with confidence intervals.
3. Add right-turn, backward, lateral, and multiple-speed evaluation scenarios.
4. Separate the OOD stress test into friction, payload/CoM, and actuator-dynamics suites.
5. Run DR ablations: physics only, reset noise only, pushes only, and the complete bundle.
6. Add realistic observation noise and control latency if sim-to-real preparation becomes the objective.
