# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="AnymalTraining-Direct-v0",
    entry_point=f"{__name__}.anymaltraining_env:AnymaltrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymaltraining_env_cfg:AnymaltrainingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="AnymalTraining-Rough-Direct-v0",
    entry_point=f"{__name__}.anymaltraining_env:AnymaltrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymaltraining_env_cfg:AnymaltrainingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="AnymalTraining-Rough-DR-Direct-v0",
    entry_point=f"{__name__}.anymaltraining_env:AnymaltrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.anymaltraining_env_cfg:AnymaltrainingRoughDomainRandomizedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DomainRandomizedRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AnymalTraining-Rough-PhaseClock-Direct-v0",
    entry_point=f"{__name__}.anymaltraining_env:AnymaltrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymaltraining_env_cfg:AnymaltrainingPhaseClockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PhaseClockRoughPPORunnerCfg",
    },
)

gym.register(
    id="AnymalTraining-Rough-OOD-Direct-v0",
    entry_point=f"{__name__}.anymaltraining_env:AnymaltrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymaltraining_env_cfg:AnymaltrainingRoughOODEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
