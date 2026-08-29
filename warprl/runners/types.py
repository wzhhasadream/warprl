from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EnvironmentType = Literal[
    "mujoco",
    "myosuite",
    "dmc",
    "humanoid_bench",
    "playground",
    "maniskill",
    "isaaclab",
    "mjlab",
]



@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment metadata required by the off-policy runner."""

    env_type: EnvironmentType
    seed: int
    eval_episode: int



@dataclass(frozen=True)
class OffPolicyRunnerConfig:
    """Training-loop controls independent from the environment implementation."""

    total_timesteps: int
    grad_step_per_interaction_step: float
    num_eval: int = 20
    num_log: int = 50
    record_video: bool = False
    save_agent: bool = False
    save_onnx: bool = False


@dataclass(frozen=True)
class OnPolicyRunnerConfig:
    """Training-loop controls for on-policy rollout-collection algorithms."""

    total_timesteps: int
    rollout_steps: int
    gamma: float = 0.99
    num_eval: int = 20
    num_log: int = 50
    record_video: bool = False
    save_agent: bool = False
    save_onnx: bool = False
