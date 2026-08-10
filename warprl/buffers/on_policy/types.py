"""Shared data containers for feedforward on-policy rollouts."""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = jax.Array | torch.Tensor


class RolloutTransition(NamedTuple):
    """One vectorized environment step collected by the current policy."""

    observations: Tensor  # [num_envs, *observation_shape]
    actions: Tensor  # [num_envs, *action_shape]
    rewards: Tensor  # [num_envs]
    terminated: Tensor  # [num_envs], true MDP terminations
    truncated: Tensor  # [num_envs], time-limit truncations
    values: Tensor  # [num_envs] or [num_envs, 1], V(s_t)
    log_probs: Tensor  # [num_envs] or [num_envs, 1], log pi(a_t | s_t)
    actions_mean: Tensor  # [num_envs, *action_shape]
    actions_std: Tensor  # [num_envs, *action_shape]


class Trajectory(NamedTuple):
    """A complete rollout stored with time as the leading dimension."""

    observations: Tensor  # [num_steps, num_envs, *observation_shape]
    actions: Tensor  # [num_steps, num_envs, *action_shape]
    actions_mean: Tensor  # [num_steps, num_envs, *action_shape]
    actions_std: Tensor  # [num_steps, num_envs, *action_shape]
    log_probs: Tensor  # [num_steps, num_envs]
    values: Tensor  # [num_steps + 1, num_envs]
    dones: Tensor  # [num_steps, num_envs]
    rewards: Tensor  # [num_steps, num_envs]


class RolloutBatch(NamedTuple):
    """A PPO mini-batch, optionally stacked over several update steps."""

    observations: Tensor  # [batch_size, *observation_shape] or [num_batches, batch_size, *observation_shape]
    actions: Tensor  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    actions_mean: Tensor  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    actions_std: Tensor  # [batch_size, *action_shape] or [num_batches, batch_size, *action_shape]
    values: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]
    advantages: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]
    returns: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]
    old_log_probs: Tensor  # [batch_size, 1] or [num_batches, batch_size, 1]

__all__ = ["RolloutBatch", "RolloutTransition", "Trajectory"]
