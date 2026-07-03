from collections import deque
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from .__init__ import Batch, Transition
from .base_buffer import BaseBuffer


_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}


def _to_torch_dtype(dtype: Any) -> torch.dtype:
    dtype = np.dtype(dtype)
    return _NP_TO_TORCH_DTYPE.get(dtype, torch.float32)


class TorchBuffer(BaseBuffer):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_size: int = int(1e6),
        linear_decay_step: int = 0,
        min_weight: float = 0.1,
        n_step: int = 1,
        gamma: float = 0.99,
        num_envs: int = 1,
        use_approximate_sampling: bool = True,
        num_buckets: int = 2000,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__(observation_space, action_space, max_size)
        self.linear_decay_step = linear_decay_step
        self.abs_linear_decay_step = abs(linear_decay_step)
        self.n_step = n_step
        self.gamma = gamma
        self.min_weight = min_weight
        self.num_envs = num_envs
        self.use_approximate_sampling = (
            use_approximate_sampling and self.max_size % self.num_envs == 0
        )
        self.num_buckets = num_buckets
        self.device = torch.device(device)

        assert self.n_step >= 1, f"n_step must be positive, got {self.n_step}"
        assert self.num_envs >= 1, f"num_envs must be positive, got {self.num_envs}"
        assert self.max_size >= self.num_envs, f"max_size must be >= num_envs, got {self.max_size} and {self.num_envs}"
        assert self.num_buckets >= 1, f"num_buckets must be positive, got {self.num_buckets}"
        assert 0 <= self.min_weight <= 1, f"min_weight must be in [0, 1], got {self.min_weight}"

        obs_dtype = _to_torch_dtype(self.obsveration_space.dtype)
        action_dtype = _to_torch_dtype(self.action_space.dtype)
        self.obsverations = torch.empty((self.max_size, *self.obsveration_shape), dtype=obs_dtype, device=self.device)
        self.next_obsverations = torch.empty(
            (self.max_size, *self.obsveration_shape), dtype=obs_dtype, device=self.device)
        self.actions = torch.empty((self.max_size, *self.action_shape), dtype=action_dtype, device=self.device)
        self.rewards = torch.empty((self.max_size,), dtype=torch.float32, device=self.device)
        self.terminations = torch.empty((self.max_size,), dtype=torch.float32, device=self.device)
        self.trunactions = torch.empty((self.max_size,), dtype=torch.float32, device=self.device)
        self.discounts = torch.empty((self.max_size,), dtype=torch.float32, device=self.device)
        self.deque: deque[Transition] = deque(maxlen=self.n_step)
        self.ptr = 0
        self.size = 0
        self.full = False
        if self.linear_decay_step != 0:
            self.timestamps = torch.empty((self.max_size,), dtype=torch.int64, device=self.device)
            self.current_time = 0

    def _tensor(self, value: Any, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device).reshape(shape)

    def _as_transition(self, transition: Transition) -> Transition:
        return Transition(
            observations=self._tensor(
                transition.observations, self.obsverations.dtype, (self.num_envs, *self.obsveration_shape)),
            actions=self._tensor(transition.actions, self.actions.dtype, (self.num_envs, *self.action_shape)),
            rewards=self._tensor(transition.rewards, torch.float32, (self.num_envs,)),
            truncations=self._tensor(transition.truncations, torch.float32, (self.num_envs,)),
            terminations=self._tensor(transition.terminations, torch.float32, (self.num_envs,)),
            next_observations=self._tensor(
                transition.next_observations, self.next_obsverations.dtype, (self.num_envs, *self.obsveration_shape)),
        )

    def _get_n_step_transition(self) -> tuple[Transition, torch.Tensor]:
        prev_transition = self.deque[0]
        curr_transition = self.deque[-1]

        n_step_reward = curr_transition.rewards.clone()
        n_step_termination = curr_transition.terminations.clone()
        n_step_truncation = curr_transition.truncations.clone()
        n_step_next_observation = curr_transition.next_observations.clone()
        effective_n_steps = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        for idx in reversed(range(len(self.deque) - 1)):
            transition = self.deque[idx]
            episode_end = transition.terminations.bool() | transition.truncations.bool()
            n_step_reward = transition.rewards + self.gamma * n_step_reward * (~episode_end).float()

            n_step_termination[episode_end] = transition.terminations[episode_end]
            n_step_truncation[episode_end] = transition.truncations[episode_end]
            n_step_next_observation[episode_end] = transition.next_observations[episode_end]
            effective_n_steps = torch.where(episode_end, torch.ones_like(effective_n_steps), effective_n_steps + 1.0)

        n_step_transition = Transition(
            observations=prev_transition.observations,
            actions=prev_transition.actions,
            rewards=n_step_reward,
            terminations=n_step_termination,
            truncations=n_step_truncation,
            next_observations=n_step_next_observation,
        )
        discounts = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), effective_n_steps)
        return n_step_transition, discounts

    def _store(self, transition: Transition, discount: torch.Tensor) -> None:
        add_count = int(transition.rewards.shape[0])
        add_indices = (torch.arange(add_count, device=self.device) + self.ptr) % self.max_size
        old_size = self.size

        self.obsverations[add_indices] = transition.observations
        self.actions[add_indices] = transition.actions
        self.rewards[add_indices] = transition.rewards
        self.terminations[add_indices] = transition.terminations
        self.trunactions[add_indices] = transition.truncations
        self.next_obsverations[add_indices] = transition.next_observations
        self.discounts[add_indices] = discount

        if self.linear_decay_step != 0:
            self.timestamps[add_indices] = self.current_time
            self.current_time += 1

        self.ptr = (self.ptr + add_count) % self.max_size
        self.size = min(self.size + add_count, self.max_size)
        self.full = self.full or old_size + add_count >= self.max_size

    def add(self, transition: Transition) -> None:
        transition = self._as_transition(transition)
        self.deque.append(transition)

        if len(self.deque) >= self.n_step:
            n_step_transition, discount = self._get_n_step_transition()
            self._store(n_step_transition, discount)

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        if self.linear_decay_step == 0:
            return torch.randint(0, self.size, (batch_size,), device=self.device)

        if self.use_approximate_sampling:
            return self._sample_indices_with_approximate_bias(batch_size)

        return self._sample_indices_with_bias(batch_size)

    def _linear_weights(self, timestamps: torch.Tensor) -> torch.Tensor:
        age = (self.current_time - timestamps).float()
        if self.linear_decay_step > 0:
            return torch.clamp(1.0 - age / self.abs_linear_decay_step, min=self.min_weight)
        return torch.clamp(self.min_weight + age / self.abs_linear_decay_step, max=1.0)

    def _sample_indices_with_bias(self, batch_size: int) -> torch.Tensor:
        weights = self._linear_weights(self.timestamps[:self.size])
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = torch.ones_like(weights)
            weight_sum = weights.sum()
        probabilities = weights / weight_sum
        return torch.multinomial(probabilities, batch_size, replacement=True)

    def _sample_indices_with_approximate_bias(self, batch_size: int) -> torch.Tensor:
        bucket_size = max((self.size + self.num_buckets - 1) // self.num_buckets, 1)

        logical_starts = torch.arange(0, self.size, bucket_size, device=self.device)
        logical_ends = torch.clamp(logical_starts + bucket_size, max=self.size)
        logical_midpoints = (logical_starts + logical_ends - 1) // 2
        if self.full:
            bucket_midpoints = (self.ptr + logical_midpoints) % self.max_size
        else:
            bucket_midpoints = logical_midpoints

        bucket_weights = self._linear_weights(self.timestamps[bucket_midpoints])
        bucket_weight_sum = bucket_weights.sum()
        if bucket_weight_sum <= 0:
            bucket_weights = torch.ones_like(bucket_weights)
            bucket_weight_sum = bucket_weights.sum()
        bucket_probabilities = bucket_weights / bucket_weight_sum

        sampled_buckets = torch.multinomial(bucket_probabilities, batch_size, replacement=True)
        sampled_starts = logical_starts[sampled_buckets]
        sampled_ends = logical_ends[sampled_buckets]
        offsets = (torch.rand((batch_size,), device=self.device) * (sampled_ends - sampled_starts).float()).long()
        logical_indices = sampled_starts + offsets
        if self.full:
            return (self.ptr + logical_indices) % self.max_size
        return logical_indices

    def can_sample(self) -> bool:
        return self.size > 0

    def sample(self, batch_size: int) -> Batch:
        assert self.can_sample(), "Cannot sample from an empty buffer"
        indices = self._sample_indices(batch_size)
        return Batch(
            observations=self.obsverations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices, None],
            dones=self.terminations[indices, None],
            next_observations=self.next_obsverations[indices],
            discounts=self.discounts[indices, None],
        )

    def reset(self) -> None:
        self.deque.clear()
        self.ptr = 0
        self.size = 0
        self.full = False
        if self.linear_decay_step != 0:
            self.current_time = 0

    def save(self, path: str) -> None:
        torch.save(
            {
                "observations": self.obsverations[:self.size],
                "actions": self.actions[:self.size],
                "rewards": self.rewards[:self.size],
                "terminations": self.terminations[:self.size],
                "truncations": self.trunactions[:self.size],
                "next_observations": self.next_obsverations[:self.size],
                "discounts": self.discounts[:self.size],
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path)
        for key, value in data:
            setattr(self, key, value)

    def __len__(self) -> int:
        return self.size
