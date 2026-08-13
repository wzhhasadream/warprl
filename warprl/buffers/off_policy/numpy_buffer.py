from .base_buffer import BaseBuffer
from gymnasium import spaces
import numpy as np
from collections import deque
from .types import Batch, Transition


class NumpyBuffer(BaseBuffer):
    def __init__(self,
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
        assert self.n_step >= 1, f"n_step must be positive, got {self.n_step}"
        assert self.num_envs >= 1, f"num_envs must be positive, got {self.num_envs}"
        assert self.max_size >= self.num_envs, f"max_size must be >= num_envs, got {self.max_size} and {self.num_envs}"
        assert self.num_buckets >= 1, f"num_buckets must be positive, got {self.num_buckets}"
        assert 0 <= self.min_weight <= 1, f"min_weight must be in [0, 1], got {self.min_weight}"
        self.obsverations = np.empty((self.max_size, * self.obsveration_shape), dtype=self.obsveration_space.dtype)
        self.next_obsverations = np.empty((self.max_size, * self.obsveration_shape), dtype=self.obsveration_space.dtype)
        self.actions = np.empty((self.max_size, * self.action_shape), dtype=self.action_space.dtype)
        self.rewards = np.empty((self.max_size, ), dtype=np.float32)
        self.terminations = np.empty((self.max_size, ), dtype=np.float32)
        self.trunactions = np.empty((self.max_size, ), dtype=np.float32)
        self.discounts = np.empty((self.max_size, ), dtype=np.float32)
        self.deque = deque(maxlen=self.n_step)
        self.ptr = 0
        self.size = 0
        self.full = False
        if self.linear_decay_step != 0:
            # Track when each time slot was added
            self.timestamps = np.empty(self.max_size, dtype=np.int64)
            self.current_time = 0

    def _as_transition(self, transition: Transition) -> Transition:
        return Transition(
            observations=np.asarray(transition.observations, dtype=self.obsveration_space.dtype).reshape(
                self.num_envs, *self.obsveration_shape),
            actions=np.asarray(transition.actions, dtype=self.action_space.dtype).reshape(
                self.num_envs, *self.action_shape),
            rewards=np.asarray(transition.rewards, dtype=np.float32).reshape(self.num_envs),
            truncations=np.asarray(transition.truncations, dtype=np.float32).reshape(self.num_envs),
            terminations=np.asarray(transition.terminations, dtype=np.float32).reshape(self.num_envs),
            next_observations=np.asarray(transition.next_observations, dtype=self.obsveration_space.dtype).reshape(
                self.num_envs, *self.obsveration_shape),
        )

    def _get_n_step_transition(self) -> tuple[Transition, np.ndarray]:
        prev_transition = self.deque[0]
        curr_transition = self.deque[-1]

        n_step_reward = np.array(curr_transition.rewards, dtype=np.float32)
        n_step_termination = np.array(curr_transition.terminations, dtype=np.float32)
        n_step_truncation = np.array(curr_transition.truncations, dtype=np.float32)
        n_step_next_observation = np.array(curr_transition.next_observations, copy=True)
        effective_n_steps = 1

        for idx in reversed(range(len(self.deque) - 1)):
            transition = self.deque[idx]
            episode_end = np.logical_or(
                transition.terminations.astype(bool),
                transition.truncations.astype(bool),
            )
            n_step_reward = transition.rewards + self.gamma * n_step_reward * (1.0 - episode_end.astype(np.float32))

            n_step_termination[episode_end] = transition.terminations[episode_end]
            n_step_truncation[episode_end] = transition.truncations[episode_end]
            n_step_next_observation[episode_end] = transition.next_observations[episode_end]
            effective_n_steps = np.where(episode_end, 1, effective_n_steps + 1)

        n_step_transition = Transition(
            observations=prev_transition.observations,
            actions=prev_transition.actions,
            rewards=n_step_reward,
            terminations=n_step_termination,
            truncations=n_step_truncation,
            next_observations=n_step_next_observation,
        )
        return n_step_transition, (np.float32(self.gamma) ** effective_n_steps).astype(np.float32)

    def _store(self, transition: Transition, discount: np.ndarray) -> None:
        add_count = len(transition.rewards)
        add_indices = (self.ptr + np.arange(add_count)) % self.max_size
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

    def add(self,
        transition: Transition):
        transition = self._as_transition(transition)
        self.deque.append(transition)

        if len(self.deque) >= self.n_step:
            n_step_transition, discount = self._get_n_step_transition()
            self._store(n_step_transition, discount)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.linear_decay_step == 0:
            return np.random.randint(0, self.size, size=batch_size)

        if self.use_approximate_sampling:
            return self._sample_indices_with_approximate_bias(batch_size)

        return self._sample_indices_with_bias(batch_size)

    def _sample_indices_with_bias(self, batch_size: int) -> np.ndarray:
        valid_timestamps = self.timestamps[:self.size]
        age = self.current_time - valid_timestamps
        if self.linear_decay_step > 0:
            weights = np.maximum(self.min_weight, 1.0 - age / self.abs_linear_decay_step)
        else:
            weights = np.minimum(1.0, self.min_weight + age / self.abs_linear_decay_step)

        if weights.sum() <= 0:
            weights = np.ones_like(weights, dtype=np.float32)

        probabilities = weights / weights.sum()
        return np.random.choice(self.size, size=batch_size, p=probabilities)

    def _sample_indices_with_approximate_bias(self, batch_size: int) -> np.ndarray:
        bucket_size = max((self.size + self.num_buckets - 1) // self.num_buckets, 1)

        logical_starts = np.arange(0, self.size, bucket_size)
        logical_ends = np.minimum(logical_starts + bucket_size, self.size)
        logical_midpoints = (logical_starts + logical_ends - 1) // 2
        bucket_midpoints = (self.ptr + logical_midpoints) % self.max_size if self.full else logical_midpoints
        bucket_timestamps = self.timestamps[bucket_midpoints]
        bucket_ages = self.current_time - bucket_timestamps

        if self.linear_decay_step > 0:
            bucket_weights = np.maximum(self.min_weight, 1.0 - bucket_ages / self.abs_linear_decay_step)
        else:
            bucket_weights = np.minimum(1.0, self.min_weight + bucket_ages / self.abs_linear_decay_step)

        if bucket_weights.sum() <= 0:
            bucket_weights = np.ones_like(bucket_weights, dtype=np.float32)

        bucket_probabilities = bucket_weights / bucket_weights.sum()
        sampled_buckets = np.random.choice(len(logical_starts), size=batch_size, p=bucket_probabilities)
        sampled_starts = logical_starts[sampled_buckets]
        sampled_ends = logical_ends[sampled_buckets]
        offsets = (np.random.random(size=batch_size) * (sampled_ends - sampled_starts)).astype(np.int64)
        logical_indices = sampled_starts + offsets
        return (self.ptr + logical_indices) % self.max_size if self.full else logical_indices

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
        np.savez(
            path,
            obsverations=self.obsverations[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            terminations=self.terminations[:self.size],
            truncations=self.trunactions[:self.size],
            next_obsverations=self.next_obsverations[:self.size],
            discounts=self.discounts[:self.size],
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for key, value in data.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return self.size
