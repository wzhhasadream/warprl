"""Memory-efficient replay buffer for frame-first pixel observations."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view

from .base_buffer import BaseBuffer
from .types import Batch, Transition


class NumpyLazyFrameBuffer(BaseBuffer):
    """Store individual frames and reconstruct stacks only while sampling."""

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
    ) -> None:
        super().__init__(observation_space, action_space, max_size)
        if (
            not isinstance(observation_space, spaces.Box)
            or len(observation_space.shape) != 4
        ):
            raise TypeError(
                "NumpyLazyFrameBuffer requires a 4-D Box observation space "
                "of shape [F, H, W, C]"
            )
        if num_envs != 1:
            raise ValueError("NumpyLazyFrameBuffer supports num_envs=1 only")
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if num_buckets < 1 or not 0 <= min_weight <= 1:
            raise ValueError("num_buckets must be positive and min_weight must be in [0, 1]")

        self.max_size = int(max_size)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.linear_decay_step = int(linear_decay_step)
        self.abs_linear_decay_step = abs(self.linear_decay_step)
        self.min_weight = float(min_weight)
        self.num_envs = 1
        self.use_approximate_sampling = bool(use_approximate_sampling)
        self.num_buckets = int(num_buckets)

        obs_shape = tuple(observation_space.shape)
        self._observation_shape = obs_shape
        self.frame_stack = obs_shape[0]
        if max_size < max(self.frame_stack, n_step):
            raise ValueError("max_size is too small for frame_stack/n_step")
        frame_shape = obs_shape[1:]
        obs_dtype = np.float32 if observation_space.dtype == np.float64 else observation_space.dtype
        action_dtype = np.float32 if action_space.dtype == np.float64 else action_space.dtype

        # The tail mirrors the first frames so all sampled windows stay contiguous.
        length = self.max_size + self.n_step + self.frame_stack - 1
        self._frames = np.empty((length, *frame_shape), dtype=obs_dtype)
        self._actions = np.empty((length, *self.action_shape), dtype=action_dtype)
        self._rewards = np.empty(length, dtype=np.float32)
        self._terminations = np.empty(length, dtype=np.float32)
        self._truncations = np.empty(length, dtype=np.float32)
        self._valid = np.zeros(length, dtype=bool)
        self._timestamps = np.empty(length, dtype=np.int64) if self.linear_decay_step else None
        window = self.frame_stack + self.n_step
        self._frame_windows = sliding_window_view(self._frames, (window, *frame_shape))
        self._frame_windows = self._frame_windows[(slice(None),) + (0,) * len(frame_shape)]
        self._reward_windows = sliding_window_view(self._rewards, self.n_step)
        self._termination_windows = sliding_window_view(self._terminations, self.n_step)
        self._truncation_windows = sliding_window_view(self._truncations, self.n_step)

        self.reset()

    def _transition(self, transition: Transition) -> tuple[np.ndarray, ...]:
        full_shape = (1, *self._observation_shape)
        observation = np.asarray(transition.observations, dtype=self._frames.dtype).reshape(full_shape)[0]
        next_observation = np.asarray(transition.next_observations, dtype=self._frames.dtype).reshape(full_shape)[0]
        action = np.asarray(transition.actions, dtype=self._actions.dtype).reshape((1, *self.action_shape))[0]
        reward = np.asarray(transition.rewards, dtype=np.float32).reshape(1)[0]
        termination = np.asarray(transition.terminations, dtype=np.float32).reshape(1)[0]
        truncation = np.asarray(transition.truncations, dtype=np.float32).reshape(1)[0]
        return observation, next_observation, action, reward, termination, truncation

    def _index(self, index: int) -> int:
        index %= self.max_size
        return index + self.max_size if index < self.frame_stack else index

    def _mark(self, index: int, valid: bool) -> None:
        index = self._index(index)
        if valid:
            self._num_valid += int(not self._valid[index])
            self._valid[index] = True
            if self._timestamps is not None:
                self._timestamps[index] = self._current_time
        else:
            self._num_valid -= int(self._valid[index])
            self._valid[index] = False

    def _add_frame(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        reward: np.float32,
        termination: np.float32,
        truncation: np.float32,
    ) -> None:
        index = self._buffer_idx
        targets = (index, self.max_size + index) if index < self.n_step + self.frame_stack - 1 else (index,)
        for target in targets:
            self._frames[target] = frame
            self._actions[target] = action
            self._rewards[target] = reward
            self._terminations[target] = termination
            self._truncations[target] = truncation
            if self._timestamps is not None:
                self._timestamps[target] = self._current_time

        if self._trajectory_length >= self.n_step:
            self._mark(index - self.n_step + 1, True)
        self._mark(index + self.frame_stack, False)
        self._buffer_idx = (index + 1) % self.max_size
        self.ptr = self._buffer_idx
        self._writes += 1
        self.full = self._writes >= self.max_size
        self.size = self._num_valid

    def add(self, transition: Transition) -> None:
        observations, next_observations, action, reward, termination, truncation = self._transition(transition)
        if self._trajectory_length == 0:
            for frame in observations:
                self._add_frame(frame, action, reward, termination, truncation)
            self._trajectory_length = 1
        self._add_frame(next_observations[-1], action, reward, termination, truncation)
        self._trajectory_length = 0 if bool(termination) or bool(truncation) else self._trajectory_length + 1
        self._current_time += 1

    def can_sample(self) -> bool:
        return self._num_valid > 0

    def _weights(self, timestamps: np.ndarray) -> np.ndarray:
        age = (self._current_time - timestamps).astype(np.float32)
        if self.linear_decay_step > 0:
            return np.maximum(self.min_weight, 1.0 - age / self.abs_linear_decay_step)
        return np.minimum(1.0, self.min_weight + age / self.abs_linear_decay_step)

    def _sampling_weights(self, timestamps: np.ndarray) -> np.ndarray:
        weights = self._weights(timestamps)
        if weights.sum() <= 0:
            return np.ones_like(weights, dtype=np.float32)
        return weights

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        valid = np.flatnonzero(self._valid)
        if self.linear_decay_step == 0:
            return np.random.choice(valid, batch_size, replace=True)
        if not self.use_approximate_sampling:
            weights = self._sampling_weights(self._timestamps[valid])
            return np.random.choice(valid, batch_size, p=weights / weights.sum())

        # Valid slots follow ring order, not chronological order; bucket by age.
        valid = valid[np.argsort(self._timestamps[valid], kind="stable")]
        bucket_size = max((valid.size + self.num_buckets - 1) // self.num_buckets, 1)
        starts = np.arange(0, valid.size, bucket_size)
        ends = np.minimum(starts + bucket_size, valid.size)
        midpoints = (starts + ends - 1) // 2
        weights = self._sampling_weights(self._timestamps[valid[midpoints]])
        buckets = np.random.choice(len(starts), batch_size, p=weights / weights.sum())
        offsets = (np.random.random(batch_size) * (ends[buckets] - starts[buckets])).astype(np.int64)
        return valid[starts[buckets] + offsets]

    def sample(self, batch_size: int) -> Batch:
        if not self.can_sample():
            raise AssertionError("Cannot sample from an empty buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        indices = self._sample_indices(batch_size)
        frames = self._frame_windows[indices - self.frame_stack]
        rewards = self._reward_windows[indices]
        terminations = self._termination_windows[indices]
        truncations = self._truncation_windows[indices]
        done = (terminations > 0) | (truncations > 0)
        has_done = done.any(axis=1)
        steps = np.where(has_done, np.argmax(done, axis=1) + 1, self.n_step)
        mask = np.arange(self.n_step)[None] < steps[:, None]
        discounts = np.power(np.float32(self.gamma), steps).astype(np.float32)
        rewards = np.sum(
            rewards * np.power(np.float32(self.gamma), np.arange(self.n_step)) * mask,
            axis=1,
            dtype=np.float32,
        )
        next_frames = frames[
            np.arange(batch_size)[:, None],
            steps[:, None] + np.arange(self.frame_stack),
        ]
        endpoints = indices + steps - 1
        return Batch(
            observations=frames[:, : self.frame_stack],
            actions=np.array(self._actions[indices]),
            rewards=rewards[:, None],
            dones=np.array(self._terminations[endpoints, None]),
            next_observations=next_frames,
            discounts=discounts[:, None],
        )

    def reset(self) -> None:
        self._buffer_idx = 0
        self._trajectory_length = 0
        self._writes = 0
        self._num_valid = 0
        self._current_time = 0
        self.ptr = 0
        self.size = 0
        self.full = False
        self._valid.fill(False)

    def save(self, path: str) -> None:
        np.savez(
            path,
            frames=self._frames,
            actions=self._actions,
            rewards=self._rewards,
            terminations=self._terminations,
            truncations=self._truncations,
            valid=self._valid,
            timestamps=self._timestamps if self._timestamps is not None else np.empty(0, np.int64),
            state=np.array([self._buffer_idx, self._trajectory_length, self._writes, self._num_valid, self._current_time]),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for name in ("frames", "actions", "rewards", "terminations", "truncations", "valid"):
            getattr(self, f"_{name}")[...] = data[name]
        if self._timestamps is not None and data["timestamps"].size:
            self._timestamps[...] = data["timestamps"]
        self._buffer_idx, self._trajectory_length, self._writes, self._num_valid, self._current_time = map(int, data["state"])
        self.ptr, self.size, self.full = self._buffer_idx, self._num_valid, self._writes >= self.max_size

    def __len__(self) -> int:
        return self._num_valid


NumpyPixelBuffer = NumpyLazyFrameBuffer
