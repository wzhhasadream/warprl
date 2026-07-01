from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium import spaces

from .__init__ import Batch, Transition
from .base_buffer import get_action_dim, get_obs_shape


def _reshape_obs(
    x: Any,
    n_envs: int,
    obs_shape: tuple[int, ...],
    dtype: jnp.dtype,
    device: jax.Device | None = None,
) -> jax.Array:
    return jnp.asarray(x, dtype=dtype, device=device).reshape((n_envs, *obs_shape))


def _reshape_action(
    x: Any,
    n_envs: int,
    action_shape: tuple[int, ...],
    dtype: jnp.dtype,
    device: jax.Device | None = None,
) -> jax.Array:
    return jnp.asarray(x, dtype=dtype, device=device).reshape((n_envs, *action_shape))


def _reshape_scalar(x: Any, n_envs: int, device: jax.Device | None = None) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32, device=device).reshape((n_envs,))


def resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    if device is None:
        return None

    if isinstance(device, str):
        if device == "cuda":
            device = "gpu"

        if ":" in device:
            device_type, index = device.split(":")
            if device_type == "cuda":
                device_type = "gpu"
            return jax.devices(device_type)[int(index)]

        return jax.devices(device)[0]

    return device


def _to_jax_dtype(dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        return jnp.float32
    return dtype


@struct.dataclass
class JaxBuffer:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    next_observations: jax.Array
    discounts: jax.Array
    timestamps: jax.Array
    ptr: jax.Array
    time_size: jax.Array
    size: jax.Array
    current_time: jax.Array
    window_observations: jax.Array
    window_actions: jax.Array
    window_rewards: jax.Array
    window_terminations: jax.Array
    window_truncations: jax.Array
    window_next_observations: jax.Array
    window_ptr: jax.Array
    window_size: jax.Array

    max_size: int = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    obs_shape: tuple[int, ...] = struct.field(pytree_node=False)
    action_shape: tuple[int, ...] = struct.field(pytree_node=False)
    obs_dtype: jnp.dtype = struct.field(pytree_node=False)
    action_dtype: jnp.dtype = struct.field(pytree_node=False)
    linear_decay_step: int = struct.field(pytree_node=False, default=0)
    abs_linear_decay_step: int = struct.field(pytree_node=False, default=0)
    min_weight: float = struct.field(pytree_node=False, default=0.1)
    n_step: int = struct.field(pytree_node=False, default=1)
    gamma: float = struct.field(pytree_node=False, default=0.99)
    use_approximate_sampling: bool = struct.field(pytree_node=False, default=True)
    num_buckets: int = struct.field(pytree_node=False, default=2000)
    device: jax.Device | None = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
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
        device: str | jax.Device | None = "cuda:0"
    ) -> "JaxBuffer":
        assert n_step >= 1, f"n_step must be positive, got {n_step}"
        assert num_envs >= 1, f"n_envs must be positive, got {num_envs}"
        assert max_size >= num_envs, f"max_size must be >= num_envs, got {max_size} and {num_envs}"
        assert num_buckets >= 1, f"num_buckets must be positive, got {num_buckets}"
        assert 0 <= min_weight <= 1, f"min_weight must be in [0, 1], got {min_weight}"
        use_approximate_sampling = use_approximate_sampling and max_size % num_envs == 0

        device = resolve_device(device)

        obs_shape = get_obs_shape(observation_space)
        if isinstance(obs_shape, dict):
            raise NotImplementedError("JaxBuffer does not support Dict observation spaces")
        obs_shape = tuple(obs_shape)
        action_shape = (get_action_dim(action_space),)
        obs_dtype = _to_jax_dtype(observation_space.dtype)
        action_dtype = _to_jax_dtype(action_space.dtype)
        observations = jnp.empty((max_size, *obs_shape), dtype=obs_dtype, device=device)
        actions = jnp.empty((max_size, *action_shape),
                            dtype=action_dtype, device=device)
        rewards = jnp.empty((max_size,), dtype=jnp.float32, device=device)
        terminations = jnp.empty((max_size,), dtype=jnp.float32, device=device)
        truncations = jnp.empty((max_size,), dtype=jnp.float32, device=device)
        next_observations = jnp.empty((max_size, *obs_shape), dtype=obs_dtype, device=device)
        discounts = jnp.empty((max_size,), dtype=jnp.float32, device=device)
        timestamps = jnp.zeros((max_size,), dtype=jnp.int32, device=device)

        window_observations = jnp.empty((n_step, num_envs, *obs_shape), dtype=obs_dtype, device=device)
        window_actions = jnp.empty((n_step, num_envs, *action_shape), dtype=action_dtype, device=device)
        window_rewards = jnp.empty((n_step, num_envs), dtype=jnp.float32, device=device)
        window_terminations = jnp.empty((n_step, num_envs), dtype=jnp.float32, device=device)
        window_truncations = jnp.empty((n_step, num_envs), dtype=jnp.float32, device=device)
        window_next_observations = jnp.empty((n_step, num_envs, *obs_shape), dtype=obs_dtype, device=device)

        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            next_observations=next_observations,
            discounts=discounts,
            timestamps=timestamps,
            ptr=jnp.array(0, dtype=jnp.int32, device=device),
            time_size=jnp.array(0, dtype=jnp.int32, device=device),
            size=jnp.array(0, dtype=jnp.int32, device=device),
            current_time=jnp.array(0, dtype=jnp.int32, device=device),
            window_observations=window_observations,
            window_actions=window_actions,
            window_rewards=window_rewards,
            window_terminations=window_terminations,
            window_truncations=window_truncations,
            window_next_observations=window_next_observations,
            window_ptr=jnp.array(0, dtype=jnp.int32, device=device),
            window_size=jnp.array(0, dtype=jnp.int32, device=device),
            max_size=int(max_size),
            num_envs=int(num_envs),
            obs_shape=obs_shape,
            action_shape=action_shape,
            obs_dtype=obs_dtype,
            action_dtype=action_dtype,
            linear_decay_step=int(linear_decay_step),
            abs_linear_decay_step=abs(int(linear_decay_step)),
            min_weight=float(min_weight),
            n_step=int(n_step),
            gamma=float(gamma),
            use_approximate_sampling=bool(use_approximate_sampling),
            num_buckets=int(num_buckets),
            device=device
        )
    @partial(jax.jit , donate_argnums=0)
    def add(self, transition: Transition) -> "JaxBuffer":
        observations = _reshape_obs(transition.observations, self.num_envs, self.obs_shape, self.obs_dtype, self.device)
        actions = _reshape_action(transition.actions, self.num_envs, self.action_shape, self.action_dtype, self.device)
        rewards = _reshape_scalar(transition.rewards, self.num_envs, self.device)
        terminations = _reshape_scalar(transition.terminations, self.num_envs, self.device)
        truncations = _reshape_scalar(transition.truncations, self.num_envs, self.device)
        next_observations = _reshape_obs(transition.next_observations, self.num_envs, self.obs_shape, self.obs_dtype, self.device)

        next_buffer = self.replace(
            window_observations=self.window_observations.at[self.window_ptr].set(observations),
            window_actions=self.window_actions.at[self.window_ptr].set(actions),
            window_rewards=self.window_rewards.at[self.window_ptr].set(rewards),
            window_terminations=self.window_terminations.at[self.window_ptr].set(terminations),
            window_truncations=self.window_truncations.at[self.window_ptr].set(truncations),
            window_next_observations=self.window_next_observations.at[self.window_ptr].set(next_observations),
            window_ptr=(self.window_ptr + 1) % self.n_step,
            window_size=jnp.minimum(self.window_size + 1, self.n_step),
        )
        return jax.lax.cond(
            next_buffer.window_size >= self.n_step,
            lambda buffer: buffer._store_from_window(),
            lambda buffer: buffer,
            next_buffer,
        )

    def can_sample(self) -> jax.Array:
        return self.size > 0

    def _ordered_window_indices(self) -> jax.Array:
        return (self.window_ptr + jnp.arange(self.n_step, dtype=self.window_ptr.dtype)) % self.n_step

    def _get_n_step_transition(self) -> tuple[Transition, jax.Array]:
        indices = self._ordered_window_indices()
        observations = self.window_observations[indices]
        actions = self.window_actions[indices]
        rewards = self.window_rewards[indices]
        terminations = self.window_terminations[indices]
        truncations = self.window_truncations[indices]
        next_observations = self.window_next_observations[indices]

        n_step_reward = rewards[-1]
        n_step_termination = terminations[-1]
        n_step_truncation = truncations[-1]
        n_step_next_observation = next_observations[-1]
        effective_n_steps = jnp.ones((self.num_envs,), dtype=jnp.float32, device=self.device)

        for idx in reversed(range(self.n_step - 1)):
            episode_end = jnp.logical_or(terminations[idx] > 0.0, truncations[idx] > 0.0)
            n_step_reward = rewards[idx] + self.gamma * n_step_reward * (1.0 - episode_end.astype(jnp.float32))
            n_step_termination = jnp.where(episode_end, terminations[idx], n_step_termination)
            n_step_truncation = jnp.where(episode_end, truncations[idx], n_step_truncation)
            obs_mask = episode_end.reshape(
                (self.num_envs,) + (1,) * len(self.obs_shape))
            n_step_next_observation = jnp.where(
                obs_mask,
                next_observations[idx],
                n_step_next_observation,
            )
            effective_n_steps = jnp.where(episode_end, 1.0, effective_n_steps + 1.0)

        transition = Transition(
            observations=observations[0],
            actions=actions[0],
            rewards=n_step_reward,
            terminations=n_step_termination,
            truncations=n_step_truncation,
            next_observations=n_step_next_observation,
        )
        discounts = jnp.power(jnp.asarray(self.gamma, dtype=jnp.float32, device=self.device), effective_n_steps)
        return transition, discounts

    def _store_from_window(self) -> "JaxBuffer":
        transition, discounts = self._get_n_step_transition()
        add_indices = (self.ptr + jnp.arange(self.num_envs, dtype=self.ptr.dtype, device=self.device)) % self.max_size

        new_observations = self.observations.at[add_indices].set(transition.observations)
        new_actions = self.actions.at[add_indices].set(transition.actions)
        new_rewards = self.rewards.at[add_indices].set(transition.rewards)
        new_terminations = self.terminations.at[add_indices].set(transition.terminations)
        new_truncations = self.truncations.at[add_indices].set(transition.truncations)
        new_next_observations = self.next_observations.at[add_indices].set(transition.next_observations)
        new_discounts = self.discounts.at[add_indices].set(discounts)
        new_timestamps = self.timestamps.at[add_indices].set(self.current_time)

        new_ptr = (self.ptr + self.num_envs) % self.max_size
        new_size = jnp.minimum(self.size + self.num_envs, self.max_size)

        return self.replace(
            observations=new_observations,
            actions=new_actions,
            rewards=new_rewards,
            terminations=new_terminations,
            truncations=new_truncations,
            next_observations=new_next_observations,
            discounts=new_discounts,
            timestamps=new_timestamps,
            ptr=new_ptr,
            time_size=new_size,
            size=new_size,
            current_time=self.current_time + 1,
            window_observations=self.window_observations,
            window_actions=self.window_actions,
            window_rewards=self.window_rewards,
            window_terminations=self.window_terminations,
            window_truncations=self.window_truncations,
            window_next_observations=self.window_next_observations,
            window_ptr=self.window_ptr,
            window_size=self.window_size,
        )

    def _valid_mask(self) -> jax.Array:
        idx = jnp.arange(self.max_size, dtype=self.size.dtype)
        return idx < self.size

    def _linear_weights(self) -> jax.Array:
        age = (self.current_time - self.timestamps).astype(jnp.float32)
        decay = jnp.asarray(max(self.abs_linear_decay_step, 1), dtype=jnp.float32)
        min_weight = jnp.asarray(self.min_weight, dtype=jnp.float32)
        if self.linear_decay_step > 0:
            return jnp.maximum(min_weight, 1.0 - age / decay)
        return jnp.minimum(1.0, min_weight + age / decay)

    def _weighted_probabilities(self) -> jax.Array:
        valid = self._valid_mask()
        weights = jnp.where(valid, self._linear_weights(), 0.0)
        weight_sum = jnp.sum(weights)
        valid_count = jnp.maximum(jnp.sum(valid), 1)
        uniform = valid.astype(jnp.float32) / valid_count.astype(jnp.float32)
        weighted = weights / jnp.maximum(weight_sum, jnp.finfo(jnp.float32).tiny)
        return jnp.where(weight_sum > 0, weighted, uniform)

    def _sample_uniform_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        return jax.random.randint(key, (batch_size,), 0, jnp.maximum(self.size, 1))

    def _sample_exact_biased_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        return jax.random.choice(
            key,
            self.max_size,
            shape=(batch_size,),
            p=self._weighted_probabilities(),
        )

    def _sample_approx_biased_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        num_buckets = min(self.num_buckets, self.max_size)
        bucket_size = jnp.maximum((self.size + num_buckets - 1) // num_buckets, 1)
        bucket_ids = jnp.arange(num_buckets, dtype=jnp.int32, device=self.device)
        logical_starts = bucket_ids * bucket_size
        logical_ends = jnp.minimum(logical_starts + bucket_size, self.size)
        non_empty = logical_starts < self.size
        logical_midpoints = (logical_starts + logical_ends - 1) // 2
        is_full = self.size >= self.max_size
        bucket_midpoints = jnp.where(
            is_full,
            (self.ptr + logical_midpoints) % self.max_size,
            logical_midpoints,
        )

        weights = self._linear_weights()[bucket_midpoints]
        weights = jnp.where(non_empty, weights, 0.0)
        weight_sum = jnp.sum(weights)
        fallback = non_empty.astype(jnp.float32) / jnp.maximum(jnp.sum(non_empty), 1)
        probabilities = jnp.where(weight_sum > 0, weights / jnp.maximum(weight_sum, jnp.finfo(jnp.float32).tiny), fallback)

        key_bucket, key_offset = jax.random.split(key)
        sampled_buckets = jax.random.choice(key_bucket, num_buckets, shape=(batch_size,), p=probabilities)
        sampled_starts = logical_starts[sampled_buckets]
        sampled_ends = logical_ends[sampled_buckets]
        widths = jnp.maximum(sampled_ends - sampled_starts, 1)
        logical_indices = sampled_starts + (
            jax.random.uniform(key_offset, (batch_size,)) * widths.astype(jnp.float32)
        ).astype(jnp.int32)
        return jnp.where(is_full, (self.ptr + logical_indices) % self.max_size, logical_indices)

    def _sample_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        if self.linear_decay_step == 0:
            return self._sample_uniform_indices(key, batch_size)
        if self.use_approximate_sampling:
            return self._sample_approx_biased_indices(key, batch_size)
        return self._sample_exact_biased_indices(key, batch_size)
        
    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(self, key: jax.Array, batch_size: int) -> Batch:
        indices = self._sample_indices(key, batch_size)

        return Batch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices, None],
            dones=self.terminations[indices, None],
            next_observations=self.next_observations[indices],
            discounts=self.discounts[indices, None],
        )

    def reset(self) -> "JaxBuffer":
        return self.replace(
            ptr=jnp.array(0, dtype=self.ptr.dtype, device=self.device),
            time_size=jnp.array(0, dtype=self.time_size.dtype, device=self.device),
            size=jnp.array(0, dtype=self.size.dtype, device=self.device),
            current_time=jnp.array(0, dtype=self.current_time.dtype, device=self.device),
            window_ptr=jnp.array(0, dtype=self.window_ptr.dtype, device=self.device),
            window_size=jnp.array(0, dtype=self.window_size.dtype, device=self.device),
        )

    def save(self, path: str) -> None:
        size = int(self.size)
        dataset = {
            "observations": np.asarray(self.observations[:size]),
            "actions": np.asarray(self.actions[:size]),
            "rewards": np.asarray(self.rewards[:size]),
            "terminations": np.asarray(self.terminations[:size]),
            "truncations": np.asarray(self.truncations[:size]),
            "next_observations": np.asarray(self.next_observations[:size]),
            "discounts": np.asarray(self.discounts[:size]),
            "timestamps": np.asarray(self.timestamps[:size]),
            "ptr": np.asarray(self.ptr),
            "time_size": np.asarray(self.time_size),
            "size": np.asarray(self.size),
            "current_time": np.asarray(self.current_time),
        }
        np.savez(path, **dataset)


    def load(self, path: str) -> "JaxBuffer":
        data = np.load(path)
        loaded = {key: jnp.asarray(data[key], device=self.device) for key in data.files}
        return self.replace(**loaded)

    def __len__(self) -> int:
        return int(self.size)
