from typing import NamedTuple, Any
import numpy as np
import jax.numpy as jnp
import jax
from gymnasium import spaces
import gymnasium as gym
from flax import struct

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


ObsTree = jax.Array

class Batch(NamedTuple):
    observations: ObsTree
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    next_observations: ObsTree
    discounts: jax.Array




def create_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_observations: np.ndarray,
    discounts: np.ndarray | None = None,
) -> Batch:
    """
    Create a batch dictionary for JAX.


    Args:
        observations: shape (batch_size, obs_dim)
        actions: shape (batch_size, action_dim)
        rewards: shape (batch_size, 1)
        dones: shape (batch_size, 1)
        next_observations: shape (batch_size, obs_dim)
        discounts: shape (batch_size, 1), discount multiplier for bootstrapping

    Returns:
        Batch dictionary with JAX arrays
    """
    if discounts is None:
        discounts = np.ones_like(rewards, dtype=np.float32)

    return Batch(
        jnp.array(observations), 
        jnp.array(actions), 
        jnp.array(rewards.reshape(-1, 1)), 
        jnp.array(dones.reshape(-1, 1)), 
        jnp.array(next_observations),
        jnp.array(discounts.reshape(-1, 1))
        )




class ReplayBuffer:
    """
    Replay buffer for online RL with multi-env support and optional linear bias sampling.

    Args:
        obs_shape_space: Observation space from environment
        action_shape_space: Action space from environment
        max_time_size: Maximum number of time slots
        n_envs: Number of parallel environments
        linear_decay_steps: Controls sampling bias direction:
            - 0: uniform sampling (no bias)
            - >0: newer-biased (prefer recent experiences)
            - <0: older-biased (prefer older experiences)
        min_weight: Minimum weight for biased experiences (0.1 = 10% of maximum weight)
    """

    def __init__(
        self,
        obs_shape_space: spaces.Space,
        action_shape_space: spaces.Space,
        max_size: int = int(1e6),
        n_envs: int = 1,
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = True,
        optimize_memory_usage: bool = False
    ):

        self.max_time_size = max(int(max_size) // int(n_envs), 1)
        self.n_envs = n_envs
        self.time_size = 0
        self.size = 0
        self.ptr = 0
        self.full = False
        self.optimize_memory_usage = optimize_memory_usage

        # Linear bias parameters
        self._raw_linear_decay_steps = linear_decay_steps  # Keep original sign
        self.linear_decay_steps = abs(linear_decay_steps)  # Use absolute value for calculations
        if use_approximate_sampling:
            self.num_buckets = num_buckets
        self.use_approximate_sampling = use_approximate_sampling
        self.min_weight = min_weight

        # Validate parameters
        assert 0 <= min_weight <= 1, f"min_weight must be in [0, 1], got {min_weight}"

        # Extract shapes from spaces
        obs_shape = get_obs_shape(obs_shape_space)
        action_dim = get_action_dim(action_shape_space)

        # Handle both int and tuple for obs_shape
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = obs_shape

        self.action_shape = (action_dim,)

        # Initialize buffers with proper shapes
        self.observations = np.zeros((self.max_time_size, self.n_envs, *self.obs_shape), dtype=obs_shape_space.dtype)
        self.actions = np.zeros((self.max_time_size, self.n_envs, *self.action_shape), dtype=action_shape_space.dtype)
        self.rewards = np.zeros((self.max_time_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.max_time_size, self.n_envs), dtype=np.float32)
        if linear_decay_steps != 0:
            self.timestamps = np.zeros(self.max_time_size, dtype=np.int64)  # Track when each time slot was added
            self.current_time = 0
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.max_time_size, self.n_envs, *self.obs_shape), dtype=obs_shape_space.dtype)    
        # Initialize dictionary for truncated observations
        # Key: buffer_index (int), Value: {env_index (int): observation (np.ndarray)}
        if optimize_memory_usage:
            self.truncated_next_obs = {}

    @classmethod
    def from_env(
        cls,
        env: gym.vector.VectorEnv,  
        max_size: int = int(1e6),
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = True,
        optimize_memory_usage: bool = False,
    ) -> 'ReplayBuffer':
        """Create ReplayBuffer from environment - convenience method."""
        obs_shape_space = env.single_observation_space
        action_shape_space = env.single_action_space
        n_envs = getattr(env, 'num_envs', 1)
        return cls(
            obs_shape_space,
            action_shape_space,
            n_envs=n_envs,
            linear_decay_steps=linear_decay_steps,
            min_weight=min_weight,
            num_buckets=num_buckets,
            use_approximate_sampling=use_approximate_sampling,
            optimize_memory_usage=optimize_memory_usage,
            max_size=max_size,
        )

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float | np.ndarray,
            next_obs: np.ndarray, done: bool | np.ndarray, truncations: None | np.ndarray = None):
        """Add transition(s) to the buffer. Supports both single and multi-env."""
        if self.optimize_memory_usage:
            assert truncations is not None, "truncations must be provided when optimize_memory_usage is True"

        self.observations[self.ptr] = obs.reshape(self.n_envs, *self.obs_shape)
        self.actions[self.ptr] = action.reshape(self.n_envs, *self.action_shape)
        if not self.optimize_memory_usage:
            self.next_observations[self.ptr] = next_obs.reshape(self.n_envs, *self.obs_shape)
        else:
            self.observations[(self.ptr + 1) % self.max_time_size] = next_obs.reshape(self.n_envs, *self.obs_shape)
            if self.ptr in self.truncated_next_obs:
                del self.truncated_next_obs[self.ptr]
            trunc_indices = np.where(np.atleast_1d(truncations))[0]
            if len(trunc_indices) > 0:
                reshaped_next_obs = next_obs.reshape(self.n_envs, *self.obs_shape)
                self.truncated_next_obs[self.ptr] = {}
                for env_idx in trunc_indices:
                    self.truncated_next_obs[self.ptr][env_idx] = reshaped_next_obs[env_idx].copy()
                    
        self.rewards[self.ptr] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self.dones[self.ptr] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        if self.linear_decay_steps != 0:
            self.timestamps[self.ptr] = self.current_time
            self.current_time += 1
            
        self.ptr += 1
        self.time_size = min(self.time_size + 1, self.max_time_size)
        self.size = self.time_size * self.n_envs

        if self.ptr == self.max_time_size:
            self.full = True
            self.ptr = 0


    def _valid_start_indices(self, n_step: int) -> np.ndarray:
        """Return physical time indices that have a valid n-step suffix."""
        if self.time_size <= 0:
            return np.zeros((1,), dtype=np.int64)

        n_step = max(1, min(int(n_step), self.time_size, self.max_time_size))
        valid_count = max(self.time_size - n_step + 1, 1)

        if self.full:
            return (self.ptr + np.arange(valid_count, dtype=np.int64)) % self.max_time_size
        return np.arange(valid_count, dtype=np.int64)

    def _sample_time_indices(self, batch_size: int, n_step: int) -> np.ndarray:
        """Sample valid starting indices while preserving the configured bias."""
        valid_indices = self._valid_start_indices(n_step)

        if self._raw_linear_decay_steps == 0:
            return np.random.choice(valid_indices, size=batch_size)

        if self.use_approximate_sampling and int(n_step) == 1:
            return self._sample_with_approximate_bias(batch_size)

        valid_timestamps = self.timestamps[valid_indices]
        age = self.current_time - valid_timestamps
        if self._raw_linear_decay_steps > 0:
            weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_steps)
        else:
            weights = np.minimum(1.0, self.min_weight + age / self.linear_decay_steps)

        probabilities = weights / weights.sum()
        return np.random.choice(valid_indices, size=batch_size, p=probabilities)

    def _next_obs_at(self, batch_index: np.ndarray, env_index: np.ndarray) -> np.ndarray:
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_index + 1) % self.max_time_size, env_index].copy()
            for i in range(len(batch_index)):
                idx = batch_index[i]
                env_idx = env_index[i]
                if idx in self.truncated_next_obs and env_idx in self.truncated_next_obs[idx]:
                    next_obs[i] = self.truncated_next_obs[idx][env_idx]
            return next_obs
        return self.next_observations[batch_index, env_index]

    def _gather_n_step(
        self,
        batch_index: np.ndarray,
        env_index: np.ndarray,
        n_step: int,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build n-step rewards, final dones, final next observations, and discounts."""
        if self.optimize_memory_usage and int(n_step) > 1:
            raise NotImplementedError("n-step sampling is not supported with optimize_memory_usage=True")

        n_step = max(1, min(int(n_step), self.time_size, self.max_time_size))
        offsets = np.arange(n_step, dtype=np.int64)
        sequence_indices = (batch_index[:, None] + offsets[None, :]) % self.max_time_size

        rewards = self.rewards[sequence_indices, env_index[:, None]]
        dones = self.dones[sequence_indices, env_index[:, None]]

        shifted_dones = np.concatenate(
            [np.zeros_like(dones[:, :1]), dones[:, :-1]],
            axis=1,
        )
        reward_masks = np.cumprod(1.0 - shifted_dones, axis=1)
        discount_powers = (np.float32(gamma) ** offsets).astype(np.float32)
        n_step_rewards = np.sum(rewards * reward_masks * discount_powers[None, :], axis=1)

        has_done = dones > 0
        first_done = np.argmax(has_done, axis=1)
        final_offsets = np.where(np.any(has_done, axis=1), first_done, n_step - 1)
        final_indices = sequence_indices[np.arange(len(batch_index)), final_offsets]

        final_dones = self.dones[final_indices, env_index]
        final_next_obs = self._next_obs_at(final_indices, env_index)
        effective_n_steps = np.sum(reward_masks, axis=1)
        discounts = (np.float32(gamma) ** effective_n_steps).astype(np.float32)

        return n_step_rewards, final_dones, final_next_obs, discounts

    def sample(self, batch_size: int, n_step: int = 1, gamma: float = 0.99) -> Batch:
        """
        Args:
            batch_size: Number of samples to draw

        Returns:
            Batch, each field shape (batch_size, ...) with:
                'observations': shape: (batch_size, obs_dim)
                'actions': shape: (batch_size, action_dim)
                'rewards': shape: (batch_size, 1)
                'dones': shape: (batch_size, 1)
                'next_observations': shape: (batch_size, obs_dim)
        """

        batch_index = self._sample_time_indices(batch_size, n_step)
        env_index = np.random.randint(0, self.n_envs, size=batch_size)
        rewards, dones, next_obs, discounts = self._gather_n_step(
            batch_index, env_index, n_step, gamma)

        return create_batch(
            observations=self.observations[batch_index, env_index], 
            actions=self.actions[batch_index, env_index],
            rewards=rewards,
            dones=dones,
            next_observations=next_obs,
            discounts=discounts,
        )

    def _sample_with_bias(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with linear bias weighting.
        Returns batch_index.
        """
        valid_timestamps = self.timestamps[:self.time_size] 
        age = self.current_time - valid_timestamps

        if self._raw_linear_decay_steps > 0:
            weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_steps)
        else:
            weights = np.minimum(1.0, self.min_weight + age / self.linear_decay_steps)

        # If replaybuffer is full, set the weight of the oldest sample to 0
        if self.optimize_memory_usage:
            if self.full:
                weights[self.ptr] = 0

        probabilities = weights / weights.sum()

        batch_index = np.random.choice(self.time_size, size=batch_size, p=probabilities)
    
        return batch_index

    def _sample_with_approximate_bias(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with approximate linear bias weighting using bucketing.
        Returns batch_index.
        """
        # 1. Determine bucket size
        bucket_size = self.time_size // self.num_buckets
        if bucket_size == 0:
            return self._sample_with_bias(batch_size)

        # 2. Calculate approximate weight for each bucket
        mid_point_indices = np.arange(bucket_size // 2, self.time_size, bucket_size)
        bucket_timestamps = self.timestamps[mid_point_indices]
        bucket_ages = self.current_time - bucket_timestamps

        if self._raw_linear_decay_steps > 0:
            bucket_weights = np.maximum(self.min_weight, 1.0 - bucket_ages / self.linear_decay_steps)
        else:
            bucket_weights = np.minimum(1.0, self.min_weight + bucket_ages / self.linear_decay_steps)

        if bucket_weights.sum() == 0:
            bucket_weights = np.ones_like(bucket_weights)

        bucket_probabilities = bucket_weights / bucket_weights.sum()

        # 3. Sample bucket indices
        sampled_bucket_indices = np.random.choice(
            len(bucket_probabilities),
            size=batch_size,
            p=bucket_probabilities
        )

        # 4. Sample uniformly within each chosen bucket
        bucket_starts = sampled_bucket_indices * bucket_size
        random_offsets = np.random.randint(0, bucket_size, size=batch_size)
        batch_index = bucket_starts + random_offsets

        # Ensure indices are within range
        batch_index = np.minimum(batch_index, self.time_size - 1)
        if self.optimize_memory_usage:
            if self.full:
                mask = (batch_index == self.ptr)
                if np.any(mask):
                    batch_index[mask] = (batch_index[mask] + 1) % self.max_time_size

        return batch_index

    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.time_size = 0
        self.size = 0
        self.current_time = 0
        if self.linear_decay_steps != 0:
            self.timestamps.fill(0)


    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        decay_info = f", decay_steps={self.linear_decay_steps}, min_weight={self.min_weight}" if self.linear_decay_steps > 0 else ""
        return f"ReplayBuffer(size={self.size}/{self.max_time_size * self.n_envs}, time_size={self.time_size}/{self.max_time_size}, obs_shape={self.obs_shape}, n_envs={self.n_envs}{decay_info})"




#######################################################
############ JAX ReplayBuffer #########################
#######################################################


def _reshape_obs_leaf(x: Any, shape_spec: tuple[int, ...], n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs, *shape_spec))


def _reshape_action(x: Any, shape_spec: tuple[int, ...], n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs, *shape_spec))


def _reshape_scalar_vec(x: Any, n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs,))


def _gather_time_env(x: jax.Array, time_idx: jax.Array, env_idx: jax.Array) -> jax.Array:
    return x[time_idx, env_idx]



@struct.dataclass
class GPUReplayBuffer:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    next_observations: jax.Array
    ptr: jax.Array
    time_size: jax.Array
    size: jax.Array
    timestamps: jax.Array
    current_time: jax.Array

    max_time_size: int = struct.field(pytree_node=False)
    n_envs: int = struct.field(pytree_node=False)
    obs_shape: tuple[int, ...] = struct.field(pytree_node=False)
    action_shape: tuple[int, ...] = struct.field(pytree_node=False)
    obs_dtype: jnp.dtype = struct.field(pytree_node=False)
    action_dtype: jnp.dtype = struct.field(pytree_node=False)
    raw_linear_decay_steps: int = struct.field(pytree_node=False, default=0)
    linear_decay_steps: int = struct.field(pytree_node=False, default=0)
    min_weight: float = struct.field(pytree_node=False, default=0.1)

    @classmethod
    def create(
        cls,
        obs_shape_space: spaces.Space,
        action_shape_space: spaces.Space,
        max_size: int = int(1e6),
        n_envs: int = 1,
        *,
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = False,
        optimize_memory_usage: bool = False,
    ) -> 'GPUReplayBuffer':
        """Initialize a JIT-friendly replay buffer.

        Notes:
        - If `linear_decay_steps != 0`, sampling uses an exact time-biased distribution
            (matching the CPU ReplayBuffer implementation in this repo):
            * `> 0`: newer-biased (prefer recent experiences)
            * `< 0`: older-biased (prefer old experiences)
        """
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
        if optimize_memory_usage:
            raise NotImplementedError("GPUReplayBuffer does not support optimize_memory_usage")
        if use_approximate_sampling and linear_decay_steps != 0:
            raise NotImplementedError("GPUReplayBuffer does not support approximate biased sampling")

        max_time_size = max(int(max_size) // int(n_envs), 1)
        max_time_size = int(max_time_size)

        obs_shape = get_obs_shape(obs_shape_space)
        if isinstance(obs_shape, dict):
            raise NotImplementedError("GPUReplayBuffer does not support Dict observation spaces")
        observation_shape = tuple(obs_shape)
        action_shape = (get_action_dim(action_shape_space),)
        obs_dtype = jnp.dtype(obs_shape_space.dtype)
        action_dtype = jnp.dtype(action_shape_space.dtype)

        def zeros_obs(shape_spec: tuple[int, ...]) -> jax.Array:
            return jnp.zeros((max_time_size, n_envs, *shape_spec), dtype=obs_dtype)

        observations = zeros_obs(observation_shape)
        next_observations = zeros_obs(observation_shape)

        actions = jnp.zeros(
            (max_time_size, n_envs, *action_shape), dtype=action_dtype)
        rewards = jnp.zeros((max_time_size, n_envs), dtype=jnp.float32)
        dones = jnp.zeros((max_time_size, n_envs), dtype=jnp.float32)

        ptr = jnp.array(0, dtype=jnp.int32)
        time_size = jnp.array(0, dtype=jnp.int32)
        size = jnp.array(0, dtype=jnp.int32)
        timestamps = jnp.zeros((max_time_size,), dtype=jnp.int32)
        current_time = jnp.array(0, dtype=jnp.int32)

        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_observations=next_observations,
            ptr=ptr,
            time_size=time_size,
            size=size,
            timestamps=timestamps,
            current_time=current_time,
            max_time_size=max_time_size,
            n_envs=n_envs,
            obs_shape=observation_shape,
            action_shape=action_shape,
            obs_dtype=obs_dtype,
            action_dtype=action_dtype,
            raw_linear_decay_steps=int(linear_decay_steps),
            linear_decay_steps=abs(int(linear_decay_steps)),
            min_weight=float(min_weight),
        )

    @classmethod
    def from_env(
        cls,
        env: gym.vector.VectorEnv,
        max_size: int = int(1e6),
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = False,
        optimize_memory_usage: bool = False,
    ) -> 'GPUReplayBuffer':
        """Create GPUReplayBuffer from an environment."""
        return cls.create(
            env.single_observation_space,
            env.single_action_space,
            max_size=max_size,
            n_envs=getattr(env, 'num_envs', 1),
            linear_decay_steps=linear_decay_steps,
            min_weight=min_weight,
            num_buckets=num_buckets,
            use_approximate_sampling=use_approximate_sampling,
            optimize_memory_usage=optimize_memory_usage,
        )


    def add(
            self,
            obs: jax.Array | np.ndarray,
            action: jax.Array | np.ndarray,
            reward: jax.Array | np.ndarray,
            next_obs: jax.Array | np.ndarray,
            done: jax.Array | np.ndarray,
    ) -> 'GPUReplayBuffer':
        """Add one transition for each env (vectorized) in a JIT-compatible way."""
        obs_arr = _reshape_obs_leaf(
            obs, self.obs_shape, self.n_envs, self.obs_dtype)
        next_obs_arr = _reshape_obs_leaf(
            next_obs, self.obs_shape, self.n_envs, self.obs_dtype)
        new_observations = self.observations.at[self.ptr].set(obs_arr)
        new_next_observations = self.next_observations.at[self.ptr].set(
            next_obs_arr)

        action_arr = _reshape_action(
            action, self.action_shape, self.n_envs, self.action_dtype)
        reward_arr = _reshape_scalar_vec(reward, self.n_envs, jnp.float32)
        done_arr = _reshape_scalar_vec(done, self.n_envs, jnp.float32)

        new_actions = self.actions.at[self.ptr].set(action_arr)
        new_rewards = self.rewards.at[self.ptr].set(reward_arr)
        new_dones = self.dones.at[self.ptr].set(done_arr)

        bias_enabled = self.raw_linear_decay_steps != 0
        new_timestamps = jax.lax.cond(
            bias_enabled,
            lambda ts: ts.timestamps.at[ts.ptr].set(ts.current_time),
            lambda ts: ts.timestamps,
            self,
        )
        new_current_time = jax.lax.cond(
            bias_enabled,
            lambda ts: ts.current_time + jnp.array(1, dtype=ts.current_time.dtype),
            lambda ts: ts.current_time,
            self,
        )

        new_ptr = (self.ptr + 1) % self.max_time_size
        new_time_size = jnp.minimum(self.time_size + 1, self.max_time_size)
        new_size = new_time_size * jnp.array(self.n_envs, dtype=new_time_size.dtype)

        return self.replace(
            observations=new_observations,
            actions=new_actions,
            rewards=new_rewards,
            dones=new_dones,
            next_observations=new_next_observations,
            ptr=new_ptr,
            time_size=new_time_size,
            size=new_size,
            timestamps=new_timestamps,
            current_time=new_current_time,
        )

    def _valid_time_mask(self) -> jax.Array:
        idx = jnp.arange(self.max_time_size, dtype=self.time_size.dtype)
        return idx < self.time_size

    def _valid_start_mask(self, n_step: int) -> jax.Array:
        idx = jnp.arange(self.max_time_size, dtype=self.time_size.dtype)
        if n_step <= 1:
            return idx < self.time_size

        valid_count = jnp.maximum(self.time_size - int(n_step) + 1, 1)
        age_order = (idx - self.ptr) % self.max_time_size
        full_mask = age_order < valid_count
        not_full_mask = idx < valid_count
        is_full = self.size >= self.max_time_size * self.n_envs
        return jnp.where(is_full, full_mask, not_full_mask)

    def _linear_bias_weights(self) -> jax.Array:
        age = (self.current_time - self.timestamps).astype(jnp.float32)
        decay = jnp.array(max(self.linear_decay_steps, 1), dtype=jnp.float32)
        min_weight = jnp.array(self.min_weight, dtype=jnp.float32)

        if self.raw_linear_decay_steps > 0:
            return jnp.maximum(min_weight, 1.0 - age / decay)
        return jnp.minimum(1.0, min_weight + age / decay)

    def _biased_time_probabilities(self, n_step: int = 1) -> jax.Array:
        valid = self._valid_start_mask(n_step)
        weights = jnp.where(valid, self._linear_bias_weights(), 0.0)
        weight_sum = jnp.sum(weights)

        valid_count = jnp.maximum(jnp.sum(valid), 1)
        uniform_prob = valid.astype(jnp.float32) / valid_count.astype(jnp.float32)

        safe_weight_sum = jnp.maximum(weight_sum, jnp.finfo(jnp.float32).tiny)
        biased_prob = weights / safe_weight_sum
        return jnp.where(weight_sum > 0.0, biased_prob, uniform_prob)

    def _sample_uniform_time_indices(self, key: jax.Array, batch_size: int, n_step: int) -> jax.Array:
        if n_step <= 1:
            max_time = jnp.maximum(self.time_size, 1)
            return jax.random.randint(key, (batch_size,), 0, max_time)

        valid_count = jnp.maximum(self.time_size - int(n_step) + 1, 1)
        offsets = jax.random.randint(key, (batch_size,), 0, valid_count)
        full_indices = (self.ptr + offsets) % self.max_time_size
        is_full = self.size >= self.max_time_size * self.n_envs
        return jnp.where(is_full, full_indices, offsets)

    def _sample_biased_time_indices(self, key: jax.Array, batch_size: int, n_step: int) -> jax.Array:
        def sample_non_empty(_: jax.Array) -> jax.Array:
            probabilities = self._biased_time_probabilities(n_step)
            return jax.random.choice(
                key,
                self.max_time_size,
                shape=(batch_size,),
                p=probabilities,
            )

        return jax.lax.cond(
            self.time_size > 0,
            sample_non_empty,
            lambda _: jnp.zeros((batch_size,), dtype=jnp.int32),
            operand=jnp.array(0, dtype=jnp.int32),
        )

    def _sample_time_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        if self.raw_linear_decay_steps == 0:
            return self._sample_uniform_time_indices(key, batch_size, 1)
        return self._sample_biased_time_indices(key, batch_size, 1)

    def _sample_n_step_time_indices(self, key: jax.Array, batch_size: int, n_step: int) -> jax.Array:
        if self.raw_linear_decay_steps == 0:
            return self._sample_uniform_time_indices(key, batch_size, n_step)
        return self._sample_biased_time_indices(key, batch_size, n_step)

    def _gather_n_step(
        self,
        time_idx: jax.Array,
        env_idx: jax.Array,
        n_step: int,
        gamma: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        offsets = jnp.arange(int(n_step), dtype=time_idx.dtype)
        sequence_idx = (time_idx[:, None] + offsets[None, :]) % self.max_time_size
        sequence_env_idx = env_idx[:, None]

        rewards = self.rewards[sequence_idx, sequence_env_idx]
        dones = self.dones[sequence_idx, sequence_env_idx]

        shifted_dones = jnp.concatenate(
            [jnp.zeros_like(dones[:, :1]), dones[:, :-1]],
            axis=1,
        )
        reward_masks = jnp.cumprod(1.0 - shifted_dones, axis=1)
        discount_powers = jnp.power(jnp.asarray(gamma, dtype=jnp.float32), offsets.astype(jnp.float32))
        n_step_rewards = jnp.sum(rewards * reward_masks * discount_powers[None, :], axis=1)

        has_done = dones > 0.0
        first_done = jnp.argmax(has_done, axis=1)
        final_offsets = jnp.where(jnp.any(has_done, axis=1), first_done, int(n_step) - 1)
        final_time_idx = sequence_idx[jnp.arange(time_idx.shape[0]), final_offsets]

        next_observations = _gather_time_env(self.next_observations, final_time_idx, env_idx)
        final_dones = _gather_time_env(self.dones, final_time_idx, env_idx)
        effective_n_steps = jnp.sum(reward_masks, axis=1)
        discounts = jnp.power(jnp.asarray(gamma, dtype=jnp.float32), effective_n_steps)

        return n_step_rewards, final_dones, next_observations, discounts

    def sample(
            self,
            key: jax.Array,
            batch_size: int,
            n_step: int = 1,
            gamma: float = 0.99,
        ) -> Batch:
            """Sample a batch of transitions.

            Args:
            key: PRNGKey
            batch_size: number of transitions
            """

            key_t, key_e = jax.random.split(key, 2)
            time_idx = self._sample_n_step_time_indices(key_t, batch_size, n_step)
            env_idx = jax.random.randint(key_e, (batch_size,), 0, self.n_envs)

            observations = _gather_time_env(self.observations, time_idx, env_idx)
            actions = _gather_time_env(self.actions, time_idx, env_idx)
            rewards, dones, next_observations, discounts = self._gather_n_step(
                time_idx, env_idx, n_step, gamma)

            return Batch(
                observations=observations,
                actions=actions,
                rewards=rewards[:, None],
                dones=dones[:, None],
                next_observations=next_observations,
                discounts=discounts[:, None],
            )


JAXReplayBuffer = GPUReplayBuffer
