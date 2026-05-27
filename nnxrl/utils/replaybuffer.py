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




def create_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_observations: np.ndarray
) -> Batch:
    """
    Create a batch dictionary for JAX.


    Args:
        observations: shape (batch_size, obs_dim)
        actions: shape (batch_size, action_dim)
        rewards: shape (batch_size, 1)
        dones: shape (batch_size, 1)
        next_observations: shape (batch_size, obs_dim)

    Returns:
        Batch dictionary with JAX arrays
    """
    return Batch(
        jnp.array(observations), 
        jnp.array(actions), 
        jnp.array(rewards.reshape(-1, 1)), 
        jnp.array(dones.reshape(-1, 1)), 
        jnp.array(next_observations)
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


    def sample(self, batch_size: int) -> Batch:
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

        if self._raw_linear_decay_steps == 0:
            if self.optimize_memory_usage:
                if not self.full:  
                    batch_index = np.random.randint(0, self.ptr, size=batch_size)
                else:  
                    offsets = np.random.randint(0, self.max_time_size - 1, size=batch_size) # avoid sampling from the current pointer location
                    batch_index = (self.ptr + 1 + offsets) % self.max_time_size
            else:
                batch_index = np.random.randint(0, self.time_size, size=batch_size)
        else:
            if self.use_approximate_sampling:
                batch_index = self._sample_with_approximate_bias(batch_size)
            else:
                batch_index = self._sample_with_bias(batch_size)

        env_index = np.random.randint(0, self.n_envs, size=batch_size)

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_index + 1) % self.max_time_size, env_index].copy()
            for i in range(batch_size):
                idx = batch_index[i]
                env_idx = env_index[i]
                if idx in self.truncated_next_obs and env_idx in self.truncated_next_obs[idx]:
                    next_obs[i] = self.truncated_next_obs[idx][env_idx]
        else:
            next_obs = self.next_observations[batch_index, env_index]

        return create_batch(
            observations=self.observations[batch_index, env_index], 
            actions=self.actions[batch_index, env_index],
            rewards=self.rewards[batch_index, env_index],
            dones=self.dones[batch_index, env_index],
            next_observations=next_obs 
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
        observation_shape: tuple[int, ...] | int,
        action_shape: tuple[int, ...],
        max_size: int = int(10e6),
        n_envs: int = 1,
        *,
        obs_dtype: jnp.dtype = jnp.float32,
        action_dtype: jnp.dtype = jnp.float32,
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
    ) -> 'GPUReplayBuffer':
        """Initialize a JIT-friendly replay buffer.

        Notes:
        - If `linear_decay_steps != 0`, sampling uses an exact time-biased distribution
            (matching the CPU ReplayBuffer implementation in this repo):
            * `> 0`: newer-biased (prefer recent experiences)
            * `< 0`: older-biased (prefer old experiences)
        """


        max_time_size = max(int(max_size) // int(n_envs), 1)

        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")

        max_time_size = int(max_time_size)

        def zeros_obs(shape_spec: tuple[int, ...]) -> jax.Array:
            return jnp.zeros((max_time_size, n_envs, *shape_spec), dtype=obs_dtype)

        if isinstance(observation_shape, tuple):
            observations = zeros_obs(observation_shape)
            next_observations = zeros_obs(observation_shape)
        elif isinstance(observation_shape, int):
            observation_shape = (observation_shape,)
            observations = zeros_obs(observation_shape)
            next_observations = zeros_obs(observation_shape)
        else:
            raise TypeError(
                f"expected obs_dim to be int|tuple, got {type(observation_shape)}")

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

    def _linear_bias_weights(self) -> jax.Array:
        age = (self.current_time - self.timestamps).astype(jnp.float32)
        decay = jnp.array(max(self.linear_decay_steps, 1), dtype=jnp.float32)
        min_weight = jnp.array(self.min_weight, dtype=jnp.float32)

        if self.raw_linear_decay_steps > 0:
            return jnp.maximum(min_weight, 1.0 - age / decay)
        return jnp.minimum(1.0, min_weight + age / decay)

    def _biased_time_probabilities(self) -> jax.Array:
        valid = self._valid_time_mask()
        weights = jnp.where(valid, self._linear_bias_weights(), 0.0)
        weight_sum = jnp.sum(weights)

        valid_count = jnp.maximum(jnp.sum(valid), 1)
        uniform_prob = valid.astype(jnp.float32) / valid_count.astype(jnp.float32)

        safe_weight_sum = jnp.maximum(weight_sum, jnp.finfo(jnp.float32).tiny)
        biased_prob = weights / safe_weight_sum
        return jnp.where(weight_sum > 0.0, biased_prob, uniform_prob)

    def _sample_uniform_time_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        max_time = jnp.maximum(self.time_size, 1)
        return jax.random.randint(key, (batch_size,), 0, max_time)

    def _sample_biased_time_indices(self, key: jax.Array, batch_size: int) -> jax.Array:
        def sample_non_empty(_: jax.Array) -> jax.Array:
            probabilities = self._biased_time_probabilities()
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
            return self._sample_uniform_time_indices(key, batch_size)
        return self._sample_biased_time_indices(key, batch_size)

    def sample(
            self,
            key: jax.Array,
            batch_size: int
        ) -> Batch:
            """Sample a batch of transitions.

            Args:
            key: PRNGKey
            batch_size: number of transitions
            """

            key_t, key_e = jax.random.split(key, 2)
            time_idx = self._sample_time_indices(key_t, batch_size)
            env_idx = jax.random.randint(key_e, (batch_size,), 0, self.n_envs)

            observations = _gather_time_env(self.observations, time_idx, env_idx)
            next_observations = _gather_time_env(
                self.next_observations, time_idx, env_idx)

            actions = _gather_time_env(self.actions, time_idx, env_idx)
            rewards = _gather_time_env(self.rewards, time_idx, env_idx)
            dones = _gather_time_env(self.dones, time_idx, env_idx)

            return Batch(
                observations=observations,
                actions=actions,
                rewards=rewards[:, None],
                dones=dones[:, None],
                next_observations=next_observations,
            )


JAXReplayBuffer = GPUReplayBuffer
