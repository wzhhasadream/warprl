"""A minimal fixed-horizon PPO rollout buffer for JAX."""

from __future__ import annotations

from functools import partial
from .base_buffer import BaseBuffer, get_obs_shape, get_action_dim
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any
from .types import RolloutBatch, RolloutTransition, Trajectory
from gymnasium import spaces
from pathlib import Path
import orbax.checkpoint as ocp

def _to_jax_dtype(dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float64:
        return jnp.float32
    return dtype


def _resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    if device is None or isinstance(device, jax.Device):
        return device
    platform, _, index = device.replace("cuda", "gpu").partition(":")
    devices = jax.devices(platform)
    return devices[int(index)] if index else devices[0]


@struct.dataclass
class JaxBuffer(BaseBuffer):
    """Immutable on-policy rollout storage for vectorized feedforward PPO."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    values: jax.Array
    log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    actions_mean: jax.Array
    actions_std: jax.Array
    step: jax.Array
    full: jax.Array

    returns_ready: bool = struct.field(pytree_node=False)
    rollout_steps: int = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    observation_shape: tuple[int, ...] = struct.field(pytree_node=False)
    action_shape: tuple[int, ...] = struct.field(pytree_node=False)

    @staticmethod
    def compute_gae(
        rewards: jax.Array,
        values: jax.Array,
        dones: jax.Array,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute GAE-Lambda advantages and value targets for one rollout.

        ``dones[t]`` marks whether the transition from ``s_t`` to ``s_{t+1}``
        ended an episode. Time-limit values are added to rewards before this
        function is called, so terminal transitions do not bootstrap here.
        """
        gamma = jnp.asarray(gamma, dtype=values.dtype)
        gae_lambda = jnp.asarray(gae_lambda, dtype=values.dtype)

        def scan_step(advantage: jax.Array, inputs: tuple[jax.Array, ...]):
            reward, value, next_value, done = inputs
            mask = 1.0 - done.astype(values.dtype)
            delta = reward + gamma * next_value * mask - value
            advantage = delta + gamma * gae_lambda * mask * advantage
            return advantage, advantage

        _, advantages = jax.lax.scan(
            scan_step,
            jnp.zeros_like(values[0]),
            (rewards, values[:-1], values[1:], dones),
            reverse=True,
        )
        return advantages, advantages + values[:-1]

    @classmethod
    def create(
        cls,
        rollout_steps: int,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *,
        device: str | jax.Device | None = None,
    ) -> "JaxBuffer":
        """Allocate fixed storage for one rollout."""
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError("rollout_steps and num_envs must both be positive")
        # Extract shapes from spaces
        observation_shape = get_obs_shape(observation_space)
        action_dim = get_action_dim(action_space)

        # Handle both int and tuple for obs_shape
        if isinstance(observation_shape, int):
            observation_shape = (observation_shape,)
        else:
            observation_shape = observation_shape
        action_shape = (action_dim, )
        observation_dtype = _to_jax_dtype(observation_space.dtype)
        action_dtype = _to_jax_dtype(action_space.dtype)
        device = _resolve_device(device)
        return cls(
            observations=jnp.empty(
                (rollout_steps, num_envs, *observation_shape), observation_dtype, device=device),
            actions=jnp.empty((rollout_steps, num_envs, *action_shape),
                              action_dtype, device=device),
            rewards=jnp.empty((rollout_steps, num_envs),
                              jnp.float32, device=device),
            dones=jnp.empty((rollout_steps, num_envs), jnp.float32, device=device),
            values=jnp.empty((rollout_steps + 1, num_envs),
                             jnp.float32, device=device),
            log_probs=jnp.empty((rollout_steps, num_envs),
                                jnp.float32, device=device),
            advantages=jnp.empty((rollout_steps, num_envs),
                                 jnp.float32, device=device),
            returns=jnp.empty((rollout_steps, num_envs),
                              jnp.float32, device=device),
            actions_mean=jnp.empty((rollout_steps, num_envs, *action_shape),
                                  action_dtype, device=device),
            actions_std=jnp.empty((rollout_steps, num_envs, *action_shape),
                                 action_dtype, device=device),
            step=jnp.asarray(0, dtype=jnp.int32),
            returns_ready=False,
            rollout_steps=int(rollout_steps),
            num_envs=int(num_envs),
            observation_shape=observation_shape,
            action_shape=action_shape,
            full=jnp.asarray(False),
        )


    @property
    def num_samples(self) -> int:
        return int(self.step) * self.num_envs


    def _array(self, value: jax.Array, dtype: jnp.dtype, shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(value, dtype=dtype).reshape(shape)

    @partial(jax.jit, donate_argnums=0)
    def add(self, transition: RolloutTransition) -> "JaxBuffer":
        """Functionally append one transition after Python-side validation."""
        index = self.step
        next_step = self.step + 1
        terminated = self._array(transition.terminated, jnp.bool_, (self.num_envs,))
        truncated = self._array(transition.truncated, jnp.bool_, (self.num_envs,))
        return self.replace(
            observations=self.observations.at[index].set(
                self._array(transition.observations, self.observations.dtype,
                                (self.num_envs, *self.observation_shape))
            ),
            actions=self.actions.at[index].set(
                self._array(transition.actions, self.actions.dtype,
                                (self.num_envs, *self.action_shape))
            ),
            actions_mean=self.actions_mean.at[index].set(
                self._array(transition.actions_mean, self.actions.dtype, (self.num_envs, *self.action_shape))
            ),
            actions_std=self.actions_std.at[index].set(
                self._array(transition.actions_std, self.actions.dtype, (self.num_envs, *self.action_shape))
            ),
            rewards=self.rewards.at[index].set(self._array(
                transition.rewards, jnp.float32, (self.num_envs,))),
            dones=self.dones.at[index].set(jnp.logical_or(terminated, truncated)),
            values=self.values.at[index].set(self._array(
                transition.values, jnp.float32, (self.num_envs,))),
            log_probs=self.log_probs.at[index].set(self._array(
                transition.log_probs, jnp.float32, (self.num_envs,))),
            step=next_step,
            full=next_step == self.rollout_steps,
            returns_ready=False,
        )
    # NOTE: Keep this undecorated; it is traced by the outer PPO update JIT.
    # @jax.jit
    def compute_returns_and_advantages(
        self,
        last_values: jax.Array,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_scale: float | None = None,
    ) -> "JaxBuffer":
        """Functionally compute returns after Python-side validation."""
        values = self.values.at[-1].set(self._array(last_values,
                                        jnp.float32, (self.num_envs,)))
        rewards = self.rewards
        if reward_scale is not None:
            rewards = rewards * reward_scale
        advantages, returns = self.compute_gae(
            rewards, values, self.dones, gamma, gae_lambda)
        return self.replace(
            values=values,
            advantages=advantages,
            returns=returns,
            returns_ready=True,
        )

    def trajectory(self) -> Trajectory:
        """Return the currently collected rollout without flattening it."""
        count = int(self.step)
        return Trajectory(
            observations=self.observations[:count],
            actions=self.actions[:count],
            actions_mean=self.actions_mean[:count],
            actions_std=self.actions_std[:count],
            log_probs=self.log_probs[:count],
            values=self.values[: count + 1],
            dones=self.dones[:count],
            rewards=self.rewards[:count],
        )

    def can_sample(self) -> bool:
        return bool(self.full) and self.returns_ready

    def _flat_batch(self) -> RolloutBatch:
        return RolloutBatch(
            observations=self.observations.reshape(
                (-1, *self.observation_shape)),
            actions=self.actions.reshape((-1, *self.action_shape)),
            actions_mean=self.actions_mean.reshape((-1, *self.action_shape)),
            actions_std=self.actions_std.reshape((-1, *self.action_shape)),
            values=self.values[:-1].reshape(-1, 1),
            advantages=self.advantages.reshape(-1, 1),
            returns=self.returns.reshape(-1, 1),
            old_log_probs=self.log_probs.reshape(-1, 1),
        )

    def minibatch_indices(self, key: jax.Array, num_mini_batches: int) -> jax.Array:
        """Return one shuffled partition of flattened rollout indices."""
        total = self.rollout_steps * self.num_envs
        if total % num_mini_batches:
            raise ValueError(
                "rollout_steps * num_envs must be divisible by num_mini_batches")
        return jax.random.permutation(key, total).reshape(num_mini_batches, total // num_mini_batches)

    def batch_from_indices(self, indices: jax.Array) -> RolloutBatch:
        """Gather a PPO mini-batch from flattened rollout indices."""
        batch = self._flat_batch()
        return RolloutBatch(*(field[indices] for field in batch))

    def normalize_advantages(self) -> "JaxBuffer":
        advantages = (self.advantages - self.advantages.mean()) / (jnp.std(self.advantages) + 1e-8)
        return self.replace(advantages=advantages)

    # NOTE: Keep this undecorated so the Python signature stays visible.
    # The outer PPO update JIT traces this method together with the minibatch scan.
    # @partial(jax.jit, static_argnames=("num_mini_batches", "num_epochs"))
    def sample(
        self,
        key: jax.Array,
        num_mini_batches: int,
        num_epochs: int = 1
    ) -> RolloutBatch:
        """Return shuffled PPO mini-batches after returns are ready."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not self.returns_ready:
            raise ValueError("Compute returns and advantages before requesting mini-batches.")

        def scan_epoch(next_key: jax.Array, _: None):
            next_key, epoch_key = jax.random.split(next_key)
            return next_key, self.minibatch_indices(epoch_key, num_mini_batches)

        _, indices = jax.lax.scan(scan_epoch, key, None, length=num_epochs)
        indices = indices.reshape(num_epochs * num_mini_batches, -1)
        return self.batch_from_indices(indices)

    def reset(self) -> "JaxBuffer":
        """Functionally reset the write cursor while retaining allocation."""
        return self.replace(
            step=jnp.asarray(0, dtype=jnp.int32),
            full=jnp.asarray(False),
            returns_ready=False,
        )

    def __len__(self) -> int:
        return self.num_samples

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

        with ocp.StandardCheckpointer() as ckpt:
            # Keep an empty replay buffer from being treated as an empty item.
            ckpt.save(checkpoint_dir, {"buffer": self})
            ckpt.wait_until_finished()

    def load(self, checkpoint_dir: str | Path) -> "JaxBuffer":
        checkpoint_dir = Path(checkpoint_dir)
        with ocp.StandardCheckpointer() as ckpt:
            restored = ckpt.restore(checkpoint_dir, {"buffer": self})

        return restored["buffer"]


