from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

def _reshape_to_samples(batch: jax.Array, obs_shape: tuple[int, ...]) -> jax.Array:
    """Flatten leading dims into a single sample dimension."""
    batch = jnp.asarray(batch, dtype=jnp.float32)
    return batch.reshape((-1,) + obs_shape)


class RMS(nnx.Module):
    def __init__(
        self,
        obs_shape: int | Sequence[int],
        epsilon: float = 1e-8,
    ) -> None:
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)

        self.mean = nnx.BatchStat(jnp.zeros(obs_shape, dtype=jnp.float32))
        self.var = nnx.BatchStat(jnp.ones(obs_shape, dtype=jnp.float32))
        self.count = nnx.BatchStat(jnp.asarray(epsilon, dtype=jnp.float32))
        self.epsilon = epsilon


    def update(self, batch: jax.Array) -> None:
        batch = _reshape_to_samples(batch, self.mean.value.shape)

        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean.value
        total_count = self.count.value + batch_count

        new_mean = self.mean.value + delta * batch_count / total_count
        m2 = (
            self.var.value * self.count.value
            + batch_var * batch_count
            + jnp.square(delta) * self.count.value * batch_count / total_count
        )

        self.mean.value = new_mean
        self.var.value = m2 / total_count
        self.count.value = total_count

    def normalize(self, batch: jax.Array, update: bool = True) -> jax.Array:

        normalized_batch = (batch - self.mean.value) / jnp.sqrt(self.var.value + self.epsilon)
        
        if update:
            self.update(batch)
        return normalized_batch


class OnPolicyRMS(nnx.Module):
    """RMS normalizer for on-policy rollouts with a frozen forward snapshot."""

    def __init__(
        self,
        obs_shape: int | Sequence[int],
        epsilon: float = 1e-8,
    ) -> None:
        # Updated from rollout observations.
        self.rms = RMS(obs_shape, epsilon)
        # Used by policy/value forward passes until the next sync.
        self.frozen_rms = RMS(obs_shape, epsilon)

    def normalize(self, batch: jax.Array, update: bool = False) -> jax.Array:
        if update:
            self.rms.update(batch)
        return self.frozen_rms.normalize(batch, update=False)

    def update(self, batch: jax.Array) -> None:
        """Update live rollout statistics without changing forward normalization."""
        self.rms.update(batch)

    def sync(self) -> None:
        """Freeze the latest rollout statistics for subsequent forward passes."""
        self.frozen_rms.mean.value = self.rms.mean.value
        self.frozen_rms.var.value = self.rms.var.value
        self.frozen_rms.count.value = self.rms.count.value

    def __call__(self, batch: jax.Array, update: bool = False) -> jax.Array:
        return self.normalize(batch, update)



class RewardNormalizer(nnx.Module):
    """Reward normalization state based on discounted-return statistics."""

    def __init__(
        self,
        num_envs: int | None = None,
        gamma: float = 0.99,
        g_max: float = 5.0,
        epsilon: float = 1e-8,
        use_max_bound: bool = True,
    ) -> None:
        shape = () if num_envs is None else (num_envs,)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.use_max_bound = use_max_bound
        self.g = nnx.BatchStat(jnp.zeros(shape, dtype=jnp.float32))
        self.g_rms = RMS((), epsilon=epsilon)
        self.g_abs_max = nnx.BatchStat(jnp.array(0.0, dtype=jnp.float32))

    def update(self, rewards: jax.Array, episode_dones: jax.Array) -> None:
        """Update discounted-return statistics from one environment step."""
        rewards = jnp.asarray(rewards, dtype=jnp.float32).reshape(
            self.g.value.shape
        )
        dones = jnp.asarray(episode_dones, dtype=jnp.float32).reshape(
            self.g.value.shape
        )
        g = self.gamma * (1.0 - dones) * self.g.value + rewards
        self.g.value = g
        self.g_rms.update(g)
        self.g_abs_max.value = jnp.maximum(
            self.g_abs_max.value, jnp.max(jnp.abs(g))
        )

    def denominator(self) -> jax.Array:
        """Return the reward scaling denominator."""
        var_denominator = jnp.sqrt(self.g_rms.var.value + self.epsilon)
        max_denominator = self.g_abs_max.value / jnp.maximum(
            self.g_max, self.epsilon
        )
        if self.use_max_bound:
            return jnp.maximum(var_denominator, max_denominator)
        return var_denominator

    def normalize(self, rewards: jax.Array) -> jax.Array:
        """Scale rewards using current discounted-return statistics."""
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        return rewards / self.denominator()
