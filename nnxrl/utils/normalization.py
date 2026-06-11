from typing import  Sequence

import jax
import jax.numpy as jnp
from flax import struct


def _reshape_to_samples(batch: jax.Array, obs_shape: tuple[int, ...]) -> jax.Array:
    """Flatten leading dims into a single sample dimension."""
    batch = jnp.asarray(batch, dtype=jnp.float32)
    return batch.reshape((-1,) + obs_shape)


@struct.dataclass
class RMS:
    """Running mean/variance state for normalization."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array
    epsilon: float

    @classmethod
    def create(cls, obs_shape: int | Sequence[int], epsilon: float = 1e-8):
        """Create an RMSState for a single observation tensor."""
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        return cls(
            mean=jnp.zeros(obs_shape, dtype=jnp.float32),
            var=jnp.ones(obs_shape, dtype=jnp.float32),
            count=jnp.array(epsilon, dtype=jnp.float32),
            epsilon=epsilon
        )

    @classmethod
    def load(
        cls,
        mean: jax.Array,
        var: jax.Array,
        count: int | float | jax.Array,
        epsilon: float = 1e-8,
    ) -> "RMS":
        mean = jnp.asarray(mean, dtype=jnp.float32)
        var = jnp.asarray(var, dtype=jnp.float32)
        count = jnp.asarray(count, dtype=jnp.float32)

        if mean.shape != var.shape:
            raise ValueError(
                f"mean.shape {mean.shape} must match var.shape {var.shape}"
            )
        if count.ndim != 0:
            raise ValueError(f"count must be a scalar, got shape {count.shape}")
        if float(count) <= 0.0:
            raise ValueError(f"count must be positive, got {float(count)}")

        return cls(
            mean=mean,
            var=var,
            count=count,
            epsilon=epsilon
        )

    def update(
        self,
        batch: jax.Array ,
    ) -> 'RMS':
        """Update RMS statistics with a new batch using Welford's algorithm."""
        batch = _reshape_to_samples(batch, tuple(self.mean.shape))

        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        # Welford's algorithm for combining statistics.
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        return self.replace(mean=new_mean, var=new_var, count=tot_count)

    def normalize(self, batch: jax.Array, update: bool = True):
        rms = self
        if update:
            rms = self.update(batch)

        normalized_state = (batch - rms.mean) / jnp.sqrt(rms.var + rms.epsilon)
        return normalized_state, rms


@struct.dataclass
class RewardNormalizer:
    """Reward normalization state based on discounted-return statistics."""

    gamma: float
    g_max: float
    g: jax.Array
    g_rms: RMS
    g_abs_max: jax.Array
    epsilon: float
    use_max_bound: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(
        cls,
        num_envs: int | None = None,
        gamma: float = 0.99,
        g_max: float = 5.0,
        epsilon: float = 1e-8,
        use_max_bound: bool = True,
    ) -> "RewardNormalizer":
        """Create reward normalizer state.

        If num_envs is None, a scalar discounted return is tracked. Otherwise,
        one discounted return is tracked for each parallel environment.
        """
        shape = () if num_envs is None else (num_envs,)
        return cls(
            gamma=gamma,
            g_max=g_max,
            g=jnp.zeros(shape, dtype=jnp.float32),
            g_rms=RMS.create((), epsilon=epsilon),
            g_abs_max=jnp.array(0.0, dtype=jnp.float32),
            epsilon=epsilon,
            use_max_bound=use_max_bound,
        )

    @jax.jit
    def update(self, rewards: jax.Array, dones: jax.Array) -> "RewardNormalizer":
        """Update discounted-return statistics from one environment step."""
        rewards = jnp.asarray(rewards, dtype=jnp.float32).reshape(self.g.shape)
        dones = jnp.asarray(dones, dtype=jnp.float32).reshape(self.g.shape)

        g = self.gamma * (1.0 - dones) * self.g + rewards
        g_rms = self.g_rms.update(g)
        g_abs_max = jnp.maximum(self.g_abs_max, jnp.max(jnp.abs(g)))
        return self.replace(g=g, g_rms=g_rms, g_abs_max=g_abs_max)

    def denominator(self) -> jax.Array:
        """Return the reward scaling denominator."""
        var_denominator = jnp.sqrt(self.g_rms.var + self.epsilon)
        max_denominator = self.g_abs_max / jnp.maximum(self.g_max, self.epsilon)
        if self.use_max_bound:
            return jnp.maximum(var_denominator, max_denominator)
        else:
            return var_denominator

    @jax.jit
    def normalize(self, rewards: jax.Array) -> jax.Array:
        """Scale rewards using current discounted-return statistics."""
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        denominator = self.denominator()

        while denominator.ndim < rewards.ndim:
            denominator = jnp.expand_dims(denominator, axis=-1)

        return rewards / denominator

    @jax.jit
    def update_and_normalize(
        self,
        rewards: jax.Array,
        dones: jax.Array,
    ) -> tuple[jax.Array, "RewardNormalizer"]:
        """Update statistics from rewards and return normalized rewards."""
        normalizer = self.update(rewards, dones)
        return normalizer.normalize(rewards), normalizer
