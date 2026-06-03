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
    def create(cls, obs_shape: int | Sequence[int], epsilon: float = 1e-4):
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
        epsilon: float = 1e-4,
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

        norm_state = (batch - rms.mean) / jnp.sqrt(rms.var + rms.epsilon)
        return norm_state, rms


