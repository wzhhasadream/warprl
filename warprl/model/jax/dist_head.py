"""JAX distributional critic heads and losses."""

import jax
import jax.numpy as jnp
from flax import nnx



class CategoricalPolicy(nnx.Module):
    """Fixed-support categorical critic head for C51-style value estimates."""

    def __init__(self, num_bins: int, v_min: float, v_max: float) -> None:
        self._bins = nnx.Variable(
            jnp.linspace(v_min, v_max, num_bins, dtype=jnp.float32)[None, :]
        )

    @property
    def bins(self) -> jax.Array:
        if self._bins.value.ndim == 3:
            return self._bins.value[0]
        else:
            return self._bins.value

    def q_values(self, logits: jax.Array) -> jax.Array:
        """Compute expected values for logits with arbitrary leading dimensions."""
        logits = jnp.asarray(logits).astype(jnp.float32)
        bins = self.bins
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * bins, axis=-1, keepdims=True)

    def select_min_logits(self, logits: jax.Array) -> jax.Array:
        """Select each batch element's logits from the lowest-value critic head."""
        indices = jnp.argmin(self.q_values(logits), axis=0)
        gather_indices = jnp.broadcast_to(
            indices[None, ...], (1, logits.shape[1], logits.shape[2])
        )
        return jnp.take_along_axis(logits, gather_indices, axis=0)[0]

    def target_probs(
        self,
        target_logits: jax.Array,
        target_values: jax.Array,
    ) -> jax.Array:
        """Project target categorical masses onto the fixed C51 support."""
        target_logits = jnp.asarray(target_logits, dtype=jnp.float32)
        bins = self.bins
        num_bins = bins.shape[-1]
        target_values = jnp.asarray(target_values, dtype=jnp.float32)
        target_values = jnp.broadcast_to(target_values, target_logits.shape)
        v_min = bins[..., :1]
        v_max = bins[..., -1:]
        bin_width = bins[..., 1:2] - v_min
        target_values = jnp.clip(target_values, v_min, v_max)

        positions = (target_values - v_min) / bin_width
        lower = jnp.floor(positions).astype(jnp.int32)
        upper = jnp.ceil(positions).astype(jnp.int32)
        lower = jnp.clip(lower, 0, num_bins - 1)
        upper = jnp.clip(upper, 0, num_bins - 1)

        probs = jax.nn.softmax(target_logits, axis=-1)
        lower_weight = upper.astype(jnp.float32) + (lower == upper).astype(jnp.float32) - positions
        upper_weight = positions - lower.astype(jnp.float32)
        lower_mask = jax.nn.one_hot(lower, num_bins, dtype=jnp.float32)
        upper_mask = jax.nn.one_hot(upper, num_bins, dtype=jnp.float32)
        projected = jnp.sum(
            probs[..., :, None]
            * (
                lower_weight[..., :, None] * lower_mask
                + upper_weight[..., :, None] * upper_mask
            ),
            axis=-2,
        )
        return jax.lax.stop_gradient(projected)

    def _loss_one(self, logits: jax.Array, target_probs: jax.Array) -> jax.Array:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))

    def loss(self, logits: jax.Array, target_probs: jax.Array) -> jax.Array:
        """Return C51 cross-entropy for one critic or an ensemble of critics."""
        logits = jnp.asarray(logits, dtype=jnp.float32)
        target_probs = jax.lax.stop_gradient(
            jnp.asarray(target_probs, dtype=jnp.float32)
        )
        if logits.ndim == 2:
            return self._loss_one(logits, target_probs)
        return jax.vmap(self._loss_one, in_axes=(0, None))(logits, target_probs)


class QuantilePolicy(nnx.Module):
    """Quantile critic head with quantile-Huber regression loss."""

    def __init__(self, num_taus: int) -> None:
        self._taus = nnx.Variable(
            (jnp.arange(num_taus, dtype=jnp.float32) + 0.5) / num_taus
        )

    @property
    def taus(self) -> jax.Array:
        if self._taus.value.ndim == 2:
            return self._taus.value[0]
        else:
            return self._taus.value
    
    def q_values(self, quantiles: jax.Array) -> jax.Array:
        """Compute expected values from quantiles along the last axis."""
        return jnp.mean(jnp.asarray(quantiles, dtype=jnp.float32), axis=-1, keepdims=True)

    def _loss_one(
        self,
        quantiles: jax.Array,
        target_quantiles: jax.Array,
        kappa: float,
    ) -> jax.Array:
        diff = target_quantiles[:, None, :] - quantiles[:, :, None]
        abs_diff = jnp.abs(diff)
        huber = jnp.where(
            abs_diff <= kappa,
            0.5 * jnp.square(diff),
            kappa * (abs_diff - 0.5 * kappa),
        )
        taus = self.taus
        weight = jnp.abs(taus[:, None] - (diff < 0).astype(diff.dtype))
        return jnp.mean(jnp.sum(weight * huber / kappa, axis=1))

    def loss(
        self,
        quantiles: jax.Array,
        target_quantiles: jax.Array,
        kappa: float = 1.0,
    ) -> jax.Array:
        """Return quantile-Huber loss for one critic or an ensemble."""
        quantiles = jnp.asarray(quantiles, dtype=jnp.float32)
        target_quantiles = jax.lax.stop_gradient(
            jnp.asarray(target_quantiles, dtype=jnp.float32)
        )
        if quantiles.ndim == 2:
            return self._loss_one(quantiles, target_quantiles, kappa)
        return jax.vmap(self._loss_one, in_axes=(0, None, None))(
            quantiles, target_quantiles, kappa
        )
