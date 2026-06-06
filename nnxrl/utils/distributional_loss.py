import jax.numpy as jnp
from functools import lru_cache
import numpy as np
from flax import nnx
import jax

@lru_cache(maxsize=None)
def make_taus(num_quantiles: int):
    """Return quantile midpoints tau_i = (i + 0.5) / N."""
    return (np.arange(num_quantiles, dtype=np.float32) + 0.5) / num_quantiles


def huber_replace(diff, kappa: float = 1.0):
    """Huber loss used by quantile regression."""
    return jnp.where(
        jnp.abs(diff) <= kappa,
        0.5 * diff ** 2,
        kappa * (jnp.abs(diff) - 0.5 * kappa),
    )


def quantile_huber_loss(diff, taus, kappa: float = 1.0):
    """Pairwise quantile Huber loss.

    Args:
        diff: Pairwise TD errors with shape [B, N, N], where
            diff[b, i, j] = target_quantile[b, j] - predicted_quantile[b, i].
        taus: Quantile midpoints with shape [N].
        kappa: Huber threshold.
    """
    weight = jnp.abs(taus[..., None] - (diff < 0).astype(diff.dtype))
    huber_loss = huber_replace(diff, kappa)
    return (weight * huber_loss / kappa).sum(axis=1).mean()

@nnx.vmap(in_axes=(0, None, None))
def quantile_loss(q_distributional, target_q_distributional, kappa: float = 1.0):
    """Distributional quantile regression loss for an ensemble critic.

    Public call shapes:
        q_distributional: [num_q, B, N]
        target_q_distributional: [B, N]

    Inside this vmapped function:
        q_distributional: [B, N]
        target_q_distributional: [B, N]
    """
    assert q_distributional.shape == target_q_distributional.shape, (
        "The shape of q_distributional and target_q_distributional should be the same"
    )
    num_quantiles = q_distributional.shape[1]
    taus = jnp.asarray(make_taus(num_quantiles),
                           dtype=q_distributional.dtype)
    diff = target_q_distributional[:, None, :] - q_distributional[:, :, None]
    return quantile_huber_loss(diff, taus, kappa)





@lru_cache(maxsize=None)
def make_bin_values(num_bins: int, min_v: float = -5.0, max_v: float = 5.0, dtype=jnp.float32):
    """Return fixed categorical value atoms in [min_v, max_v]."""
    return jnp.linspace(min_v, max_v, num_bins, dtype=dtype)


def categorical_q_values(
    logits: jax.Array,
    min_v: float = -5.0,
    max_v: float = 5.0,
) -> jax.Array:
    """Compute expected Q values from categorical critic logits."""
    num_bins = logits.shape[-1]
    bin_values = make_bin_values(num_bins, min_v, max_v, dtype=logits.dtype)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * bin_values, axis=-1, keepdims=True)          #(num_q, batch, 1)


def select_min_q_logits(
    logits: jax.Array,
    min_v: float = -5.0,
    max_v: float = 5.0,
) -> jax.Array:
    """Select logits from the critic with the minimum expected Q value.

    Args:
        logits: Categorical critic logits with shape [num_q, batch, num_bins].
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [num_q, batch, num_bins], got {logits.shape}")

    q_values = categorical_q_values(logits, min_v, max_v)
    min_indices = jnp.argmin(q_values, axis=0)  # (batch, 1)
    gather_indices = min_indices[None, :, :]
    return jnp.take_along_axis(logits, gather_indices, axis=0)[0]    # # [batch, num_bins]





def categorical_projection(
    target_logits: jax.Array,
    target_values: jax.Array,
    min_v: float = -5.0,
    max_v: float = 5.0,
) -> jax.Array:
    """Project target distribution masses at target_values onto fixed value bins."""
    num_bins = target_logits.shape[-1]
    dtype = target_logits.dtype
    bin_width = (max_v - min_v) / (num_bins - 1)
    target_values = jnp.asarray(target_values, dtype=dtype)
    target_values = jnp.clip(target_values, min_v, max_v)

    b = (target_values - min_v) / bin_width
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    lower = jnp.clip(lower, 0, num_bins - 1)
    upper = jnp.clip(upper, 0, num_bins - 1)

    target_probs = jax.nn.softmax(target_logits, axis=-1)
    lower_weight = upper.astype(dtype) + (lower == upper).astype(dtype) - b
    upper_weight = b - lower.astype(dtype)

    lower_mask = jax.nn.one_hot(lower, num_bins, dtype=dtype)
    upper_mask = jax.nn.one_hot(upper, num_bins, dtype=dtype)
    projected = jnp.sum(
        target_probs[..., :, None]
        * (
            lower_weight[..., :, None] * lower_mask
            + upper_weight[..., :, None] * upper_mask
        ),
        axis=-2,
    )
    return jax.lax.stop_gradient(projected)


@nnx.vmap(in_axes=(0, None))
def categorical_ce_loss(
    logits: jax.Array,
    target_probs: jax.Array,
) -> jax.Array:
    """Categorical distributional critic cross-entropy loss.

    logits and target_probs have shape [B, num_bins] inside this vmapped
    function and [num_q, B, num_bins] at the public call site.
    """
    assert logits.shape == target_probs.shape, (
        "The shape of logits and target_probs should be the same"
    )
    target_probs = jax.lax.stop_gradient(target_probs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
