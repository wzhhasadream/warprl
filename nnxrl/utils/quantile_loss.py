import jax.numpy as jnp
from functools import lru_cache
import numpy as np
from flax import nnx

@lru_cache(maxsize=None)
def make_taus(num_quantiles: int):
    return (np.arange(num_quantiles, dtype=np.float32) + 0.5) / num_quantiles

def huber_replace(diff, kappa: float = 1.0):
    return jnp.where(
        jnp.abs(diff) <= kappa,
        0.5 * diff ** 2,
        kappa * (jnp.abs(diff) - 0.5 * kappa),
    )


def quantile_huber_loss(diff, bro_taus, kappa: float = 1.0):
    # diff: [B, num_quantile, num_quantile]
    # taus: [num_quantile] or [1, num_quantile]
    weight = jnp.abs(bro_taus[..., None] - (diff < 0).astype(diff.dtype))
    huber_loss = huber_replace(diff, kappa)
    return (weight * huber_loss / kappa).sum(axis=1).mean()

@nnx.vmap(in_axes=(0, None, None))
def quantile_loss(q_distributional, target_q_distributional, kappa: float = 1.0):
    assert q_distributional.shape == target_q_distributional.shape, (
        "The shape of q_distributional and target_q_distributional should be the same"
    )
    num_quantiles = q_distributional.shape[1]
    taus = jnp.asarray(make_taus(num_quantiles),
                           dtype=q_distributional.dtype)
    diff = target_q_distributional[:, None, :] - q_distributional[:, :, None]
    return quantile_huber_loss(diff, taus, kappa)

