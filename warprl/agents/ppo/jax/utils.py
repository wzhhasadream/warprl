import jax
import jax.numpy as jnp


def diagonal_gaussian_kl(
    mu_c: jax.Array,
    std_c: jax.Array,
    mu_o: jax.Array,
    std_o: jax.Array,
) -> jax.Array:
    """Compute KL(old || current) for diagonal Gaussian policies."""
    kl = (
        jnp.log(std_c / std_o)
        + (std_o**2 + (mu_o - mu_c) ** 2) / (2.0 * std_c**2)
        - 0.5
    )
    return kl.sum(axis=-1, keepdims=True)



def adapt_lr(lr, kl, desired_kl: float = 0.01, lr_min: float = 1e-5, lr_max: float = 1e-2, factor: float = 1.5):
    lr = jnp.where(
        kl > 2.0 * desired_kl,
        jnp.maximum(lr / factor, lr_min),
        lr,
    )
    lr = jnp.where(
        (kl < 0.5 * desired_kl) & (kl > 0.0),
        jnp.minimum(lr * factor, lr_max),
        lr,
    )
    return lr  
