import jax
import jax.numpy as jnp


def build_truncated_zeta_cdf(
    mu: float | jax.Array = 2.0,
    max_n: int = 16,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Build a truncated zeta CDF over integers [1, max_n]."""
    ns = jnp.arange(1, max_n + 1, dtype=dtype)
    pmf = ns ** (-jnp.asarray(mu, dtype=dtype))
    pmf = pmf / jnp.sum(pmf)
    return jnp.cumsum(pmf)


def sample_integer_from_cdf(
    key: jax.Array,
    cdf: jax.Array,
    shape: tuple[int, ...] = (),
) -> jax.Array:
    """Sample 1-indexed integers from a CDF, matching FlashSAC's repeat length format."""
    uniforms = jax.random.uniform(key, shape=shape, dtype=cdf.dtype)
    indices = jnp.argmax(uniforms[..., None] < cdf, axis=-1)
    return (indices + 1).astype(jnp.int32)


def sample_truncated_zeta(
    key: jax.Array,
    mu: float | jax.Array = 2.0,
    max_n: int = 16,
    shape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Sample repeat lengths from the truncated zeta distribution used by FlashSAC."""
    cdf = build_truncated_zeta_cdf(mu=mu, max_n=max_n, dtype=dtype)
    return sample_integer_from_cdf(key, cdf, shape=shape)
