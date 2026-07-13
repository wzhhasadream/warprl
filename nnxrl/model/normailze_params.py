import jax.numpy as jnp
from flax import nnx
import jax

def normalize_linear_kernel(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    if kernel.ndim == 2:  
        axis = 0
    elif kernel.ndim == 3: 
        axis = 1
    else:
        raise ValueError(f"Unsupported Linear kernel shape: {kernel.shape}")

    norm = jnp.linalg.norm(kernel, axis=axis, keepdims=True)
    return kernel / jnp.maximum(norm, eps)



def normalize_scale_bias(
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Normalize affine scale and bias jointly to sqrt(feature_dim)."""
    feature_dim = scale.shape[-1]
    sqsum = jnp.sum(scale * scale + bias * bias, axis=-1, keepdims=True)
    factor = jnp.sqrt(jnp.asarray(feature_dim, dtype=scale.dtype)) * jax.lax.rsqrt(sqsum + eps)
    return scale * factor, bias * factor


def normalize_scale(scale: jax.Array, eps: float = 1e-8) -> jax.Array:
    feature_dim = scale.shape[-1]
    sqsum = jnp.sum(scale * scale, axis=-1, keepdims=True)
    factor = jnp.sqrt(jnp.asarray(feature_dim, dtype=scale.dtype)
                      ) * jax.lax.rsqrt(sqsum + eps)
    return scale * factor


def project_param(module: nnx.Module) -> None:
    for _, m in nnx.iter_modules(module):
        if isinstance(m, nnx.Linear):
            m.kernel[...] = normalize_linear_kernel(m.kernel[...])


        if isinstance(m, (nnx.LayerNorm, nnx.BatchNorm)):
            scale = getattr(m, "scale", None)
            bias = getattr(m, "bias", None)
            if scale is not None and bias is not None:
                scale[...], bias[...] = normalize_scale_bias(
                    scale[...], bias[...]
                )

        elif hasattr(nnx, "RMSNorm") and isinstance(m, nnx.RMSNorm):
            scale = getattr(m, "scale", None)
            if scale is not None:
                scale[...] = normalize_scale(scale[...])
