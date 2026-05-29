import jax.numpy as jnp
from flax import nnx
import jax

def normalize_linear_kernel(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    if kernel.ndim == 2:   # 单个模型
        axis = (0, 1)
    elif kernel.ndim == 3:  # 集成模型
        axis = (1, 2)
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
    for _, m in module.iter_modules():
        if isinstance(m, nnx.Linear):
            m.kernel.value = normalize_linear_kernel(m.kernel.value)

        # elif isinstance(m, (nnx.LayerNorm, nnx.BatchNorm)):
        #     scale = getattr(m, "scale", None)
        #     bias = getattr(m, "bias", None)
        #     if scale is not None and bias is not None:
        #         scale.value, bias.value = normalize_scale_bias(
        #             scale.value, bias.value
        #         )

        # elif hasattr(nnx, "RMSNorm") and isinstance(m, nnx.RMSNorm):
        #     scale = getattr(m, "scale", None)
        #     if scale is not None:
        #         scale.value = normalize_scale(scale.value)
