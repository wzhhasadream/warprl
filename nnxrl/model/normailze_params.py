import jax.numpy as jnp
from flax import nnx
import jax

def normalize_linear_kernel(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    if kernel.ndim == 2:   # 单个模型
        axis = 0
    elif kernel.ndim == 3:  # 集成模型
        axis = 1
    else:
        raise ValueError(f"Unsupported Linear kernel shape: {kernel.shape}")

    norm = jnp.linalg.norm(kernel, ord=2, axis=axis, keepdims=True)
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
    seen: set[int] = set()
    for _, value in nnx.iter_graph(module):
        obj_id = id(value)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(value, nnx.Linear):
            value.kernel.value = normalize_linear_kernel(value.kernel.value)

        else:
            pass