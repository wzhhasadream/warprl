from typing import Callable, Any
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype


def orthogonal(scale: jax.Array = jnp.sqrt(2)):
    return nnx.initializers.orthogonal(scale)


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        rngs: nnx.Rngs,
        layer_norm: bool = False,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish
    ):
        dims = [in_dim] + list(hidden_dims)

        self.layers = [
            nnx.Linear(
                dims[i], dims[i + 1],
                rngs=rngs,
                kernel_init=orthogonal()
            )
            for i in range(len(hidden_dims))
        ]

        self.layer_norm = layer_norm
        self.activation_fn = activation_fn
        if layer_norm:
            self.norms = [
                nnx.LayerNorm(num_features=dims[i + 1], rngs=rngs)
                for i in range(len(hidden_dims))
            ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.norms[i](x)
            x = self.activation_fn(x)

        return x


# Adapted from FlashSac: https: // github.com/Holiday-Robot/FlashSAC/blob/main/flash_rl/agents/flashSAC/layer.py
class FlashSACEmbedder(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs, use_bias: bool = True, compute_dtype: Dtype = jnp.float32):
        self.norm = nnx.BatchNorm(
            num_features=input_dim, rngs=rngs, dtype=compute_dtype)
        self.w = nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(
            1), use_bias=use_bias, dtype=compute_dtype)

    def __call__(self, x: jax.Array, training: bool):
        x = self.norm(x, use_running_average=not training)
        x = self.w(x)
        return x


class FlashSACBlock(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
        expansion: int = 4,
        use_bias: bool = True,
        compute_dtype: Dtype = jnp.float32
    ):
        self.w1 = nnx.Linear(
            hidden_dim,
            hidden_dim * expansion,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_dtype
        )
        self.w2 = nnx.Linear(
            hidden_dim * expansion,
            hidden_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_dtype
        )
        self.norm1 = nnx.BatchNorm(
            num_features=hidden_dim * expansion,
            rngs=rngs,
            dtype=compute_dtype
        )
        self.norm2 = nnx.BatchNorm(
            num_features=hidden_dim,
            rngs=rngs,
            dtype=compute_dtype
        )

    def __call__(self, x: jax.Array, training: bool):
        residual = x
        x = self.w1(x)
        x = self.norm1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.w2(x)
        x = self.norm2(x, use_running_average=not training)
        x = nnx.relu(x)

        return residual + x


class Encoder(nnx.Module):
    def __init__(self,
                 input_dim: int,
                 num_blocks: int,
                 hidden_dim: int,
                 rngs: nnx.Rngs,
                 use_bias: bool = True,
                 compute_type: Dtype = jnp.float32):
        self.embed = FlashSACEmbedder(
            input_dim, hidden_dim, rngs, use_bias, compute_type)
        self.blocks = [FlashSACBlock(hidden_dim, rngs, 4, use_bias, compute_type)
                       for _ in range(num_blocks)]
        self.rms = nnx.RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array, training: bool):
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        x = self.rms(x)

        return x
