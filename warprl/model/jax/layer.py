import math
from typing import Callable, Sequence
from flax import nnx
import jax
import jax.numpy as jnp
from flax.typing import Dtype

def orthogonal(scale: jax.Array = 1):
    return nnx.initializers.orthogonal(scale)


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
        layer_norm: bool = True,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32
    ):
        dims = [in_dim] + list(hidden_dims)

        self.layers = [
            nnx.Linear(
                dims[i], dims[i + 1],
                rngs=rngs,
                kernel_init=orthogonal(1),
                use_bias=use_bias,
                dtype=compute_type
            )
            for i in range(len(hidden_dims))
        ]

        self.layer_norm = layer_norm
        self.activation_fn = activation_fn
        if layer_norm:
            self.norms = [
                nnx.LayerNorm(num_features=dims[i + 1], rngs=rngs, dtype=compute_type)
                for i in range(len(hidden_dims))
            ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.norms[i](x)
            x = self.activation_fn(x)

        return x


class CNN(nnx.Module):
    """Channels-last convolutional encoder for pixel observations."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        rngs: nnx.Rngs,
        features: Sequence[int] = (32, 32, 32),
        strides: Sequence[int] = (2, 1, 1),
        kernel_size: int = 3,
        latent_dim: int = 50,
        layer_norm: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        if len(features) != len(strides):
            raise ValueError("features and strides must have the same length")

        self.compute_type = compute_type
        self.convs = []
        in_features = input_shape[-1]
        for out_features, stride in zip(features, strides):
            self.convs.append(
                nnx.Conv(
                    in_features,
                    out_features,
                    kernel_size,
                    strides=stride,
                    padding="VALID",
                    rngs=rngs,
                    kernel_init=orthogonal(1),
                    dtype=compute_type,
                )
            )
            in_features = out_features

        dummy = jnp.zeros((1, *input_shape), dtype=compute_type)
        for conv in self.convs:
            dummy = jax.nn.relu(conv(dummy))
        flatten_dim = math.prod(dummy.shape[1:])
        self.projection = nnx.Linear(
            flatten_dim,
            latent_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            dtype=compute_type,
        )
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nnx.LayerNorm(
                num_features=latent_dim,
                rngs=rngs,
                dtype=compute_type,
            )

    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(self.compute_type) / 255.0
        for conv in self.convs:
            x = jax.nn.relu(conv(x))
        x = x.reshape((x.shape[0], -1))
        x = self.projection(x)
        if self.layer_norm:
            x = self.norm(x)
        return jnp.tanh(x)



