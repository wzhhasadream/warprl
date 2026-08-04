from typing import Callable, Sequence
from flax import nnx
import jax



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
        use_bias: bool = True
    ):
        dims = [in_dim] + list(hidden_dims)

        self.layers = [
            nnx.Linear(
                dims[i], dims[i + 1],
                rngs=rngs,
                kernel_init=orthogonal(),
                use_bias=use_bias
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