from typing import Callable, Any
import jax
import jax.numpy as jnp
from flax import nnx


def orthogonal(scale: jax.Array = jnp.sqrt(2)):
    return nnx.initializers.orthogonal(scale)



def normalize_linear_kernel(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Normalize each output column of an NNX Linear kernel to unit L2 norm."""
    norm = jnp.linalg.norm(kernel, axis=0, keepdims=True)
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


def project_normalized_parameters(module: Any) -> None:
    seen: set[int] = set()
    for _, value in nnx.iter_graph(module):
        obj_id = id(value)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(value, nnx.Linear):
            value.kernel.value = normalize_linear_kernel(value.kernel.value)

        elif isinstance(value, (nnx.LayerNorm, nnx.BatchNorm)):
            if hasattr(value, "scale") and hasattr(value, "bias"):
                scale = getattr(value, "scale", None)
                bias = getattr(value, "bias", None)
                if scale is not None and bias is not None:
                    scale_v, bias_v = normalize_scale_bias(
                        scale.value, bias.value
                    )
                    scale.value = scale_v
                    bias.value = bias_v

        elif hasattr(nnx, "RMSNorm") and isinstance(value, nnx.RMSNorm):
            scale = getattr(value, "scale", None)
            if scale is not None:
                scale.value = normalize_scale(scale.value)


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


# Adapted from SimBa: https://github.com/SonyResearch/simba/blob/master/scale_rl/networks/layers.py
class ResidualBlock(nnx.Module):
    """
    Residual block used in SimBa architecture.

    Architecture:
    - LayerNorm
    - Linear(hidden_dim -> hidden_dim * 4) + ReLU
    - Linear(hidden_dim * 4 -> hidden_dim)
    - Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.hidden_dim = hidden_dim

        # Layer normalization
        self.layer_norm = nnx.BatchNorm(
            num_features=hidden_dim,
            rngs=rngs
        )

        # Feedforward network with 4x expansion
        self.dense1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim * 4,
            kernel_init=orthogonal(1),
            rngs=rngs
        )

        self.dense2 = nnx.Linear(
            in_features=hidden_dim * 4,
            out_features=hidden_dim,
            kernel_init=orthogonal(1),  
            rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        """Forward pass with residual connection."""
        # Store residual connection
        residual = x

        # Pre-norm residual block
        x = self.layer_norm(x, use_running_average=not training)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)

        # Add residual connection
        return residual + x



# Adapted from SimBa:https://github.com/SonyResearch/simba/blob/master/scale_rl/agents/sac/sac_network.py#L33
class SimBaEncoder(nnx.Module):
    """
    SimBa encoder residual block architectures.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        num_blocks: Number of residual blocks (default: 1)
        rngs: Random number generators for initialization

    Returns:
        jnp.ndarray: Encoded features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int = 1,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.first_bn = nnx.BatchNorm(
            input_dim,
            rngs=rngs
        )
        self.input_projection = nnx.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            kernel_init=orthogonal(1),
            rngs=rngs
        )

        # Stack residual blocks
        self.residual_blocks = [
            ResidualBlock(hidden_dim, rngs=rngs)
            for _ in range(num_blocks)
        ]

        # Final layer norm
        self.final_bn = nnx.BatchNorm(
            num_features=hidden_dim,
            rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            encoded: Encoded features of shape (batch_size, hidden_dim)
        """
        # Initial projection
        x = self.first_bn(x, use_running_average=not training)
        x = self.input_projection(x)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x, training)

        # Final layer normalization
        x = self.final_bn(x, use_running_average=not training)

        return x


# Adapted from FlashSac: https: // github.com/Holiday-Robot/FlashSAC/blob/main/flash_rl/agents/flashSAC/layer.py
class FlashSACEmbedder(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.norm = nnx.BatchNorm(num_features=input_dim, rngs=rngs)
        self.w = nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(1))

    def __call__(self, x: jax.Array, training: bool):
        x = self.norm(x, use_running_average=not training)
        x = self.w(x)
        return x


class FlashSACBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4):
        self.w1 = nnx.Linear(hidden_dim, hidden_dim * expansion, rngs=rngs, kernel_init=orthogonal(1))
        self.w2 = nnx.Linear(hidden_dim * expansion, hidden_dim, rngs=rngs, kernel_init=orthogonal(1))
        self.norm1 = nnx.BatchNorm(
            num_features=hidden_dim * expansion, rngs=rngs)
        self.norm2 = nnx.BatchNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool):
        residual = x
        x = self.w1(x)
        x = self.norm1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.w2(x)
        x = self.norm2(x, use_running_average=not training)
        x = nnx.relu(x)
        return x + residual


class FlashSACEncoder(nnx.Module):
    def __init__(self, 
                input_dim: int,
                num_blocks: int,
                hidden_dim: int,
                rngs: nnx.Rngs):
        self.embed = FlashSACEmbedder(input_dim, hidden_dim, rngs)
        self.blocks = [FlashSACBlock(hidden_dim, rngs) for _ in range(num_blocks)]
        self.rms = nnx.RMSNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool):
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        x = self.rms(x)
        
        return x

        


# Adapted from ProcGen starter kit: https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class CNNResidualBlock(nnx.Module):
    """A simple residual block with two 3x3 convolutions.

    This block applies:
      ReLU -> Conv(3x3) -> ReLU -> Conv(3x3) -> residual add

    """

    def __init__(self, channels: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv0 = nnx.Conv(in_features=channels, out_features=channels,
                              kernel_size=3, padding=1, kernel_init=orthogonal(), rngs=rngs)
        self.conv1 = nnx.Conv(in_features=channels, out_features=channels,
                              kernel_size=3, padding=1, kernel_init=orthogonal(), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        inputs = x
        x = jax.nn.relu(x)
        x = self.conv0(x)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nnx.Module):
    """A convolutional feature extractor stage: Conv + downsampling + residual blocks.

    Structure:
      - 3x3 convolution to `out_channels`
      - 3x3 max-pooling with stride 2 (spatial downsample by ~2)
      - Two residual blocks

    Notes:
        - Input shape should be (H, W, C) for NHWC tensors.
        - Output shape is (ceil(H/2), ceil(W/2), out_channels) with "SAME" pooling.
    """

    def __init__(self, input_shape: tuple[int, int, int], out_channels: int, rngs: nnx.Rngs):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        # input_shape is expected to be (H, W, C) for NHWC inputs.
        self.conv = nnx.Conv(
            in_features=self._input_shape[2],
            out_features=self._out_channels,
            kernel_size=3,
            padding=1,
            rngs=rngs
        )
        self.res_block0 = CNNResidualBlock(self._out_channels, rngs=rngs)
        self.res_block1 = CNNResidualBlock(self._out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)

        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.res_block0(x)
        x = self.res_block1(x)

        return x

    def get_output_shape(self):
        h,w,c = self._input_shape
        return ((h + 1) // 2, (w + 1) // 2, self._out_channels)
