import jax
import jax.numpy as jnp
from math import prod
from typing import Literal
from flax import nnx
from flax.typing import Dtype

from ...model.jax.dist_head import CategoricalPolicy, QuantilePolicy
from ...model.jax.policy import SquashedTanhGaussianPolicy


def orthogonal(scale: jax.Array = 1):
    return nnx.initializers.orthogonal(scale)


class FlashSACEmbedder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        compute_dtype: Dtype = jnp.float32,
    ):
        self.norm = nnx.BatchNorm(
            num_features=input_dim,
            rngs=rngs,
            dtype=compute_dtype,
        )
        self.w = nnx.Linear(
            input_dim,
            hidden_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_dtype,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.norm(x, use_running_average=not training)
        return self.w(x)


class FlashSACBlock(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
        expansion: int = 4,
        use_bias: bool = True,
        compute_dtype: Dtype = jnp.float32,
    ):
        self.w1 = nnx.Linear(
            hidden_dim,
            hidden_dim * expansion,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_dtype,
        )
        self.w2 = nnx.Linear(
            hidden_dim * expansion,
            hidden_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            use_bias=use_bias,
            dtype=compute_dtype,
        )
        self.norm1 = nnx.BatchNorm(
            num_features=hidden_dim * expansion,
            rngs=rngs,
            dtype=compute_dtype,
        )
        self.norm2 = nnx.BatchNorm(
            num_features=hidden_dim,
            rngs=rngs,
            dtype=compute_dtype,
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = self.w2(x)
        x = self.norm2(x, use_running_average=not training)
        return residual + nnx.relu(x)


class Encoder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
    ):
        self.embed = FlashSACEmbedder(
            input_dim,
            hidden_dim,
            rngs,
            use_bias,
            compute_type,
        )
        self.blocks = [
            FlashSACBlock(hidden_dim, rngs, 4, use_bias, compute_type)
            for _ in range(num_blocks)
        ]
        self.rms = nnx.RMSNorm(hidden_dim, rngs=rngs, dtype=compute_type)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        x = self.embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.rms(x)


def _flattened_dim(observation_dim: int | tuple[int, ...]) -> int:
    if isinstance(observation_dim, int):
        return observation_dim
    return prod(observation_dim)


class FlashSACActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        action_low: jax.Array = -1,
        action_high: jax.Array = 1,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = True,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32
    ):
        self.obs_dim = _flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.compute_type = compute_type

        self.encoder = Encoder(self.obs_dim, num_blocks, hidden_dim, rngs=rngs, use_bias=use_bias, compute_type=compute_type)
        self.fc_mean = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            dtype=compute_type
        )
        self.fc_log_std = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            dtype=compute_type
        )
        self.policy = SquashedTanhGaussianPolicy(
            action_low=action_low,
            action_high=action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            squash_log_std=squash_log_std,
        )


    def __call__(
        self,
        observations: jax.Array,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        observations = observations.astype(self.compute_type)
        x = self.encoder(observations, training=training)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        return mean.astype(jnp.float32), log_std.astype(jnp.float32)

    def get_action(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self(observations, training=training)
        action, log_prob = self.policy.sample_and_log_prob(mean, log_std, key)
        return action, log_prob

    def get_mean_action(self, observations: jax.Array) -> jax.Array:
        observations = observations.astype(self.compute_type)
        x = self.encoder(observations, training=False)
        mean = self.fc_mean(x)
        return self.policy.mean_action(mean).astype(jnp.float32)


    def get_mean_std(self, observations: jax.Array):
        mean, raw_log_std = self(observations, False)
        log_std = self.policy.transform_log_std(raw_log_std)
        return mean, log_std






class FlashSACQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
        dist_type: Literal["scalar", "ce", "quantile"] = "ce",
    ):
        if num_head < 1:
            raise ValueError(f"num_head must be positive, got {num_head}")
        self.obs_dim = _flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.dist_type = "scalar" if num_head == 1 else dist_type
        self.encoder = Encoder(
            self.obs_dim + self.action_dim, num_blocks, hidden_dim, rngs=rngs, use_bias=use_bias, compute_type=compute_type)
        self.out = nnx.Linear(
            hidden_dim,
            num_head,
            rngs=rngs,
            kernel_init=orthogonal(1),
            dtype=compute_type
        )
        self.compute_type = compute_type
        if self.dist_type == "scalar":
            if num_head != 1:
                raise ValueError("scalar critics require num_head == 1")
            self.policy = None
        elif self.dist_type == "ce":
            self.policy = CategoricalPolicy(num_head, -5, 5)
        elif self.dist_type == "quantile":
            self.policy = QuantilePolicy(num_head)
        else:
            raise ValueError(f"Unsupported dist_type: {dist_type}")

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        x = jnp.concatenate((observations, actions), axis=-1)
        x = x.astype(self.compute_type)
        x = self.encoder(x, training=training)
        return self.out(x).astype(jnp.float32)

    def q_values(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        values = self(observations, actions, training=training)
        return values if self.policy is None else self.policy.q_values(values)



class FlashSACDoubleCritic(nnx.Module):
    """Ensembled scalar or distributional critic for FlashSAC-style training."""

    @nnx.vmap(in_axes=(0, None, None, 0, None, None, None, None, None, None), out_axes=0)
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32,
        dist_type: Literal["scalar", "ce", "quantile"] = "ce",
    ):
        self.critic = FlashSACQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_head=num_head,
            use_bias=use_bias,
            compute_type=compute_type,
            dist_type=dist_type,
        )

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        return self.critic(observations, actions, training=training)

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def q_values(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        return self.critic.q_values(observations, actions, training=training)
