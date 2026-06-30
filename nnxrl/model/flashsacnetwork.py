import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from .layer import orthogonal, Encoder
from .policy import (
    SquashedTanhGaussianPolicy,
    action_scale_bias,
    flattened_dim,
)


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
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        action_low = jnp.asarray(action_low)
        action_high = jnp.asarray(action_high)
        action_scale, action_bias = action_scale_bias(action_low, action_high)
        self.action_low = nnx.Variable(action_low)
        self.action_high = nnx.Variable(action_high)
        self.action_scale = nnx.Variable(action_scale)
        self.action_bias = nnx.Variable(action_bias)
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
            action_low=self.action_low,
            action_high=self.action_high,
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
        return action, log_prob[:, None]

    def get_mean_action(self, observations: jax.Array) -> jax.Array:
        observations = observations.astype(self.compute_type)
        x = self.encoder(observations, training=False)
        mean = self.fc_mean(x)
        return (jnp.tanh(mean) * self.action_scale + self.action_bias).astype(jnp.float32)


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
        compute_type: Dtype = jnp.float32
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
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



class FlashSACDoubleCritic(nnx.Module):
    """Ensembled scalar critic for FlashSAC-style training."""

    @nnx.vmap(in_axes=(0, None, None, 0, None, None, None, None, None), out_axes=0)
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1,
        use_bias: bool = True,
        compute_type: Dtype = jnp.float32
    ):
        self.critic = FlashSACQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_head=num_head,
            use_bias=use_bias,
            compute_type=compute_type
        )

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        return self.critic(observations, actions, training=training)
