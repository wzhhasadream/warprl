import jax
import jax.numpy as jnp
from flax import nnx

from .layer import orthogonal, Encoder
from .policy import (
    SquashedTanhGaussianPolicy,
    action_scale_bias,
    flattened_dim,
    TanhDeterministicPolicy
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
        block_type: str = 'flash'
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.action_low = jnp.asarray(action_low)
        self.action_high = jnp.asarray(action_high)
        self.action_scale, self.action_bias = action_scale_bias(
            self.action_low, self.action_high
        )

        self.encoder = Encoder(self.obs_dim, num_blocks, hidden_dim, rngs=rngs, use_bias=use_bias, block_type=block_type)
        self.fc_mean = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
        )
        self.fc_log_std = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
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
        x = self.encoder(observations, training=training)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        return mean, log_std

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
        x = self.encoder(observations, training=False)
        mean = self.fc_mean(x)
        return jnp.tanh(mean) * self.action_scale + self.action_bias


    def get_mean_std(self, observations: jax.Array):
        mean, raw_log_std = self(observations, False)
        log_std = self.policy.transform_log_std(raw_log_std)
        return mean, log_std




class FlashSACTanhDetActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        action_low: jax.Array = -1,
        action_high: jax.Array = 1,
        use_bias: bool = True
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.action_low = jnp.asarray(action_low)
        self.action_high = jnp.asarray(action_high)
        self.action_scale, self.action_bias = action_scale_bias(
            self.action_low, self.action_high
        )

        self.encoder = Encoder(self.obs_dim, num_blocks, hidden_dim, rngs=rngs, use_bias=use_bias)
        self.head = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
        )
        self.policy = TanhDeterministicPolicy(
            action_low=self.action_low,
            action_high=self.action_high
        )


    def __call__(
        self,
        observations: jax.Array,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        x = self.encoder(observations, training=training)
        raw_actions = self.head(x)
        return raw_actions

    def get_action(
        self,
        observations: jax.Array,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        raw_actions = self(observations, training=training)
        actions = self.policy.action(raw_actions)
        return actions


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
        block_type: str = 'flash'
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.encoder = Encoder(
            self.obs_dim + self.action_dim, num_blocks, hidden_dim, rngs=rngs, use_bias=use_bias, block_type=block_type)
        self.out = nnx.Linear(
            hidden_dim,
            num_head,
            rngs=rngs,
            kernel_init=orthogonal(1),
        )

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        x = jnp.concatenate((observations, actions), axis=-1)
        x = self.encoder(x, training=training)
        return self.out(x)



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
        block_type: str = 'flash'
    ):
        self.critic = FlashSACQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_head=num_head,
            use_bias=use_bias
        )

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        return self.critic(observations, actions, training=training)

