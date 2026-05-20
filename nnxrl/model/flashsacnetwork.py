import jax
import jax.numpy as jnp
from flax import nnx

from .layer import FlashSACBlock, FlashSACEmbedder, orthogonal
from .policy import (
    SquashedTanhGaussianPolicy,
    action_scale_bias,
    flattened_dim
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
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.action_low = jnp.asarray(action_low)
        self.action_high = jnp.asarray(action_high)
        self.action_scale, self.action_bias = action_scale_bias(
            self.action_low, self.action_high
        )

        self.embedder = FlashSACEmbedder(
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.blocks = [
            FlashSACBlock(hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_blocks)
        ]
        self.post_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.fc_mean = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1.0),
        )
        self.fc_log_std = nnx.Linear(
            hidden_dim,
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1.0),
        )
        self.policy = SquashedTanhGaussianPolicy(
            action_low=self.action_low,
            action_high=self.action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            squash_log_std=squash_log_std,
        )

    def _encode(self, observations: jax.Array, training: bool) -> jax.Array:
        x = self.embedder(observations, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.post_norm(x)

    def __call__(
        self,
        observations: jax.Array,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        x = self._encode(observations, training=training)
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
        x = self._encode(observations, training=False)
        mean = self.fc_mean(x)
        return jnp.tanh(mean) * self.action_scale + self.action_bias


class FlashSACQNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1
    ):
        self.obs_dim = flattened_dim(obs_dim)
        self.action_dim = action_dim
        self.embedder = FlashSACEmbedder(
            input_dim=self.obs_dim + action_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.blocks = [
            FlashSACBlock(hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_blocks)
        ]
        self.post_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.out = nnx.Linear(
            hidden_dim,
            num_head,
            rngs=rngs,
            kernel_init=orthogonal(1.0),
        )

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        x = jnp.concatenate((observations, actions), axis=-1)
        x = self.embedder(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.post_norm(x)
        return self.out(x)


class FlashSACDoubleCritic(nnx.Module):
    """Ensembled scalar critic for FlashSAC-style training."""

    @nnx.vmap(in_axes=(0, None, None, 0, None, None, None), out_axes=0)
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_head: int = 1
    ):
        self.critic = FlashSACQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rngs=rngs,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_head=num_head
        )

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        training: bool = True,
    ) -> jax.Array:
        return self.critic(observations, actions, training=training)
