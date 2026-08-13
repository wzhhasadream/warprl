import jax
from flax import nnx
from typing import Callable, TypeVar, Generic, Sequence
from ....model.jax import MLP, OnPolicyRMS
from ....model.jax.policy import GaussianPolicy
import jax.numpy as jnp
from ....model.jax.layer import orthogonal
from flax.typing import Dtype

class Actor(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.elu,
        init_std: float = 1,
        compute_type: Dtype = jnp.float32,
    ) -> None:

        self.obs_dim = obs_dim
        self.obs_norm = OnPolicyRMS(obs_dim)
        self.encoder = MLP(
            obs_dim,
            hidden_dims,
            rngs,
            activation_fn=activation,
            compute_type=compute_type,
        )
        self.log_std = nnx.Param(jnp.ones((action_dim, ), dtype=jnp.float32) * jnp.log(init_std))
        self.mean_head = nnx.Linear(
            hidden_dims[-1],
            action_dim,
            rngs=rngs,
            kernel_init=orthogonal(1),
            dtype=compute_type,
        )
        self.policy: GaussianPolicy = GaussianPolicy()


    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        x = self.obs_norm(obs, update_rms)
        x = self.encoder(x)
        return x


    def sync_rms(self) -> None:
        self.obs_norm.sync()

    def get_action(
        self,
        obs: jax.Array,
        key: jax.Array | None = None,
        update_rms: bool = True,
        actions: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x = self(obs, update_rms)
        mean, log_std = self.mean_head(x).astype(jnp.float32), self.log_std.value[None, :]
        dist = self.policy.dist(mean, log_std)
        if actions is None:
            actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions).reshape(-1, 1)
        entropy = dist.entropy().reshape(-1, 1)
        std = jnp.exp(self.policy.transform_log_std(log_std))
        return actions, log_probs, entropy, mean, jnp.broadcast_to(std, mean.shape)

    def get_mean_action(self, obs: jax.Array) -> jax.Array:
        x = self(obs, False)
        return self.mean_head(x).astype(jnp.float32)


class Critic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
        activation: Callable[[jax.Array], jax.Array]=jax.nn.elu,
        compute_type: Dtype = jnp.float32
    ) -> None:
        self.obs_norm = OnPolicyRMS(obs_dim)
        self.encoder = MLP(obs_dim, hidden_dims, rngs,
                           activation_fn=activation, compute_type=compute_type)
        self.value_head = nnx.Linear(hidden_dims[-1], 1, rngs=rngs, dtype=compute_type)

    def __call__(self, obs: jax.Array, update_rms: bool = False) -> jax.Array:
        x = self.obs_norm(obs, update_rms)
        x = self.encoder(x)
        return self.value_head(x).astype(jnp.float32)


    def sync_rms(self) -> None:
        self.obs_norm.sync()


ActorT = TypeVar("ActorT", bound=Actor)
CriticT = TypeVar("CriticT", bound=Critic)



class ActorCritic(Generic[ActorT, CriticT], nnx.Module):
    def __init__(self, 
                actor: ActorT,
                critic: CriticT
    ) -> None:
        self.actor = actor
        self.critic = critic

    def get_mean_action(self, obs: jax.Array):
        return self.actor.get_mean_action(obs)

    def sync_rms(self) -> None:
        self.actor.sync_rms()
        self.critic.sync_rms()
