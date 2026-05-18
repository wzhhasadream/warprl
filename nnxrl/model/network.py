from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from .time_embedding import TimeEmbedding
from .layer import MLP, SimBaEncoder, UnitLinear, orthogonal
from .policy import (
    SquashedTanhGaussianPolicy,
    TanhDeterministicPolicy,
    GaussianPolicy,
    flattened_dim,
    squash_log_std_tanh,
)


def soft_update(online: nnx.Module, target_net: nnx.Module, tau: float = 0.005) -> None:
    """Soft-update network parameters."""
    online_params = nnx.state(online, nnx.Param)
    target_params = nnx.state(target_net, nnx.Param)
    new_params = jax.tree.map(
        lambda online_p, target_p: tau *
        online_p + (1 - tau) * target_p,
        online_params,
        target_params,
    )
    nnx.update(target_net, new_params)


def freeze_module_params(module):
    graphdef, params, rest = nnx.split(module, nnx.Param, ...)
    frozen_params = jax.tree.map(jax.lax.stop_gradient, params)
    return nnx.merge(graphdef, frozen_params, rest)




class QNetwork(nnx.Module):
    def __init__(self, obs_dim: int | tuple[int, ...],
                 action_dim: int,
                 rngs: nnx.Rngs,
                 hidden_dim=(256, 256),
                 activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
                 layer_norm: bool = False,
                 simba_encoder: bool = False,
                 unit_projection: bool = False,
                 num_head: int = 1     # for distributional q
                 ):
        self.obs_dim = flattened_dim(obs_dim)
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.mlp = SimBaEncoder(
                self.obs_dim + action_dim, 512, 2, rngs, unit_projection=unit_projection)
            out_dim = 512
        else:
            self.mlp = MLP(self.obs_dim + action_dim, list(hidden_dim),
                           rngs=rngs, layer_norm=layer_norm, activation_fn=activation_fn,
                           unit_projection=unit_projection)
            out_dim = hidden_dim[-1]
        self.out = linear_cls(
            out_dim, num_head, rngs=rngs, kernel_init=orthogonal())

    def __call__(self, x, a):
        h = jnp.concatenate([x, a], axis=1)
        h = self.mlp(h)
        return self.out(h)


class VNetwork(nnx.Module):
    def __init__(self, obs_dim: int | tuple[int, ...],
                 rngs: nnx.Rngs,
                 hidden_dim=(256, 256),
                 activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
                 layer_norm: bool = False,
                 simba_encoder: bool = False,
                 unit_projection: bool = False,
                 num_head: int = 1    # for distributional q
                 ):
        self.obs_dim = flattened_dim(obs_dim)
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.mlp = SimBaEncoder(
                self.obs_dim, 512, 2, rngs, unit_projection=unit_projection)
            out_dim = 512
        else:
            self.mlp = MLP(self.obs_dim, list(hidden_dim),
                           rngs=rngs, layer_norm=layer_norm, activation_fn=activation_fn,
                           unit_projection=unit_projection)
            out_dim = hidden_dim[-1]
        self.out = linear_cls(
            out_dim, num_head, rngs=rngs, kernel_init=orthogonal())

    def __call__(self, x):
        h = self.mlp(x)
        return self.out(h)


class EnsembleCritic(nnx.Module):
    """Ensemble Q network using NNX API."""
    @nnx.vmap(in_axes=(0, None, None, 0, None, None, None, None, None, None))
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: Sequence[int] = (256, 256),
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
        layer_norm: bool = False,
        simba_encoder: bool = False,
        unit_projection: bool = False,
        num_head: int = 1
    ):

        self.critic = QNetwork(
            obs_dim,
            action_dim,
            rngs,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            layer_norm=layer_norm,
            simba_encoder=simba_encoder,
            unit_projection=unit_projection,
            num_head=num_head,
        )

    @nnx.vmap(in_axes=(0, None, None))
    def __call__(self, observations: Any, actions: jax.Array) -> jax.Array:
        q = self.critic(observations, actions)
        return q    # (num_q, B, n_head)





class TanhDetActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: Sequence[int] = (256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
        layer_norm: bool = False,
        simba_encoder: bool = False,
        unit_projection: bool = False
    ):
        self.action_high = action_high
        self.action_low = action_low
        self.action_scale = (action_high - action_low) / 2
        self.action_dim = action_dim

        self.obs_dim = flattened_dim(obs_dim)
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.encoder = SimBaEncoder(self.obs_dim, 128, 1, rngs, unit_projection=unit_projection)
            out_dim = 128
        else:
            self.encoder = MLP(self.obs_dim,
                               hidden_dim, rngs, layer_norm, activation_fn=activation_fn,
                               unit_projection=unit_projection)
            out_dim = hidden_dim[-1]

        self.actor_head = linear_cls(
            out_dim, action_dim, rngs=rngs, kernel_init=orthogonal())

        self.policy = TanhDeterministicPolicy(
            self.action_low, self.action_high)

    def get_action(self, x: jax.Array) -> jax.Array:
        """Returns an action in [action_low, action_high].
        """
        pre_tanh = self.actor_head(self.encoder(x))
        return self.policy.action(pre_tanh)


class SquashedTanhGaussianActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: Sequence[int] = (256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
        layer_norm: bool = False,
        simba_encoder: bool = False,
        unit_projection: bool = False,
        log_std_min: jax.Array = -5,
        log_std_max: jax.Array = 2,
        squash_log_std: bool = True
    ):

        self.action_high = action_high
        self.action_low = action_low
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

        self.obs_dim = flattened_dim(obs_dim)
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.encoder = SimBaEncoder(self.obs_dim, 128, 1, rngs, unit_projection=unit_projection)
            out_dim = 128
        else:
            self.encoder = MLP(self.obs_dim, hidden_dim, rngs, layer_norm,
                               activation_fn=activation_fn,
                               unit_projection=unit_projection)
            out_dim = hidden_dim[-1]
        self.fc_mean = linear_cls(
            out_dim, action_dim, rngs=rngs, kernel_init=orthogonal())
        self.fc_logstd = linear_cls(
            out_dim, action_dim, rngs=rngs, kernel_init=orthogonal())

        self.policy = SquashedTanhGaussianPolicy(
            self.action_low, self.action_high, log_std_min, log_std_max, squash_log_std)

    def __call__(self, x: Any) -> Any:
        x = self.encoder(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return self.policy.dist(mean, log_std)

    def get_action(self, x: Any, *, key: jax.Array | None = None, actions: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        action_distribution = self(x)
        if actions is not None:
            eps = jnp.asarray(1e-6, dtype=jnp.asarray(actions).dtype)
            actions = jnp.clip(actions, self.action_low +
                               eps, self.action_high - eps)
        else:
            actions = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(actions)

        # (batch_size, action_dim), (batch_size, 1)
        return actions, log_prob[:, None]

    def get_mean_action(self, x: Any) -> jax.Array:
        h = self.encoder(x)
        mean = self.fc_mean(h)

        return jnp.tanh(mean) * self.action_scale + self.action_bias


class GaussianActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: Sequence[int] = (256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
        layer_norm: bool = False,
        simba_encoder: bool = False,
        unit_projection: bool = False,
        log_std_min: jax.Array = -20,
        log_std_max: jax.Array = 2,
        squash_log_std: bool = True,
        shared_std: bool = False
    ):

        self.action_high = action_high
        self.action_low = action_low

        self.obs_dim = flattened_dim(obs_dim)
        self.shared_std = shared_std
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.encoder = SimBaEncoder(self.obs_dim, 128, 1, rngs, unit_projection=unit_projection)
            out_dim = 128
        else:
            self.encoder = MLP(self.obs_dim, hidden_dim, rngs, layer_norm,
                               activation_fn=activation_fn,
                               unit_projection=unit_projection)
            out_dim = hidden_dim[-1]
        self.fc_mean = linear_cls(
            out_dim, action_dim, rngs=rngs, kernel_init=orthogonal())
        if not self.shared_std:
            self.fc_logstd = linear_cls(
                out_dim, action_dim, rngs=rngs, kernel_init=orthogonal())
        else:
            self.log_std = nnx.Param(jnp.zeros((action_dim, )))

        self.policy = GaussianPolicy(log_std_min, log_std_max, squash_log_std)

    def __call__(self, x: Any) -> Any:
        x = self.encoder(x)
        mean = self.fc_mean(x)
        if not self.shared_std:
            log_std = self.fc_logstd(x)
        else:
            log_std = jnp.broadcast_to(self.log_std.value, mean.shape)
        return self.policy.dist(mean, log_std)

    def get_action(self, x: Any, *, key: jax.Array | None = None, actions: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_distribution = self(x)
        if actions is not None:
            actions = actions
        else:
            actions = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()

        # (batch_size, action_dim), (batch_size, 1), (batch_size, 1)
        return actions, log_prob[:, None], entropy[:, None]

    def get_mean_action(self, x: Any) -> jax.Array:
        h = self.encoder(x)
        mean = self.fc_mean(h)

        return jnp.clip(mean, self.action_low, self.action_high)


class FlowActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int | tuple[int, ...],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: Sequence[int] = (256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.mish,
        layer_norm: bool = False,
        simba_encoder: bool = False,
        unit_projection: bool = False,
        emb_dim: int = 64,
        euler_steps: int = 10
    ):

        self.action_high = action_high
        self.action_low = action_low
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2
        self.euler_steps = euler_steps
        self.action_dim = action_dim

        self.obs_dim = flattened_dim(obs_dim)
        linear_cls = UnitLinear if unit_projection else nnx.Linear
        if simba_encoder:
            self.encoder = SimBaEncoder(
                self.obs_dim + self.action_dim + emb_dim, 128, 1, rngs,
                unit_projection=unit_projection)
            out_dim = 128
        else:
            self.encoder = MLP(self.obs_dim + self.action_dim + emb_dim, hidden_dim, rngs, layer_norm,
                               activation_fn=activation_fn,
                               unit_projection=unit_projection)
            out_dim = hidden_dim[-1]
        self.time_embed = TimeEmbedding(emb_dim=emb_dim, rngs=rngs)
        self.head = linear_cls(out_dim, action_dim,
                               rngs=rngs, kernel_init=orthogonal())

    def get_v(self, obs: jax.Array, x_t: jax.Array, t: jax.Array) -> jax.Array:
        t_emb = self.time_embed(t)
        inputs = jnp.concatenate([obs, x_t, t_emb], axis=1)
        x = self.encoder(inputs)
        return self.head(x)

    def get_action(self, x: jax.Array, *,  key: jax.Array | None = None, noise: jax.Array | None = None) -> jax.Array:
        batch_size = x.shape[0]
        if key is not None:
            a_i = jax.random.normal(key, shape=(batch_size, self.action_dim))
        elif noise is not None:
            a_i = noise
        else:
            raise KeyError("key or noise must be provided")

        for i in range(self.euler_steps):
            t = jnp.full((batch_size, 1), (i + 0.5) / self.euler_steps)
            dt = 1.0 / self.euler_steps
            a_i = a_i + dt * self.get_v(x, a_i, t)

        actions = jnp.clip(a_i, self.action_low, self.action_high)

        return actions

    def get_mean_action(self, obs):
        noise = jnp.zeros((obs.shape[0], self.action_dim), dtype=jnp.float32)

        return self.get_action(obs, noise=noise)


class ActorCritic(nnx.Module):
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic


class Alpha(nnx.Module):
    def __init__(self, init_value: float = 0.0):
        self.log_alpha = nnx.Param(jnp.asarray(init_value))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha.value)


class SquashedAlpha(nnx.Module):
    def __init__(self, init_value: float = 0.0, log_std_min: float = -10.0, log_std_max: float = 7.5):
        self.log_alpha = nnx.Param(jnp.asarray(init_value))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def __call__(self) -> jax.Array:
        log_alpha = self.log_alpha.value
        log_alpha = squash_log_std_tanh(
            log_alpha, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return jnp.exp(log_alpha)
