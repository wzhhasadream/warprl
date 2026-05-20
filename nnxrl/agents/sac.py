import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
from copy import deepcopy
from typing import Callable, Protocol
from ..model import (
SquashedTanhGaussianActor, 
Alpha,
EnsembleCritic,
soft_update,
quantile_loss,
project_normalized_parameters)
from ..utils.replaybuffer import Batch, JAXReplayBuffer
from ..utils.normalization import RMS
from ..utils.checkpoint import load_states, save_states


class SACConfig(Protocol):
    seed: int
    total_timesteps: int
    num_envs: int
    learning_starts: int
    num_evals: int
    num_head: int

    gamma: float
    tau: float

    batch_size: int
    grad_step_per_env_step: int
    policy_frequency: int
    target_frequency: int

    normalize_observation: bool

    autotune: bool
    alpha: float
    target_entropy: float
    normalize_parameters: bool


@struct.dataclass
class TrainState:
    actor: SquashedTanhGaussianActor
    critic: EnsembleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: EnsembleCritic


    rms: RMS | None = None
    grad_updates: int = 0  
    alpha: Alpha | None = None
    alpha_opt: nnx.Optimizer | None = None

    @classmethod
    def create(cls,
    actor,
    critic,
    actor_opt,
    critic_opt,
    rms=None,
    alpha=None,
    alpha_opt=None):
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            rms=rms,
            alpha=alpha,
            alpha_opt=alpha_opt,
            grad_updates=0
        )


    def save(self, path: str):
        save_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "grad_updates": self.grad_updates
        })

    def load(self, path: str):
        model_dict = load_states(path, {
            "actor": self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "target_critic": self.target_critic,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "grad_updates": self.grad_updates
        })

        return self.replace(**model_dict)

    @nnx.jit
    def get_action(self, obs):
        if self.rms is not None:
            obs_for_policy, _ = self.rms.normalize(obs, update=False)
        else:
            obs_for_policy = obs
        
        actions = self.actor.get_mean_action(obs_for_policy)
        return actions

    @nnx.jit
    def get_exploration_action(self, obs: jax.Array, key: jax.Array):
        if self.rms is not None:
            obs_for_policy, rms = self.rms.normalize(obs)
        else:
            obs_for_policy = obs
            rms = None
        actions, _ = self.actor.get_action(obs_for_policy, key=key)
        return self.replace(rms=rms), actions

    def make_update_fn(config: SACConfig):
        @nnx.jit
        def jit_update(ts: TrainState, big_batch: Batch, key: jax.Array):
            return update_sac(ts, config, key, big_batch)

        return jit_update



    

def update_critic(ts: TrainState, config: SACConfig, batch: Batch, key: jax.Array):
    alpha_value = ts.alpha() if config.autotune else config.alpha
    def critic_loss_fn(critic: EnsembleCritic, target_critic: EnsembleCritic, actor: SquashedTanhGaussianActor):
        q = critic(batch.observations, batch.actions)
        next_actions, next_log_pi = actor.get_action(
            batch.next_observations, key=key)
        next_q = target_critic(batch.next_observations, next_actions)
        min_next_q = jnp.min(next_q, axis=0)
        target_q = batch.rewards + (1.0 - batch.dones) * config.gamma * (
            min_next_q - alpha_value * next_log_pi
        )
        target_q = jax.lax.stop_gradient(target_q)
        critic_loss = jnp.mean((q - target_q) ** 2)
        info = {
            "training/q_loss": critic_loss,
            "training/q_mean": jnp.mean(q),
        }
        return critic_loss, info

    def dist_critic_loss(critic, target_critic, actor):
            next_actions, next_log_pi = actor.get_action(batch.next_observations, key=key)
            next_q_dist = target_critic(batch.next_observations, next_actions)  # (2, B, num_quantile)
            next_q_dist = next_q_dist.min(0)
            target_q_dist = batch.rewards + config.gamma * (1 - batch.dones) * (next_q_dist - alpha_value * next_log_pi)  # (B, num_quantile)
            q_dist = critic(batch.observations, batch.actions)  # (2, B, num_quantile)

            q_loss = quantile_loss(q_dist, target_q_dist).mean()

            return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q_dist.mean(),
            }

    if  config.num_head > 1:
        loss = dist_critic_loss
    elif config.num_head == 1:
        loss = critic_loss_fn
    

    (_loss, info), grads = nnx.value_and_grad(
        loss, has_aux=True)(ts.critic, ts.target_critic, ts.actor)
    ts.critic_opt.update(grads)
    if config.normalize_parameters:
        project_normalized_parameters(ts.critic)
    return ts, info



def update_actor(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor parameters and return the updated TrainState."""
    alpha_value = train_state.alpha() if config.autotune else config.alpha
    alpha_value = jax.lax.stop_gradient(alpha_value)

    def actor_loss_fn(actor_model: SquashedTanhGaussianActor, critic_model: EnsembleCritic):
        actions, log_pi = actor_model.get_action(batch.observations, key=key)
        if config.num_head == 1:
            q = critic_model(batch.observations, actions)
            min_q = jnp.min(q, axis=0)
        elif config.num_head > 1:
            q_dist = critic_model(batch.observations, actions)
            min_q = jnp.min(q_dist, axis=0).mean(-1, keepdims=True)           
        actor_loss = -jnp.mean(min_q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(train_state.actor, train_state.critic)
    train_state.actor_opt.update(grads)
    if config.normalize_parameters:
        project_normalized_parameters(train_state.actor)
    return train_state, info


def update_alpha(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update entropy temperature (alpha) and return the updated TrainState."""
    log_pi = train_state.actor.get_action(batch.observations, key=key)[1]
    log_pi = jax.lax.stop_gradient(log_pi)

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (log_pi + config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(alpha_loss_fn, has_aux=True)(train_state.alpha)
    train_state.alpha_opt.update(grads)
    return train_state, info


def update_policy(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor (and optionally alpha) once."""
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def _single_update(train_state, key):
        actor_key, alpha_key = jax.random.split(key)
        train_state, actor_info = update_actor(
            train_state, config, batch, actor_key)
        if config.autotune and train_state.alpha is not None:
            train_state, alpha_info = update_alpha(
                train_state, config, batch, alpha_key)
        else:
            alpha_info = {
                "training/alpha_loss": jnp.array(0.0, dtype=jnp.float32),
                "training/alpha_value": jnp.array(config.alpha, dtype=jnp.float32),
            }
        return train_state, {**actor_info, **alpha_info}

    updated_train_state, infos = _single_update(
        train_state, jax.random.split(key, config.policy_frequency))
    info = jax.tree.map(jnp.mean, infos)
    return updated_train_state, info



def update_sac(train_state: TrainState, config: SACConfig, key: jax.Array, big_batch: Batch):
    """(multiple SGD steps per env step)."""

    if train_state.rms is not None and config.normalize_observation:
        stacked_obs = jnp.concatenate(
            [big_batch.observations, big_batch.next_observations],
            axis=0,
        )
        normalized_obs, rms = train_state.rms.normalize(stacked_obs, update=True)
        train_state = train_state.replace(rms=rms)

        batch_size = big_batch.observations.shape[0]
        big_batch = big_batch._replace(
            observations=normalized_obs[:batch_size],
            next_observations=normalized_obs[batch_size:],
        )

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
        big_batch,
    )

    update_keys = jax.random.split(key, config.grad_step_per_env_step)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_sac_minibatch(train_state, sub_batch: Batch, key: jax.Array):
        critic_key, policy_key = jax.random.split(key, 2)
        train_state, critic_info = update_critic(train_state, config, sub_batch, critic_key)
        train_state = train_state.replace(grad_updates=train_state.grad_updates + 1)
        alpha_value = train_state.alpha() if config.autotune else jnp.array(config.alpha, dtype=jnp.float32)

        train_state, policy_info = nnx.cond(
            train_state.grad_updates % config.policy_frequency == 0,
            lambda ts: update_policy(ts, config, sub_batch, policy_key),
            lambda ts: (ts, {
            "training/actor_loss": jnp.array(0.0),
            "training/alpha_loss": jnp.array(0.0),
            "training/alpha_value": alpha_value,
            }),
            train_state,
        )
        nnx.cond(
            train_state.grad_updates % config.target_frequency == 0,
            lambda ts: soft_update(ts.critic, ts.target_critic, config.tau),
            lambda ts: None,
            train_state,
        )

        info = {**critic_info, **policy_info}
        return train_state, info

    updated_train_state, infos = update_sac_minibatch(
        train_state, batches, update_keys)
    info = jax.tree.map(lambda x : x[-1], infos)
    return updated_train_state, info


def sample_and_update_sac(
    train_state: TrainState,
    config: SACConfig,
    key: jax.Array,
    rb: JAXReplayBuffer
) -> tuple[TrainState, dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = rb.sample(
        sample_key,
        config.batch_size * config.grad_step_per_env_step
    )
    ts, info = update_sac(train_state, config, update_key, big_batch)

    return ts, info




