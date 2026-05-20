import trace
import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
from copy import deepcopy
from typing import Protocol
from ..model import (
Alpha,
FlashSACActor,
FlashSACDoubleCritic,
soft_update,
quantile_loss,
project_normalized_parameters)
from ..utils.replaybuffer import Batch
from ..utils.checkpoint import load_states, save_states


class RainbowSACConfig(Protocol):
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

    alpha: float
    target_entropy: float
    normalize_parameters: bool


@struct.dataclass
class TrainState:
    actor: FlashSACActor
    critic: FlashSACDoubleCritic
    alpha: Alpha
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: FlashSACDoubleCritic
    alpha_opt: nnx.Optimizer
    grad_updates: int = 0

    @classmethod
    def create(cls,
               actor,
               critic,
               actor_opt,
               critic_opt,
               alpha,
               alpha_opt):
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha=alpha,
            alpha_opt=alpha_opt,
            grad_updates=0
        )

    def save(self, path: str):
        save_states(path, {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
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
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "grad_updates": self.grad_updates
        })

        return self.replace(**model_dict)

    @nnx.jit
    def get_action(self, obs):
        actions = self.actor.get_mean_action(obs)
        return actions

    @nnx.jit
    def get_exploration_action(self, obs: jax.Array, key: jax.Array):
        actions, _ = self.actor.get_action(obs, key=key, training=False)
        return  actions

    def make_update_fn(self, config: RainbowSACConfig):
        @nnx.jit
        def jit_update(ts: TrainState, big_batch: Batch, key: jax.Array):
            return update_rainbowsac(ts, config, key, big_batch)

        return jit_update



def update_critic(ts: TrainState, config: RainbowSACConfig, batch: Batch, key: jax.Array):
    alpha_value = ts.alpha() 
    next_actions, next_log_pi = ts.actor.get_action(
            batch.next_observations, key=key, training=False)
    obs_all = jnp.concat([batch.observations, batch.next_observations], axis=0)
    actions_all = jnp.concat([batch.actions, next_actions], axis=0)

    def critic_loss_fn(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        q = critic(obs_all, actions_all, training=True)[:, : config.batch_size, :]
        next_q = target_critic(obs_all, actions_all, training=True)[
            :, config.batch_size:, :]
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

    def dist_critic_loss(critic, target_critic):
            next_q_dist = target_critic(
                obs_all, actions_all, training=True
            )[:, config.batch_size:, :]
            next_q_dist = next_q_dist.min(0)
            target_q_dist = batch.rewards + config.gamma * (1 - batch.dones) * (next_q_dist - alpha_value * next_log_pi)  # (B, num_quantile)
            q_dist = critic(obs_all, actions_all, training=True)[
                :, : config.batch_size, :]

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
        loss, has_aux=True)(ts.critic, ts.target_critic)
    ts.critic_opt.update(grads)
    if config.normalize_parameters:
        project_normalized_parameters(ts.critic)
    return ts, info



def update_actor(
    train_state: TrainState,
    config: RainbowSACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor parameters and return the updated TrainState."""
    alpha_value = train_state.alpha()
    alpha_value = jax.lax.stop_gradient(alpha_value)

    def actor_loss_fn(actor: FlashSACActor, critic: FlashSACDoubleCritic):
        actions, log_pi = actor.get_action(
            batch.observations, key=key, training=True)
        if config.num_head == 1:
            q = critic(batch.observations, actions, training=False)
            min_q = jnp.min(q, axis=0)
        elif config.num_head > 1:
            q_dist = critic(batch.observations, actions, training=False)
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
    config: RainbowSACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update entropy temperature (alpha) and return the updated TrainState."""
    log_pi = train_state.actor.get_action(batch.observations, key=key, training=False)[1]
    log_pi = jax.lax.stop_gradient(log_pi)

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (log_pi + config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(
        alpha_loss_fn, has_aux=True)(train_state.alpha)
    train_state.alpha_opt.update(grads)
    return train_state, info


def update_policy(
    train_state: TrainState,
    config: RainbowSACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor (and optionally alpha) once."""
    actor_key, alpha_key = jax.random.split(key)
    train_state, actor_info = update_actor(
        train_state, config, batch, actor_key)
    train_state, alpha_info = update_alpha(
            train_state, config, batch, alpha_key)
    return train_state, {**actor_info, **alpha_info}




def update_rainbowsac(train_state: TrainState, config: RainbowSACConfig, key: jax.Array, big_batch: Batch):
    """(multiple SGD steps per env step)."""

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
        big_batch,
    )

    update_keys = jax.random.split(key, config.grad_step_per_env_step)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_sac_minibatch(train_state, sub_batch: Batch, key: jax.Array):
        critic_key, policy_key = jax.random.split(key, 2)
        train_state, critic_info = update_critic(
            train_state, config, sub_batch, critic_key)
        train_state = train_state.replace(
            grad_updates=train_state.grad_updates + 1)
        alpha_value = train_state.alpha() 

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
    info = jax.tree.map(lambda x: x[-1], infos)
    return updated_train_state, info
