from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from nnxrl.utils.normalization import RewardNormalizer
from ..model import (
    Alpha,
    FlashSACActor,
    FlashSACDoubleCritic,
    soft_update,
    project_param)
from ..utils import (
    select_actor_observations,
    quantile_loss,
    select_min_q_logits,
    categorical_q_values,
    make_bin_values,
    categorical_projection,
    categorical_ce_loss)
from nnxrl.buffers import Batch


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
    asymmetric_obs: bool
    normalize_rewards: bool
    loss_type: str
    n_step: int


def update_critic(
        actor: FlashSACActor,
        critic: FlashSACDoubleCritic,
        alpha: Alpha,
        critic_opt: nnx.Optimizer,
        target_critic: FlashSACDoubleCritic,
        config: RainbowSACConfig,
        batch: Batch,
        key: jax.Array):
    alpha_value = alpha()
    actor_next_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.obs_dim)
    next_actions, next_log_pi = actor.get_action(
        actor_next_observations, key=key, training=False)
    obs_all = jnp.concatenate(
        [batch.observations, batch.next_observations], axis=0)
    actions_all = jnp.concatenate([batch.actions, next_actions], axis=0)

    def mse_loss(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        q = critic(obs_all, actions_all, training=True)[
            :, : config.batch_size, :]
        next_q = target_critic(obs_all, actions_all, training=True)[
            :, config.batch_size:, :]
        min_next_q = jnp.min(next_q, axis=0)
        target_q = batch.rewards + (1.0 - batch.dones) * batch.discounts * (
            min_next_q - alpha_value * next_log_pi
        )
        target_q = jax.lax.stop_gradient(target_q)
        critic_loss = jnp.mean((q - target_q) ** 2)
        info = {
            "training/q_loss": critic_loss,
            "training/q_mean": jnp.mean(q),
        }
        return critic_loss, info

    def quantile_loss_fn(critic, target_critic):
        next_q_dist = target_critic(
            obs_all, actions_all, training=True
        )[:, config.batch_size:, :]
        next_q_dist = next_q_dist.min(0)
        target_q_dist = batch.rewards + batch.discounts * (
            1 - batch.dones
        ) * (next_q_dist - alpha_value * next_log_pi)
        q_dist = critic(obs_all, actions_all, training=True)[
            :, : config.batch_size, :]

        q_loss = quantile_loss(q_dist, target_q_dist).mean()

        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q_dist.mean(),
        }

    def ce_loss_fn(critic, target_critic):
        next_q_logits = target_critic(
            obs_all, actions_all, training=True
        )[:, config.batch_size:, :]
        q_logits = critic(obs_all, actions_all, training=True)[
            :, : config.batch_size, :]
        next_q_logits = select_min_q_logits(next_q_logits)
        bins = make_bin_values(
            config.num_head
        )   # (num_head , )
        target_bins = (
            batch.rewards
            + batch.discounts
            * (1.0 - batch.dones)
            * (bins[None, :] - alpha_value * next_log_pi)
        )
        target_probs = categorical_projection(next_q_logits, target_bins)
        ce_loss = categorical_ce_loss(q_logits, target_probs).mean()

        return ce_loss, {
            "training/q_loss": ce_loss,
            "training/q_mean": categorical_q_values(q_logits).mean(),
        }

    if config.num_head > 1:
        if config.loss_type == "quantile_loss":
            loss = quantile_loss_fn
        elif config.loss_type == "ce_loss":
            loss = ce_loss_fn
    elif config.num_head == 1:
        loss = mse_loss

    (_loss, info), grads = nnx.value_and_grad(
        loss, has_aux=True)(critic, target_critic)
    critic_opt.update(grads)
    if config.normalize_parameters:
        project_param(critic)
    return info


def update_alpha(
    alpha: Alpha,
    alpha_opt: nnx.Optimizer,
    entropy: jax.Array,
    config: RainbowSACConfig
) -> dict[str, jax.Array]:
    """Update entropy temperature."""

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (- entropy +
                      config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(
        alpha_loss_fn, has_aux=True)(alpha)
    alpha_opt.update(grads)
    return info


def update_actor(
    critic: FlashSACDoubleCritic,
    actor: FlashSACActor,
    actor_opt: nnx.Optimizer,
    alpha: Alpha,
    config: RainbowSACConfig,
    batch: Batch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update actor parameters."""
    alpha_value = alpha()
    action_key, entropy_key = jax.random.split(key, 2)
    alpha_value = jax.lax.stop_gradient(alpha_value)
    actor_observations = select_actor_observations(
        batch.observations, config.asymmetric_obs, actor.obs_dim)
    next_actor_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.obs_dim)
    actor_obs_all = jnp.concat(
        [actor_observations, next_actor_observations], axis=0)

    def actor_loss_fn(actor: FlashSACActor, critic: FlashSACDoubleCritic):
        actions_all, log_pi_all = actor.get_action(
            actor_obs_all, key=action_key, training=True)
        actions = actions_all[: config.batch_size, ]
        log_pi = log_pi_all[: config.batch_size, ]
        if config.num_head == 1:
            q = critic(batch.observations, actions, training=False)
            min_q = jnp.min(q, axis=0)
        elif config.num_head > 1:
            if config.loss_type == "quantile_loss":
                q_dist = critic(batch.observations, actions, training=False)
                min_q = jnp.min(q_dist, axis=0).mean(-1, keepdims=True)
            elif config.loss_type == "ce_loss":
                q_logits = critic(batch.observations, actions, training=False)
                min_q = categorical_q_values(q_logits).min(0)
        actor_loss = -jnp.mean(min_q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss, "training/entropy": -log_pi.mean()}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(actor, critic)
    actor_opt.update(grads)
    if config.normalize_parameters:
        project_param(actor)
    return info


def update_policy(
    critic: FlashSACDoubleCritic,
    actor: FlashSACActor,
    actor_opt: nnx.Optimizer,
    alpha: Alpha,
    alpha_opt: nnx.Optimizer,
    config: RainbowSACConfig,
    batch: Batch,
    key: jax.Array,
):
    actor_info = update_actor(
        critic, actor, actor_opt, alpha, config, batch, key)
    entropy = actor_info["training/entropy"]
    alpha_info = update_alpha(
        alpha, alpha_opt, entropy, config)
    return {**actor_info, **alpha_info}



def make_update_rainbowsac(config: RainbowSACConfig):
    """Create a jitted RainbowSAC update function with config captured by closure."""

    @nnx.jit
    def update_rainbowsac(
        critic: FlashSACDoubleCritic,
        actor: FlashSACActor,
        actor_opt: nnx.Optimizer,
        alpha: Alpha,
        alpha_opt: nnx.Optimizer,
        critic_opt: nnx.Optimizer,
        target_critic: FlashSACDoubleCritic,
        critic_grad_updates: jax.Array,
        reward_normalizer: RewardNormalizer | None,
        key: jax.Array,
        big_batch: Batch,
    ):
        """Run multiple SGD steps per environment step."""
        if config.asymmetric_obs:
            assert actor.obs_dim != critic.critic.obs_dim

        if config.normalize_rewards and reward_normalizer is not None:
            normalized_rewards = reward_normalizer.normalize(big_batch.rewards)
            big_batch = big_batch._replace(rewards=normalized_rewards)

        batches = jax.tree.map(
            lambda x: x.reshape(
                config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
            big_batch,
        )

        update_keys = jax.random.split(key, config.grad_step_per_env_step)

        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
        def update_minibatch(carry, sub_batch: Batch, key: jax.Array):
            (
                critic,
                actor,
                actor_opt,
                alpha,
                alpha_opt,
                critic_opt,
                target_critic,
                critic_grad_updates,
            ) = carry
            policy_key, critic_key = jax.random.split(key, 2)
            alpha_value = alpha()

            policy_info = nnx.cond(
                critic_grad_updates % config.policy_frequency == 0,
                lambda critic, actor, actor_opt, alpha, alpha_opt: update_policy(
                    critic,
                    actor,
                    actor_opt,
                    alpha,
                    alpha_opt,
                    config,
                    sub_batch,
                    policy_key,
                ),
                lambda critic, actor, actor_opt, alpha, alpha_opt: {
                    "training/actor_loss": jnp.array(0.0),
                    "training/alpha_loss": jnp.array(0.0),
                    "training/alpha_value": alpha_value,
                    "training/entropy": jnp.array(0.0)
                },
                critic,
                actor,
                actor_opt,
                alpha,
                alpha_opt,
            )
            critic_info = update_critic(
                actor,
                critic,
                alpha,
                critic_opt,
                target_critic,
                config,
                sub_batch,
                critic_key,
            )
            next_critic_grad_updates = critic_grad_updates + 1
            nnx.cond(
                next_critic_grad_updates % config.target_frequency == 0,
                lambda critic, target_critic: soft_update(
                    critic, target_critic, config.tau),
                lambda critic, target_critic: None,
                critic, target_critic,
            )

            info = {**critic_info, **policy_info}
            next_carry = (
                critic,
                actor,
                actor_opt,
                alpha,
                alpha_opt,
                critic_opt,
                target_critic,
                next_critic_grad_updates,
            )
            return next_carry, info

        init_carry = (
            critic,
            actor,
            actor_opt,
            alpha,
            alpha_opt,
            critic_opt,
            target_critic,
            critic_grad_updates,
        )
        (
            critic,
            actor,
            actor_opt,
            alpha,
            alpha_opt,
            critic_opt,
            target_critic,
            critic_grad_updates,
        ), infos = update_minibatch(init_carry, batches, update_keys)
        info = jax.tree.map(lambda x: x[-1], infos)
        return (
            critic_grad_updates,
            info,
        )

    return update_rainbowsac
