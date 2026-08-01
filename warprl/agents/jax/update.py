from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from ...jax_model import (
    Alpha, 
    Network, 
    CategoricalPolicy, 
    QuantilePolicy, 
    RewardNormalizer,
)
from .warpsac_network import FlashSACActor, FlashSACDoubleCritic
from ...utils import (
    select_actor_observations,
)
from ...buffers import Batch


class WarpSACConfig(Protocol):
    seed: int
    env_type: str
    num_envs: int
    total_timesteps: int

    buffer_size: int
    learning_starts: int
    batch_size: int
    grad_step_per_interaction_step: int
    gamma: float
    decay_step: int
    n_step: int
    buffer_device: str

    compute_type: Literal["float32", "bfloat16"]
    policy_lr: float
    q_lr: float
    end_lr: float
    policy_frequency: int
    target_frequency: int
    tau: float
    target_entropy: float

    actor_hidden_dim: int
    actor_num_blocks: int
    critic_hidden_dim: int
    critic_num_blocks: int
    num_q: int
    num_head: int
    use_bias: bool
    dist_type: Literal["quantile", "ce", "scalar"]
    q_agg: Literal["mean", "min"]

    actor_normalize_parameters: bool
    critic_normalize_parameters: bool
    normalize_rewards: bool
    asymmetric_obs: bool


def update_critic(
        actor: Network[FlashSACActor],
        critic: Network[FlashSACDoubleCritic],
        alpha: Network[Alpha],
        target_critic: Network[FlashSACDoubleCritic],
        config: WarpSACConfig,
        batch: Batch,
        key: jax.Array):
    alpha_value = alpha()
    actor_next_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.model.obs_dim)
    next_actions, next_log_pi = actor.model.get_action(
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

    def quantile_loss_fn(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        next_q_dist = target_critic(
            obs_all, actions_all, training=True
        )[:, config.batch_size:, :]
        next_q_dist = next_q_dist.min(0)
        target_q_dist = batch.rewards + batch.discounts * (
            1 - batch.dones
        ) * (next_q_dist - alpha_value * next_log_pi)
        q_dist = critic(obs_all, actions_all, training=True)[
            :, : config.batch_size, :]

        q_loss = critic.critic.policy.loss(q_dist, target_q_dist).mean()

        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q_dist.mean(),
        }

    def ce_loss_fn(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        policy = critic.critic.policy
        next_q_logits = target_critic(
            obs_all, actions_all, training=True
        )[:, config.batch_size:, :]
        q_logits = critic(obs_all, actions_all, training=True)[
            :, : config.batch_size, :]
        next_q_logits = policy.select_min_logits(next_q_logits)
        target_bins = (
            batch.rewards
            + batch.discounts
            * (1.0 - batch.dones)
            * (policy.bins - alpha_value * next_log_pi)
        )
        target_probs = policy.target_probs(next_q_logits, target_bins)
        ce_loss = policy.loss(q_logits, target_probs).mean()

        return ce_loss, {
            "training/q_loss": ce_loss,
            "training/q_mean": policy.q_values(q_logits).mean(),
        }

    policy = critic.model.critic.policy
    if isinstance(policy, CategoricalPolicy):
        loss_fn = ce_loss_fn
    elif isinstance(policy, QuantilePolicy):
        loss_fn = quantile_loss_fn
    else:
        loss_fn = mse_loss

    (_loss, info), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(critic.model, target_critic.model)
    critic.grad_step(grads)
    if config.critic_normalize_parameters:
        critic.project_param()
    return info


def update_alpha(
    alpha: Network[Alpha],
    entropy: jax.Array,
    config: WarpSACConfig
) -> dict[str, jax.Array]:
    """Update entropy temperature."""

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (- entropy +
                      config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(
        alpha_loss_fn, has_aux=True)(alpha.model)
    alpha.grad_step(grads)
    return info


def update_actor(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update actor parameters."""
    alpha_value = alpha()
    action_key, entropy_key = jax.random.split(key, 2)
    alpha_value = jax.lax.stop_gradient(alpha_value)
    actor_observations = select_actor_observations(
        batch.observations, config.asymmetric_obs, actor.model.obs_dim)
    next_actor_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.model.obs_dim)
    actor_obs_all = jnp.concat(
        [actor_observations, next_actor_observations], axis=0)

    def actor_loss_fn(actor: FlashSACActor, critic: FlashSACDoubleCritic):
        actions_all, log_pi_all = actor.get_action(
            actor_obs_all, key=action_key, training=True)
        actions = actions_all[: config.batch_size, ]
        log_pi = log_pi_all[: config.batch_size, ]
        q = critic.q_values(batch.observations, actions, training=False)
        q = getattr(jnp, config.q_agg)(q, axis=0)
        actor_loss = -jnp.mean(q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss, "training/entropy": -log_pi.mean()}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(actor.model, critic.model)
    actor.grad_step(grads)
    if config.actor_normalize_parameters:
        actor.project_param()
    return info


def update_policy(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
    key: jax.Array,
):
    actor_info = update_actor(critic, actor, alpha, config, batch, key)
    entropy = actor_info["training/entropy"]
    alpha_info = update_alpha(alpha, entropy, config)
    return {**actor_info, **alpha_info}


def make_update_warpsac(config: WarpSACConfig):
    """Create a jitted WarpSAC update function with config captured by closure."""

    @nnx.jit
    def update_warpsac(
        critic: Network[FlashSACDoubleCritic],
        actor: Network[FlashSACActor],
        alpha: Network[Alpha],
        target_critic: Network[FlashSACDoubleCritic],
        reward_normalizer: RewardNormalizer | None,
        critic_grad_updates: jax.Array,
        key: jax.Array,
        big_batch: Batch,
    ):
        """Run multiple SGD steps per environment step."""
        if config.asymmetric_obs:
            assert actor.model.obs_dim != critic.model.critic.obs_dim

        if config.normalize_rewards and reward_normalizer is not None:
            normalized_rewards = reward_normalizer.normalize(big_batch.rewards)
            big_batch = big_batch._replace(rewards=normalized_rewards)


        update_keys = jax.random.split(
            key, config.grad_step_per_interaction_step)

        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
        def update_minibatch(carry, sub_batch: Batch, key: jax.Array):
            (
                critic,
                actor,
                alpha,
                target_critic,
                critic_grad_updates,
            ) = carry
            policy_key, critic_key = jax.random.split(key, 2)
            alpha_value = alpha()
            do_policy_update = critic_grad_updates % config.policy_frequency == 0
            policy_info = nnx.cond(
                do_policy_update,
                lambda critic, actor, alpha: update_policy(
                    critic,
                    actor,
                    alpha,
                    config,
                    sub_batch,
                    policy_key,
                ),
                lambda critic, actor, alpha: {
                    "training/actor_loss": jnp.array(0.0),
                    "training/alpha_loss": jnp.array(0.0),
                    "training/alpha_value": alpha_value,
                    "training/entropy": jnp.array(0.0)
                },
                critic,
                actor,
                alpha,
            )
            critic_info = update_critic(
                actor,
                critic,
                alpha,
                target_critic,
                config,
                sub_batch,
                critic_key,
            )
            next_critic_grad_updates = critic_grad_updates + 1
            nnx.cond(
                next_critic_grad_updates % config.target_frequency == 0,
                lambda target_critic: target_critic.soft_update(),
                lambda target_critic: None,
                target_critic,
            )

            info = {
                **critic_info,
                **policy_info,
                "training/policy_updated": do_policy_update.astype(jnp.float32),
            }
            next_carry = (
                critic,
                actor,
                alpha,
                target_critic,
                next_critic_grad_updates,
            )
            return next_carry, info

        init_carry = (
            critic,
            actor,
            alpha,
            target_critic,
            critic_grad_updates,
        )
        (
            critic,
            actor,
            alpha,
            target_critic,
            critic_grad_updates,
        ), infos = update_minibatch(init_carry, big_batch, update_keys)
        policy_mask = infos["training/policy_updated"]
        policy_update_count = policy_mask.sum()
        safe_count = jnp.maximum(policy_update_count, 1.0)

        def policy_mean(values: jax.Array) -> jax.Array:
            mask_shape = (policy_mask.shape[0],) + (1,) * (values.ndim - 1)
            weights = policy_mask.reshape(mask_shape)
            mean = jnp.sum(values * weights, axis=0) / safe_count
            return jnp.where(policy_update_count > 0, mean, jnp.nan)

        critic_info = jax.tree.map(
            lambda values: values.mean(axis=0),
            {
                "training/q_loss": infos["training/q_loss"],
                "training/q_mean": infos["training/q_mean"],
            },
        )
        policy_info = jax.tree.map(
            policy_mean,
            {
                "training/actor_loss": infos["training/actor_loss"],
                "training/alpha_loss": infos["training/alpha_loss"],
                "training/entropy": infos["training/entropy"],
            },
        )
        info = {
            **critic_info,
            **policy_info,
            "training/alpha_value": alpha(),
            "training/policy_update_count": policy_update_count,
        }
        return (
            critic_grad_updates,
            info,
        )

    return update_warpsac
