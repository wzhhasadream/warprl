from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .network import Actor, ActorCritic, Critic
from .utils import adapt_lr, diagonal_gaussian_kl
from ....buffers.on_policy.jax_buffer import JaxBuffer
from ....buffers.on_policy.types import RolloutBatch
from ...config.ppo import PPOConfig
from ....model.jax import Network, clip_grads
from ....utils import select_actor_observations

def update_ppo_minibatch(
    agent: Network[ActorCritic[Actor, Critic]],
    batch: RolloutBatch,
    cfg: PPOConfig,
) -> dict[str, jax.Array]:
    def ppo_loss(
        model: ActorCritic[Actor, Critic],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        actor, critic = model.actor, model.critic
        actor_observations = select_actor_observations(
            batch.observations,
            cfg.asymmetric_obs,
            actor.obs_dim,
        )
        _, new_log_probs, entropy, new_actions_mean, new_actions_std = actor.get_action(
            actor_observations,
            key=None,
            update_rms=False,
            actions=batch.actions,
        )
        values = critic(batch.observations, update_rms=False)

        log_ratio = new_log_probs - batch.old_log_probs
        ratio = jnp.exp(log_ratio)

        if cfg.algo == "ppo":

            pg_loss_unclipped = ratio * batch.advantages
            pg_loss_clipped = jnp.clip(
                ratio,
                1.0 - cfg.clip_coef,
                1.0 + cfg.clip_coef,
            ) * batch.advantages
            pg_loss = -jnp.mean(jnp.minimum(pg_loss_unclipped, pg_loss_clipped))

        elif cfg.algo == "spo":

            pg_objective = (
                batch.advantages * ratio
                - jnp.abs(batch.advantages) * jnp.square(ratio - 1.0) / (2.0 * cfg.clip_coef)
            )
            pg_loss = -jnp.mean(pg_objective)

        if cfg.clip_value:
            value_pred_clipped = batch.values + jnp.clip(
                values - batch.values,
                -cfg.clip_coef,
                cfg.clip_coef,
            )
            value_loss_unclipped = jnp.square(values - batch.returns)
            value_loss_clipped = jnp.square(value_pred_clipped - batch.returns)
            value_loss = 0.5 * jnp.maximum(
                value_loss_unclipped,
                value_loss_clipped,
            ).mean()
        else:
            value_loss = 0.5 * jnp.square(values - batch.returns).mean()

        entropy_loss = jnp.mean(entropy)
        loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

        kl = diagonal_gaussian_kl(
            new_actions_mean,
            new_actions_std,
            batch.actions_mean,
            batch.actions_std,
        ).mean()
        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > cfg.clip_coef)

        return loss, {
            "training/loss": loss,
            "training/pg_loss": pg_loss,
            "training/value_loss": value_loss,
            "training/entropy": entropy_loss,
            "training/kl": kl,
            "training/clipfrac": clipfrac,
        }

    (_loss, info), grads = nnx.value_and_grad(ppo_loss, has_aux=True)(agent.model)
    if cfg.algo == "ppo":
        lr = agent.opt.opt_state.hyperparams["learning_rate"].value
        lr = adapt_lr(
            lr,
            info["training/kl"]
        )
        agent.opt.opt_state[1].hyperparams["learning_rate"].value = lr
    grads["actor"] = clip_grads(grads["actor"], cfg.max_grad_norm)
    grads["critic"] = clip_grads(grads["critic"], cfg.max_grad_norm)
    agent.grad_step(grads)
    return agent, info


def make_update_ppo(cfg: PPOConfig):
    @nnx.jit
    def update_ppo(
        agent: Network[ActorCritic[Actor, Critic]],
        buffer: JaxBuffer,
        last_obs: jax.Array,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        last_value = agent.model.critic(last_obs, update_rms=False)
        buffer = buffer.compute_returns_and_advantages(
            last_value,
            cfg.gamma,
            cfg.gae_lambda
        )
        if cfg.normalize_advantages:
            buffer = buffer.normalize_advantages()
        batches = buffer.sample(key, cfg.num_mini_batches, cfg.num_epochs)

        def scan_minibatch(
            agent: Network[ActorCritic[Actor, Critic]],
            batch: RolloutBatch,
        ):
            return update_ppo_minibatch(agent, batch, cfg)

        scan_fn = nnx.scan(scan_minibatch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        _, info = scan_fn(agent, batches)
        agent.model.sync_rms()
        return jax.tree.map(lambda x : jnp.mean(x), info)

    return update_ppo
