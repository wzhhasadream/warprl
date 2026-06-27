from copy import deepcopy
from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct

from ..model import (
    FlashSACDoubleCritic,
    FlashSACTanhDetActor,
    project_param,
    soft_update,
)
from ..utils import (
    Batch,
    GPUReplayBuffer,
    JAXReplayBuffer,
    RewardNormalizer,
    categorical_ce_loss,
    categorical_projection,
    categorical_q_values,
    load_states,
    make_bin_values,
    quantile_loss,
    save_states,
    select_actor_observations,
    select_min_q_logits,
)


class RainbowTD3Config(Protocol):
    seed: int
    total_timesteps: int
    num_envs: int
    learning_starts: int
    num_head: int

    gamma: float
    tau: float

    batch_size: int
    grad_step_per_env_step: int
    policy_frequency: int

    exploration_noise: float
    policy_noise: float
    noise_clip: float

    normalize_parameters: bool
    loss_type: str


@struct.dataclass
class TrainState:
    actor: FlashSACTanhDetActor
    critic: FlashSACDoubleCritic
    target_actor: FlashSACTanhDetActor
    target_critic: FlashSACDoubleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    asymmetric_obs: bool = struct.field(pytree_node=False, default=False)
    grad_updates: int = 0
    reward_normalizer: RewardNormalizer | None = None

    @classmethod
    def create(
        cls,
        actor: FlashSACTanhDetActor,
        critic: FlashSACDoubleCritic,
        actor_opt: nnx.Optimizer,
        critic_opt: nnx.Optimizer,
        reward_normalizer: RewardNormalizer | None = None,
    ) -> "TrainState":
        target_actor = deepcopy(actor)
        target_critic = deepcopy(critic)
        asymmetric_obs = actor.obs_dim != critic.critic.obs_dim
        return cls(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            asymmetric_obs=asymmetric_obs,
            grad_updates=0,
            reward_normalizer=reward_normalizer,
        )

    def save(self, path: str) -> None:
        save_states(
            path,
            {
                "actor": self.actor,
                "critic": self.critic,
                "target_actor": self.target_actor,
                "target_critic": self.target_critic,
                "actor_opt": self.actor_opt,
                "critic_opt": self.critic_opt,
                "grad_updates": self.grad_updates,
                "reward_normalizer": self.reward_normalizer,
            },
        )

    def load(self, path: str) -> "TrainState":
        model_dict = load_states(
            path,
            {
                "actor": self.actor,
                "critic": self.critic,
                "target_actor": self.target_actor,
                "target_critic": self.target_critic,
                "actor_opt": self.actor_opt,
                "critic_opt": self.critic_opt,
                "grad_updates": self.grad_updates,
                "reward_normalizer": self.reward_normalizer,
            },
        )
        return self.replace(**model_dict)

    def get_action(self, obs: jax.Array) -> jax.Array:
        return _get_action(self.actor, obs, self.asymmetric_obs)

    def get_exploration_action(
        self,
        obs: jax.Array,
        exploration_noise: float,
        key: jax.Array,
    ) -> tuple["TrainState", jax.Array]:
        actions = _get_exploration_action(
            self.actor,
            obs,
            self.asymmetric_obs,
            exploration_noise,
            key,
        )
        return self, actions

    def update_reward_normalizer(self, raw_rewards: jax.Array, done: jax.Array) -> "TrainState":
        if self.reward_normalizer is None:
            return self
        return self.replace(
            reward_normalizer=self.reward_normalizer.update(raw_rewards, done)
        )

    def make_update_fn(self, config: RainbowTD3Config):
        @nnx.jit(donate_argnums=0)
        def jit_update(ts: TrainState, big_batch: Batch, key: jax.Array):
            return update_rainbowtd3(ts, config, key, big_batch)

        return jit_update

    def make_train_step(self, config: RainbowTD3Config):
        @nnx.jit(donate_argnums=(0, 1))
        def train_step(ts: TrainState, rb: GPUReplayBuffer, transition: dict, key: jax.Array):
            if ts.reward_normalizer is not None and _normalize_rewards(config):
                ts = ts.update_reward_normalizer(
                    transition["rewards"],
                    jnp.logical_or(transition["terminations"], transition["truncations"]),
                )
            rb = rb.add(
                transition["observations"],
                transition["actions"],
                transition["rewards"],
                transition["next_observations"],
                transition["terminations"],
            )
            zeros = jnp.array(0.0)
            zero_info = {
                "training/q_loss": zeros,
                "training/q_mean": zeros,
                "training/actor_loss": zeros,
            }
            sample_key, update_key = jax.random.split(key)
            ts, info = nnx.cond(
                rb.size >= config.learning_starts,
                lambda ts, rb: update_rainbowtd3(
                    ts,
                    config,
                    update_key,
                    rb.sample(sample_key, config.batch_size * config.grad_step_per_env_step),
                ),
                lambda ts, rb: (ts, zero_info),
                ts,
                rb,
            )
            return ts, rb, info

        return train_step


def _loss_type(config: RainbowTD3Config) -> str:
    return getattr(config, "loss_type", "ce_loss")


def _asymmetric_obs(config: RainbowTD3Config) -> bool:
    return getattr(config, "asymmetric_obs", False)


def _normalize_rewards(config: RainbowTD3Config) -> bool:
    return getattr(config, "normalize_rewards", True)


@nnx.jit(static_argnames=("asymmetric_obs",))
def _get_action(
    actor: FlashSACTanhDetActor,
    obs: jax.Array,
    asymmetric_obs: bool,
) -> jax.Array:
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)
    return actor.get_action(obs, training=False)


@nnx.jit(static_argnames=("asymmetric_obs",))
def _get_exploration_action(
    actor: FlashSACTanhDetActor,
    obs: jax.Array,
    asymmetric_obs: bool,
    exploration_noise: float,
    key: jax.Array,
) -> jax.Array:
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)
    actions = actor.get_action(obs, training=False)
    noise = jax.random.normal(key, actions.shape) * exploration_noise
    clipped_actions = actions + noise * actor.action_scale
    return jnp.clip(clipped_actions, actor.action_low, actor.action_high)


def _target_policy_smoothing(
    target_actor: FlashSACTanhDetActor,
    next_actor_observations: jax.Array,
    *,
    key: jax.Array,
    policy_noise: float,
    noise_clip: float,
) -> jax.Array:
    target_actions = target_actor.get_action(next_actor_observations, training=False)
    noise = jax.random.normal(key, target_actions.shape) * policy_noise
    clipped_noise = jnp.clip(noise, -noise_clip, noise_clip) * target_actor.action_scale
    return jnp.clip(
        target_actions + clipped_noise,
        target_actor.action_low,
        target_actor.action_high,
    )


def update_critic(
    actor: FlashSACTanhDetActor,
    critic: FlashSACDoubleCritic,
    critic_opt: nnx.Optimizer,
    target_actor: FlashSACTanhDetActor,
    target_critic: FlashSACDoubleCritic,
    config: RainbowTD3Config,
    batch: Batch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    actor_next_observations = select_actor_observations(
        batch.next_observations,
        _asymmetric_obs(config),
        actor.obs_dim,
    )
    next_actions = _target_policy_smoothing(
        target_actor,
        actor_next_observations,
        key=key,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )
    obs_all = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    actions_all = jnp.concatenate([batch.actions, next_actions], axis=0)

    def mse_loss(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        q = critic(obs_all, actions_all, training=True)[:, : config.batch_size, :]
        next_q = target_critic(obs_all, actions_all, training=True)[
            :, config.batch_size:, :
        ]
        min_next_q = jnp.min(next_q, axis=0)
        target_q = batch.rewards + config.gamma * (1.0 - batch.dones) * min_next_q
        target_q = jax.lax.stop_gradient(target_q)
        q_loss = jnp.mean((q - target_q) ** 2)
        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": jnp.mean(q),
        }

    def quantile_loss_fn(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        next_q_dist = target_critic(obs_all, actions_all, training=True)[
            :, config.batch_size:, :
        ]
        next_q_dist = jnp.min(next_q_dist, axis=0)
        target_q_dist = batch.rewards + config.gamma * (1.0 - batch.dones) * next_q_dist
        target_q_dist = jax.lax.stop_gradient(target_q_dist)
        q_dist = critic(obs_all, actions_all, training=True)[:, : config.batch_size, :]
        q_loss = quantile_loss(q_dist, target_q_dist).mean()
        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q_dist.mean(),
        }

    def ce_loss_fn(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
        next_q_logits = target_critic(obs_all, actions_all, training=True)[
            :, config.batch_size:, :
        ]
        q_logits = critic(obs_all, actions_all, training=True)[:, : config.batch_size, :]
        next_q_logits = select_min_q_logits(next_q_logits)
        bins = jnp.asarray(make_bin_values(config.num_head), dtype=next_q_logits.dtype)
        target_bins = batch.rewards + config.gamma * (1.0 - batch.dones) * bins[None, :]
        target_probs = categorical_projection(next_q_logits, target_bins)
        q_loss = categorical_ce_loss(q_logits, target_probs).mean()
        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": categorical_q_values(q_logits).mean(),
        }

    if config.num_head == 1:
        loss = mse_loss
    elif _loss_type(config) == "ce_loss":
        loss = ce_loss_fn
    else:
        loss = quantile_loss_fn

    (_loss, info), grads = nnx.value_and_grad(loss, has_aux=True)(
        critic,
        target_critic,
    )
    critic_opt.update(grads)
    if config.normalize_parameters:
        project_param(critic)
    return info


def update_actor(
    critic: FlashSACDoubleCritic,
    actor: FlashSACTanhDetActor,
    actor_opt: nnx.Optimizer,
    config: RainbowTD3Config,
    batch: Batch,
) -> dict[str, jax.Array]:
    actor_observations = select_actor_observations(
        batch.observations,
        _asymmetric_obs(config),
        actor.obs_dim,
    )

    def actor_loss_fn(actor: FlashSACTanhDetActor, critic: FlashSACDoubleCritic):
        actions = actor.get_action(actor_observations, training=True)
        if config.num_head == 1:
            q = critic(batch.observations, actions, training=False)
            min_q = jnp.min(q, axis=0)
        elif _loss_type(config) == "ce_loss":
            q_logits = critic(batch.observations, actions, training=False)
            min_q_logits = select_min_q_logits(q_logits)
            min_q = categorical_q_values(min_q_logits)
        else:
            q_dist = critic(batch.observations, actions, training=False)
            min_q = jnp.min(q_dist, axis=0).mean(-1, keepdims=True)
        actor_loss = -jnp.mean(min_q)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn,
        argnums=0,
        has_aux=True,
    )(actor, critic)
    actor_opt.update(grads)
    if config.normalize_parameters:
        project_param(actor)
    return info


def update_policy(ts: TrainState, config: RainbowTD3Config, batch: Batch) -> tuple[TrainState, dict[str, jax.Array]]:
    policy_info = update_actor(ts.critic, ts.actor, ts.actor_opt, config, batch)
    soft_update(ts.actor, ts.target_actor, config.tau)
    soft_update(ts.critic, ts.target_critic, config.tau)
    return ts, policy_info


def update_rainbowtd3(
    ts: TrainState,
    config: RainbowTD3Config,
    key: jax.Array,
    big_batch: Batch,
) -> tuple[TrainState, dict[str, jax.Array]]:
    if _asymmetric_obs(config):
        assert ts.actor.obs_dim != ts.critic.critic.obs_dim

    if _normalize_rewards(config) and ts.reward_normalizer is not None:
        normalized_rewards = ts.reward_normalizer.normalize(big_batch.rewards)
        big_batch = big_batch._replace(rewards=normalized_rewards)

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step,
            config.batch_size,
            *x.shape[1:],
        ),
        big_batch,
    )
    update_keys = jax.random.split(key, config.grad_step_per_env_step)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_minibatch(ts: TrainState, sub_batch: Batch, key: jax.Array):
        critic_info = update_critic(
            ts.actor,
            ts.critic,
            ts.critic_opt,
            ts.target_actor,
            ts.target_critic,
            config,
            sub_batch,
            key,
        )
        ts = ts.replace(grad_updates=ts.grad_updates + 1)
        ts, policy_info = nnx.cond(
            ts.grad_updates % config.policy_frequency == 0,
            lambda ts: update_policy(ts, config, sub_batch),
            lambda ts: (ts, {"training/actor_loss": jnp.array(0.0)}),
            ts,
        )
        info = {**critic_info, **policy_info}
        return ts, info

    ts, infos = update_minibatch(ts, batches, update_keys)
    info = jax.tree.map(lambda x: x[-1], infos)
    return ts, info


def sample_and_update_rainbowtd3(
    ts: TrainState,
    config: RainbowTD3Config,
    key: jax.Array,
    rb: JAXReplayBuffer,
) -> Tuple[TrainState, dict[str, jax.Array]]:
    sample_key, update_key = jax.random.split(key, 2)
    big_batch = rb.sample(
        sample_key,
        config.batch_size * config.grad_step_per_env_step,
    )
    return update_rainbowtd3(ts, config, update_key, big_batch)
