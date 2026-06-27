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
project_param)
from ..utils import (
Batch, 
GPUReplayBuffer, 
select_actor_observations, 
load_states, 
save_states, 
quantile_loss,
RewardNormalizer,
select_min_q_logits,
categorical_q_values,
make_bin_values,
categorical_projection,
categorical_ce_loss,
sample_truncated_zeta)



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


@struct.dataclass
class TrainState:
    actor: FlashSACActor
    critic: FlashSACDoubleCritic
    alpha: Alpha
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: FlashSACDoubleCritic
    alpha_opt: nnx.Optimizer
    cached_key: jax.Array
    repeat_count: jax.Array
    repeat_n: jax.Array
    asymmetric_obs: bool = struct.field(pytree_node=False, default=False)      # Use actor-only observations when set.
    grad_updates: int = 0
    reward_normalizer: RewardNormalizer | None = None


    @classmethod
    def create(cls,
               actor,
               critic,
               actor_opt,
               critic_opt,
               alpha,
               alpha_opt,
               reward_normalizer=None):
        target_critic = deepcopy(critic)
        cached_key = jax.random.PRNGKey(0)
        repeat_n = 1
        asymmetric_obs = False
        if actor.obs_dim != critic.critic.obs_dim:
            asymmetric_obs = True
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha=alpha,
            alpha_opt=alpha_opt,
            grad_updates=0,
            asymmetric_obs=asymmetric_obs,
            cached_key=cached_key,
            reward_normalizer=reward_normalizer,
            repeat_n=repeat_n,
            repeat_count=0
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
            "grad_updates": self.grad_updates,
            "reward_normalizer": self.reward_normalizer
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
            "grad_updates": self.grad_updates,
            "reward_normalizer": self.reward_normalizer
        })

        return self.replace(**model_dict)

    def get_action(self, obs):
        return _get_action(self.actor, obs, self.asymmetric_obs)

    
    def update_reward_normalizer(self, raw_rewards: jax.Array, done: jax.Array):
        if self.reward_normalizer is None:
            return self
        return self.replace(
            reward_normalizer=self.reward_normalizer.update(raw_rewards, done)
        )


    def get_exploration_action(
        self,
        obs: jax.Array,
        key: jax.Array
    ):  
        true_action_key, actions, repeat_n, repeat_count = _get_exploration_action(
            self.actor, obs, self.asymmetric_obs, self.repeat_n, self.repeat_count, self.cached_key, key)
        
        return self.replace(cached_key=true_action_key, repeat_count = repeat_count, repeat_n = repeat_n), actions
        

    def make_update_fn(self, config: RainbowSACConfig):
        @nnx.jit(donate_argnums=0)
        def jit_update(ts: TrainState, big_batch: Batch, key: jax.Array):
            return update_rainbowsac(ts, config, key, big_batch)

        return jit_update


    def make_train_step(self, config: RainbowSACConfig):
        @nnx.jit(donate_argnums=(0, 1))
        def train_step(ts: TrainState, rb: GPUReplayBuffer, transition: dict, key: jax.Array):
            if ts.reward_normalizer is not None and config.normalize_rewards:
                ts = ts.update_reward_normalizer(
                    transition["rewards"], jnp.logical_or(transition["terminations"], transition["truncations"]))
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
                "training/alpha_loss": zeros,
                "training/alpha_value": zeros,
                "training/entropy": zeros
            }
            sample_key, update_key = jax.random.split(key)
            ts, info = nnx.cond(
                rb.size >= config.learning_starts,
                lambda ts, rb: update_rainbowsac(ts, config, update_key, rb.sample(sample_key, config.batch_size * config.grad_step_per_env_step)),
                lambda ts, rb: (ts, zero_info),
                ts, rb
            )

            return ts, rb, info

        return train_step


@nnx.jit(static_argnames=('asymmetric_obs'))
def _get_action(actor: FlashSACActor, obs: jax.Array, asymmetric_obs: bool):
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)
    actions = actor.get_mean_action(obs)
    return actions

@nnx.jit(static_argnames=("asymmetric_obs",))
def _get_exploration_action(
    actor: FlashSACActor,
    obs: jax.Array,
    asymmetric_obs: bool,
    repeat_n: jax.Array,
    repeat_count: jax.Array,
    cached_key: jax.Array,
    key: jax.Array,
):
    zeta_key, action_key = jax.random.split(key, 2)
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)

    refresh = jnp.logical_or(repeat_count == 0, repeat_count >= repeat_n)
    true_action_key = jnp.where(refresh, action_key, cached_key)            # The key to create actions
    actions = actor.get_action(obs, key=true_action_key, training=False)[0]

    new_repeat_n = jnp.where(refresh, sample_truncated_zeta(zeta_key), repeat_n)
    new_repeat_count = jnp.where(refresh, 1, repeat_count + 1)
    return true_action_key, actions, new_repeat_n, new_repeat_count


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
    actor_next_observations = select_actor_observations(batch.next_observations, config.asymmetric_obs, actor.obs_dim)
    next_actions, next_log_pi = actor.get_action(
            actor_next_observations, key=key, training=False)
    obs_all = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    actions_all = jnp.concatenate([batch.actions, next_actions], axis=0)

    def mse_loss(critic: FlashSACDoubleCritic, target_critic: FlashSACDoubleCritic):
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

    def quantile_loss_fn(critic, target_critic):
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
            + config.gamma
            * (1.0 - batch.dones)
            * (bins[None, :] - alpha_value * next_log_pi)
        )
        target_probs = categorical_projection(next_q_logits, target_bins)
        ce_loss = categorical_ce_loss(q_logits, target_probs).mean()

        return ce_loss, {
            "training/q_loss": ce_loss,
            "training/q_mean": categorical_q_values(q_logits).mean(),
            }

    if  config.num_head > 1:
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
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update entropy temperature (alpha) and return the updated TrainState."""

    def alpha_loss_fn(alpha_model: Alpha):
        alpha_loss = (-alpha_model() * (- entropy + config.target_entropy)).mean()
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
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor parameters and return the updated TrainState."""
    alpha_value = alpha()
    action_key, entropy_key = jax.random.split(key, 2)
    alpha_value = jax.lax.stop_gradient(alpha_value)
    actor_observations = select_actor_observations(batch.observations, config.asymmetric_obs, actor.obs_dim)
    next_actor_observations = select_actor_observations(batch.next_observations, config.asymmetric_obs, actor.obs_dim)
    actor_obs_all = jnp.concat([actor_observations, next_actor_observations], axis=0)

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
                q_logits =  critic(batch.observations, actions, training=False)
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



def update_rainbowsac(ts: TrainState, config: RainbowSACConfig, key: jax.Array, big_batch: Batch):
    """(multiple SGD steps per env step)."""
    if config.asymmetric_obs:
        assert ts.actor.obs_dim != ts.critic.critic.obs_dim
    
    if config.normalize_rewards and ts.reward_normalizer is not None:
        normalized_rewards = ts.reward_normalizer.normalize(big_batch.rewards)
        big_batch = big_batch._replace(rewards=normalized_rewards)

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
        big_batch,
    )

    update_keys = jax.random.split(key, config.grad_step_per_env_step)


    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_minibatch(ts, sub_batch: Batch, key: jax.Array):
        critic_key, policy_key = jax.random.split(key, 2)

        alpha_value = ts.alpha() 

        policy_info = nnx.cond(
            ts.grad_updates % config.policy_frequency == 0,
            lambda ts: update_policy(ts.critic, ts.actor, ts.actor_opt, ts.alpha, ts.alpha_opt, config, sub_batch, policy_key),
            lambda ts: {
                "training/actor_loss": jnp.array(0.0),
                "training/alpha_loss": jnp.array(0.0),
                "training/alpha_value": alpha_value,
                "training/entropy": jnp.array(0.0)
            },
            ts,
        )

        critic_info = update_critic(ts.actor, ts.critic, ts.alpha, ts.critic_opt, ts.target_critic, config, sub_batch, critic_key)
        ts = ts.replace(
            grad_updates=ts.grad_updates + 1)
            
        nnx.cond(
            ts.grad_updates % config.target_frequency == 0,
            lambda critic, target_critic: soft_update(critic, target_critic, config.tau),
            lambda critic, target_critic: None,
            ts.critic, ts.target_critic, 
        )

        info = {**critic_info, **policy_info}
        return ts, info

    updated_train_state, infos = update_minibatch(
        ts, batches, update_keys)
    info = jax.tree.map(lambda x: x[-1], infos)
    return updated_train_state, info
