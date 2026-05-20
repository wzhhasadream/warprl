from copy import deepcopy
from typing import Dict, Tuple, Protocol
import jax
import jax.numpy as jnp
from flax import nnx, struct
from ..utils.normalization import RMS
from ..utils.replaybuffer import Batch, JAXReplayBuffer
from ..model import TanhDetActor, soft_update, EnsembleCritic, quantile_loss, project_normalized_parameters
from ..utils.checkpoint import load_states, save_states


class TD3Config(Protocol):
    seed: int
    total_timesteps: int
    num_envs: int
    learning_starts: int
    num_evals: int

    gamma: float
    tau: float

    batch_size: int
    grad_step_per_env_step: int
    policy_frequency: int

    policy_lr: float

    normalize_observation: bool
    exploration_noise: float

    policy_noise: float
    noise_clip: float
    num_head: int
    normalize_parameters: bool

@struct.dataclass
class TrainState:
    """Mutable TD3 training state (models + optimizer states)."""

    actor: TanhDetActor
    critic: EnsembleCritic
    target_actor: TanhDetActor
    target_critic: EnsembleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer

    rms: RMS | None = None
    grad_updates: int = 0  

    @classmethod
    def create(cls,
    actor,
    critic,
    actor_opt,
    critic_opt,
    rms=None):
        target_actor = deepcopy(actor)
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            rms=rms,
            grad_updates=0
        )

    def save(self, path: str):
        save_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "target_critic": self.target_critic,
            "target_actor": self.target_actor,
            "rms": self.rms,
            "grad_updates": self.grad_updates
        })

    def load(self, path: str):
        model_dict = load_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "target_critic": self.target_critic,
            "target_actor": self.target_actor,
            "rms": self.rms,
            "grad_updates": self.grad_updates
        })

        return self.replace(**model_dict)

    @nnx.jit
    def get_action(self, obs: jax.Array):
        if self.rms is not None:
            obs_for_policy, _ = self.rms.normalize(obs, update=False)
        else:
            obs_for_policy = obs
        
        actions = self.actor.get_action(obs_for_policy)
        return actions

    @nnx.jit
    def get_exploration_action(self, obs: jax.Array, exploration_noise: float, key: jax.Array):
        if self.rms is not None:
            obs_for_policy, rms = self.rms.normalize(obs, update=True)
        else:
            obs_for_policy = obs
            rms = None
        actions = self.actor.get_action(obs_for_policy)
        noise = jax.random.normal(
            key, shape=actions.shape) * self.actor.action_scale * exploration_noise
        actions = jnp.clip(
            noise + actions, self.actor.action_low,  self.actor.action_high)
        return self.replace(rms=rms), actions





def _target_policy_smoothing(
    target_actor: TanhDetActor,
    next_observations: jax.Array,
    *,
    key: jax.Array,
    policy_noise: float,
    noise_clip: float,
) -> jax.Array:
    """Compute smoothed target actions for TD3 critic targets.

    TD3 target policy smoothing:
        a' = clip(pi'(s') + clip(eps, -c, c) * action_scale, action_low, action_high)
    where eps ~ Normal(0, policy_noise).
    """
    target_actions = target_actor.get_action(next_observations)
    noise = jax.random.normal(key, target_actions.shape) * policy_noise
    clipped_noise = jnp.clip(noise, -noise_clip, noise_clip) * target_actor.action_scale
    a = jnp.clip(target_actions + clipped_noise, target_actor.action_low, target_actor.action_high)
    return a


def update_critic(
    ts: TrainState,
    config: TD3Config,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update TD3 critics and return the updated TrainState."""
    target_key, _ = jax.random.split(key)
    smoothed_actions = _target_policy_smoothing(
        ts.target_actor,
        batch.next_observations,
        key=target_key,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )

    def critic_loss_fn(critic, target_critic):
        target_q = target_critic(batch.next_observations, smoothed_actions)    # [num_q, B, 1]
        min_q = jnp.min(target_q, axis=0)                                                  # [B, 1]
        target_values = batch.rewards + config.gamma * (1.0 - batch.dones) * min_q
        q = critic(batch.observations, batch.actions)
        # [num_q, B, 1] - [B, 1]
        q_loss = jnp.mean((q - target_values) ** 2)

        info = {
            "training/critic_loss": q_loss,
            "training/q_mean": jnp.mean(q),
        }
        return q_loss, info

    def quantile_loss_fn(critic, target_critic):
        q_dist = critic(batch.observations, batch.actions)    # [num_q, B, num_head]
        next_q_dist = target_critic(batch.next_observations, smoothed_actions).min(
            0)   # [B, num_head]
        target_dist = batch.rewards + config.gamma * (1.0 - batch.dones) * next_q_dist

        q_loss = quantile_loss(q_dist, target_dist).mean()

        info = {
            "training/critic_loss": q_loss,
            "training/q_mean": jnp.mean(q_dist),
        }
        return q_loss, info

    if config.num_head == 1:
        loss = critic_loss_fn
    elif config.num_head > 1:
        loss = quantile_loss_fn
    (_loss, info), grads = nnx.value_and_grad(
        loss, has_aux=True
    )(ts.critic, ts.target_critic )
    ts.critic_opt.update(grads)
    if config.normalize_parameters:
        project_normalized_parameters(ts.critic)
    return ts, info





def update_actor(
    train_state: TrainState,
    batch: Batch,
    config: TD3Config
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update TD3 actor and return the updated TrainState.
    """
    def actor_loss_fn(actor, critic):
        actions = actor.get_action(batch.observations)
        if config.num_head == 1:
            q = critic(batch.observations, actions)[0]
        elif config.num_head > 1:
            q_dist = critic(batch.observations, actions)[0]
            q = q_dist.mean(-1, keepdims=True)
        actor_loss = -jnp.mean(q)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True, argnums=0)(train_state.actor, train_state.critic)
    train_state.actor_opt.update(grads)
    if config.normalize_parameters:
        project_normalized_parameters(train_state.actor)

    soft_update(train_state.actor, train_state.target_actor, config.tau)
    soft_update(train_state.critic, train_state.target_critic, config.tau)

    return train_state, info




def update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
    big_batch: Batch     
) -> tuple[TrainState, dict[str, jax.Array]]:
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

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_td3_mininbatch(ts, batch, step):
        ts, critic_info = update_critic(ts, config, batch, jax.random.fold_in(key, step))
        ts = ts.replace(grad_updates=ts.grad_updates + 1)


        ts, policy_info = nnx.cond(
            ts.grad_updates % config.policy_frequency == 0, 
            lambda ts: update_actor(ts, batch, config), 
            lambda ts: (ts, {"training/actor_loss": jnp.array(0.0)}),
            ts)
        return ts, {**critic_info, **policy_info}

    ts, infos = update_td3_mininbatch(train_state, batches, jnp.arange(config.grad_step_per_env_step))

    info = jax.tree.map(lambda x : x[-1], infos)

    return ts, info



def sample_and_update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
    rb: JAXReplayBuffer
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = rb.sample(
        sample_key,
        config.batch_size * config.grad_step_per_env_step
    )
    ts, info = update_td3(train_state, config, update_key, big_batch)

    return ts, info

#####################################################################
################## JAX ENV ##########################################
#####################################################################




