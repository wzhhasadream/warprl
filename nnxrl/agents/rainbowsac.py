from copy import deepcopy

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from optax import adam, cosine_decay_schedule
from nnxrl.env import CPU_SIM
from nnxrl.buffers import Transition
from nnxrl.buffers.jax_buffer import JaxBuffer
from nnxrl.model import Alpha, FlashSACActor, FlashSACDoubleCritic, project_param
from nnxrl.utils import (
    RewardNormalizer,
    load_states,
    save_states,
)
from gymnasium.vector import VectorEnv
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import RainbowSACConfig, make_update_rainbowsac


def _default_learner_device() -> jax.Device:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    return gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]


class RainbowSACAgent:
    def __init__(
        self,
        envs: VectorEnv,
        cfg: RainbowSACConfig,
    ):
        self.cfg = cfg
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.num_envs = envs.num_envs
        self.action_dim = int(np.prod(np.asarray(self.action_space.shape)))
        self.critic_observation_dim = int(np.prod(np.asarray(self.observation_space.shape)))
        self.actor_observation_dim = int(
            np.prod(np.asarray(self.observation_space.shape)))
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        setattr(self.cfg, "asymmetric_obs", self.asymmetric_obs)
        setattr(self.cfg, "target_entropy", 0.5 * self.action_dim * np.log(2 * np.pi * np.e * 0.15**2))
        if self.asymmetric_obs:
            self.actor_observation_dim = envs.actor_observation_size

        self._init_train_state()
        self._init_cached_fn()

        self._action_key, self._update_key, self._sample_key = jax.random.split(jax.random.PRNGKey(getattr(cfg, "seed")), 3)


    def _init_cached_fn(self):
        update_fn = make_update_rainbowsac(self.cfg)
        self._update_fn = nnx.cached_partial(
            update_fn,
            self.critic,
            self.actor,
            self.actor_opt,
            self.alpha,
            self.alpha_opt,
            self.critic_opt,
            self.target_critic,
        )
        self._get_action_fn = nnx.cached_partial(
            get_eval_action,
            self.actor,
            self.asymmetric_obs,
        )
        self._get_exploration_action_fn = nnx.cached_partial(
            get_exploration_action,
            self.actor,
            self.asymmetric_obs,
        )

    def _init_train_state(
        self,
    ) :
        compute_type = getattr(jnp, getattr(self.cfg, "compute_type", "float32"))
        num_critic_updates = int(
            getattr(self.cfg, "total_timesteps")
            / getattr(self.cfg, "num_envs")
            * getattr(self.cfg, "grad_step_per_interaction_step")
        )
        end_lr = getattr(self.cfg, "end_lr")
        policy_lr = getattr(self.cfg, "policy_lr")
        q_lr = getattr(self.cfg, "q_lr")

        rngs = nnx.Rngs(getattr(self.cfg, "seed"))
        self.replay_buffer = JaxBuffer.create(
            action_space=self.action_space,
            observation_space=self.observation_space,
            num_envs=self.num_envs,
            max_size=getattr(self.cfg, "buffer_size", 512),
            linear_decay_step=getattr(self.cfg, "decay_step", 0),
            n_step=getattr(self.cfg, "n_step", 1),
            gamma=getattr(self.cfg, "gamma", 0.99),
            use_approximate_sampling=getattr(self.cfg, "env_type", "mujoco") in CPU_SIM,
            device=getattr(self.cfg, "buffer_device", 'cpu')
        )

        self.actor = FlashSACActor(
                self.actor_observation_dim,
                self.action_dim,
                rngs.fork(),
                hidden_dim=getattr(self.cfg, "actor_hidden_dim"),
                num_blocks=getattr(self.cfg, "actor_num_blocks"),
                action_high=1,
                action_low=-1,
                use_bias=getattr(self.cfg, "use_bias"),
                compute_type=compute_type,
            )
        self.critic = FlashSACDoubleCritic(
            self.critic_observation_dim,
            self.action_dim,
            rngs.fork(split=getattr(self.cfg, "num_q")),
            hidden_dim=getattr(self.cfg, "critic_hidden_dim"),
            num_blocks=getattr(self.cfg, "critic_num_blocks"),
            num_head=getattr(self.cfg, "num_head"),
            use_bias=getattr(self.cfg, "use_bias"),
            compute_type=compute_type,
        )
        
        if getattr(self.cfg, "normalize_parameters"):
            project_param(self.critic)
            project_param(self.actor)

        self.target_critic = deepcopy(self.critic)
        self.alpha = Alpha()
        self.actor_opt = nnx.Optimizer(
            self.actor,
            adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
        )
        self.critic_opt = nnx.Optimizer(
            self.critic,
            adam(cosine_decay_schedule(q_lr, num_critic_updates, end_lr / q_lr)),
        )
        self.alpha_opt = nnx.Optimizer(
            self.alpha,
            adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
        )
        self.reward_normalizer = (
            RewardNormalizer.create(getattr(self.cfg, "num_envs"), getattr(self.cfg, "gamma"))
            if getattr(self.cfg, "normalize_rewards")
            else None
        )

        self.learner_device = _default_learner_device()

        self.cached_key = jax.random.PRNGKey(0)
        self.repeat_count = jnp.array(0, dtype=jnp.int32)
        self.repeat_n = jnp.array(1, dtype=jnp.int32)
        self.critic_grad_updates = jnp.array(0, dtype=jnp.int32)


    def get_action(self, obs: jax.Array | np.ndarray):
        obs = obs.reshape((-1,) + self.observation_space.shape)
        actions = self._get_action_fn(obs)
        return np.asarray(actions)

    def get_exploration_action(self, obs):
        obs = obs.reshape((-1,) + self.observation_space.shape)
        self._action_key, action_key = jax.random.split(self._action_key, 2)
        self.cached_key, actions, self.repeat_n, self.repeat_count = self._get_exploration_action_fn(
            obs,
            self.repeat_n,
            self.repeat_count,
            self.cached_key,
            action_key,
        )
        return np.asarray(actions)

    def process_transition(self, transition: Transition) -> None:
        self.reward_normalizer = update_reward_normalizer(self.reward_normalizer, transition.rewards, transition.terminations, transition.truncations)
        self.replay_buffer = self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= getattr(self.cfg, "learning_starts") and self.replay_buffer.can_sample()

    def update(self):
        self._sample_key, sample_key = jax.random.split(self._sample_key, 2)
        sample_keys = jax.random.split(
            sample_key,
            getattr(self.cfg, "grad_step_per_interaction_step"),
        )
        batch = self.replay_buffer.sample_multiple_batch(
            sample_keys,
            getattr(self.cfg, "batch_size"),
        )
        batch = jax.tree.map(lambda x: jax.device_put(x, self.learner_device), batch)
        self._update_key, update_key = jax.random.split(self._update_key)
        self.critic_grad_updates, info = self._update_fn(
            self.critic_grad_updates,
            self.reward_normalizer,
            update_key,
            batch,
        )

        return info


    def save(self, path: str) -> None:
        save_states(path, {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "critic_grad_updates": self.critic_grad_updates,
            "reward_normalizer": self.reward_normalizer,
        })

    def load(self, path: str) -> None:
        model_dict = load_states(path, {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "critic_grad_updates": self.critic_grad_updates,
            "reward_normalizer": self.reward_normalizer,
        })
        for key, value in model_dict.items():
            setattr(self, key, value)
        self._init_cached_fn()
