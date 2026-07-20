from copy import deepcopy

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from optax import adam, cosine_decay_schedule
from nnxrl.env import CPU_SIM
from nnxrl.buffers import Transition
from nnxrl.buffers.jax_buffer import JaxBuffer
from nnxrl.model import Alpha, Network, RewardNormalizer
from .flashsacnetwork import FlashSACActor, FlashSACDoubleCritic
from nnxrl.utils import (
    default_learner_device
)
from gymnasium.vector import VectorEnv
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import WarpSACConfig, make_update_warpsac
from pathlib import Path



class WarpSACAgent:
    def __init__(
        self,
        envs: VectorEnv,
        cfg: WarpSACConfig,
    ) -> None:
        self.cfg = cfg
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.num_envs = envs.num_envs
        if self.observation_space.shape is None:
            raise ValueError("WarpSAC requires a fixed-shape observation space")
        if self.action_space.shape is None:
            raise ValueError("WarpSAC requires a fixed-shape action space")
        self.observation_shape = tuple(self.observation_space.shape)
        self.action_dim = int(np.prod(np.asarray(self.action_space.shape)))
        self.critic_observation_dim = int(
            np.prod(np.asarray(self.observation_shape))
        )
        self.actor_observation_dim = self.critic_observation_dim
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        setattr(self.cfg, "asymmetric_obs", self.asymmetric_obs)
        setattr(self.cfg, "target_entropy", 0.5 * self.action_dim * np.log(2 * np.pi * np.e * 0.15**2))
        if self.asymmetric_obs:
            actor_observation_size = getattr(
                envs, "actor_observation_size", None
            )
            if actor_observation_size is None:
                raise ValueError(
                    "Asymmetric observations require actor_observation_size"
                )
            self.actor_observation_dim = int(
                np.prod(np.asarray(actor_observation_size))
            )

        self._init_train_state()
        self._init_cached_fn()

        self._action_key, self._update_key, self._sample_key = jax.random.split(jax.random.PRNGKey(getattr(cfg, "seed")), 3)


    @property
    def observation_debug_info(self) -> dict[str, int | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "actor_input_dim": self.actor_observation_dim,
            "critic_input_dim": self.critic_observation_dim,
        }

    def _init_cached_fn(self):
        update_fn = make_update_warpsac(self.cfg)
        self._update_fn = nnx.cached_partial(
            update_fn,
            self.critic,
            self.actor,
            self.alpha,
            self.target_critic,
            self.reward_normalizer.model
        )
        self._get_action_fn = nnx.cached_partial(
            get_eval_action,
            self.actor.model,
            self.asymmetric_obs,
        )
        self._get_exploration_action_fn = nnx.cached_partial(
            get_exploration_action,
            self.actor.model,
            self.asymmetric_obs,
        )
        self._update_reward_normalizer = nnx.cached_partial(
            update_reward_normalizer,
            self.reward_normalizer.model
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
            max_size=getattr(self.cfg, "buffer_size", int(1e6)),
            linear_decay_step=getattr(self.cfg, "decay_step", 0),
            n_step=getattr(self.cfg, "n_step", 1),
            gamma=getattr(self.cfg, "gamma", 0.99),
            use_approximate_sampling=getattr(self.cfg, "env_type", "mujoco") in CPU_SIM,
            device=getattr(self.cfg, "buffer_device", 'cpu')
        )

        actor_model = FlashSACActor(
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
        critic_model = FlashSACDoubleCritic(
            self.critic_observation_dim,
            self.action_dim,
            rngs.fork(split=getattr(self.cfg, "num_q")),
            hidden_dim=getattr(self.cfg, "critic_hidden_dim"),
            num_blocks=getattr(self.cfg, "critic_num_blocks"),
            num_head=getattr(self.cfg, "num_head"),
            use_bias=getattr(self.cfg, "use_bias"),
            compute_type=compute_type,
            dist_type=(
                "scalar"
                if getattr(self.cfg, "num_head") == 1
                else getattr(self.cfg, "dist_type")
            ),
        )
        self.actor = Network(
            actor_model,
            nnx.Optimizer(
                actor_model,
                adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
                wrt=nnx.Param,
            ),
        )
        self.critic = Network(
            critic_model,
            nnx.Optimizer(
                critic_model,
                adam(cosine_decay_schedule(q_lr, num_critic_updates, end_lr / q_lr)),
                wrt=nnx.Param,
            ),
        )
        alpha_model = Alpha()
        self.alpha = Network(
            alpha_model,
            nnx.Optimizer(
                alpha_model,
                adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
                wrt=nnx.Param,
            ),
        )
        if getattr(self.cfg, "normalize_parameters"):
            self.actor.project_param()
            self.critic.project_param()
        self.target_critic = Network(
            deepcopy(self.critic.model),
            None,
            source_model=self.critic.model,
            tau=getattr(self.cfg, "tau"),
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(getattr(self.cfg, "num_envs"), getattr(self.cfg, "gamma")), None)
            if getattr(self.cfg, "normalize_rewards")
            else None
        )

        self.learner_device = default_learner_device()

        self.cached_key = jax.random.PRNGKey(0)
        self.repeat_count = jnp.array(0, dtype=jnp.int32)
        self.repeat_n = jnp.array(1, dtype=jnp.int32)
        self.critic_grad_updates = jnp.array(0, dtype=jnp.int32)


    def get_action(self, obs: jax.Array | np.ndarray):
        obs = obs.reshape((-1,) + self.observation_shape)
        actions = self._get_action_fn(obs)
        return np.asarray(actions)

    def get_exploration_action(self, obs: jax.Array | np.ndarray):
        obs = obs.reshape((-1,) + self.observation_shape)
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
        self._update_reward_normalizer(
            transition.rewards,
            transition.terminations,
            transition.truncations,
        )
        self.replay_buffer = self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return bool(
            self.replay_buffer.size >= getattr(self.cfg, "learning_starts")
            and self.replay_buffer.can_sample()
        )

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
            update_key,
            batch,
        )

        return info


    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        onnx_dir = Path(onnx_dir)
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = onnx_dir / "policy.onnx"
        import onnx
        from jax2onnx import to_onnx
        input_shape = ("B", self.actor_observation_dim)

        def policy_fn(obs: jax.Array) -> jax.Array:
            actor_obs = obs.reshape((-1, self.actor_observation_dim))
            return self.actor.model.get_mean_action(actor_obs)

        
        model = to_onnx(policy_fn, [input_shape])
        onnx.save(model, onnx_file)


    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        normalizer_checkpoint_dir = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint_dir.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint_dir)
