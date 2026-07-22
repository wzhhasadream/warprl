from copy import deepcopy
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from optax import adam, cosine_decay_schedule
from ...buffers import Transition
from ...buffers.jax_buffer import JaxBuffer
from ...jax_model import Alpha, Network, RewardNormalizer, FlashSACActor, FlashSACDoubleCritic
from gymnasium.vector import VectorEnv
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import WarpSACConfig, make_update_warpsac
from pathlib import Path



def default_learner_device() -> jax.Device:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    return gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]

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
        self.observation_shape = tuple(self.observation_space.shape)
        self.action_dim = int(np.prod(np.asarray(self.action_space.shape)))
        self.critic_observation_dim = int(
            np.prod(np.asarray(self.observation_shape))
        )
        self.actor_observation_dim = self.critic_observation_dim
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        self.cfg.asymmetric_obs = self.asymmetric_obs
        self.cfg.target_entropy = float(
            0.5 * self.action_dim * np.log(2 * np.pi * np.e * 0.15**2)
        )
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

        self._action_key, self._update_key, self._sample_key = jax.random.split(
            jax.random.PRNGKey(cfg.seed), 3
        )


    @property
    def observation_debug_info(self) -> dict[str, int | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "actor_input_dim": self.actor_observation_dim,
            "critic_input_dim": self.critic_observation_dim,
        }

    def _init_cached_fn(self):
        update_fn = make_update_warpsac(self.cfg)
        reward_normalizer = (
            self.reward_normalizer.model
            if self.reward_normalizer is not None
            else None
        )
        self._update_fn = nnx.cached_partial(
            update_fn,
            self.critic,
            self.actor,
            self.alpha,
            self.target_critic,
            reward_normalizer,
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
            reward_normalizer,
        )

    def _init_train_state(
        self,
    ) :
        compute_type = getattr(jnp, self.cfg.compute_type)
        num_critic_updates = int(
            self.cfg.total_timesteps
            / self.cfg.num_envs
            * self.cfg.grad_step_per_interaction_step
        )
        end_lr = self.cfg.end_lr
        policy_lr = self.cfg.policy_lr
        q_lr = self.cfg.q_lr

        rngs = nnx.Rngs(self.cfg.seed)
        self.replay_buffer = JaxBuffer.create(
            action_space=self.action_space,
            observation_space=self.observation_space,
            num_envs=self.num_envs,
            max_size=self.cfg.buffer_size,
            linear_decay_step=self.cfg.decay_step,
            n_step=self.cfg.n_step,
            gamma=self.cfg.gamma,
            use_approximate_sampling=self.cfg.buffer_device == "cpu",
            device=self.cfg.buffer_device,
        )

        actor_model = FlashSACActor(
            self.actor_observation_dim,
            self.action_dim,
            rngs.fork(),
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.actor_num_blocks,
            action_high=1,
            action_low=-1,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
        )
        critic_model = FlashSACDoubleCritic(
            self.critic_observation_dim,
            self.action_dim,
            rngs.fork(split=self.cfg.num_q),
            hidden_dim=self.cfg.critic_hidden_dim,
            num_blocks=self.cfg.critic_num_blocks,
            num_head=self.cfg.num_head,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
            dist_type=(
                "scalar"
                if self.cfg.num_head == 1
                else self.cfg.dist_type
            ),
        )
        self.actor = Network(
            actor_model,
            nnx.Optimizer(
                actor_model,
                adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
                wrt=nnx.Param,
            ),
            forward_name="get_mean_action",
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
        if self.cfg.normalize_parameters:
            self.actor.project_param()
            self.critic.project_param()
        self.target_critic = Network(
            deepcopy(self.critic.model),
            None,
            source_model=self.critic.model,
            tau=self.cfg.tau,
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(self.cfg.num_envs, self.cfg.gamma), None)
            if self.cfg.normalize_rewards
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
            self.replay_buffer.size >= self.cfg.learning_starts
            and self.replay_buffer.can_sample()
        )

    def update(self):
        self._sample_key, sample_key = jax.random.split(self._sample_key, 2)
        sample_keys = jax.random.split(
            sample_key,
            self.cfg.grad_step_per_interaction_step,
        )
        batch = self.replay_buffer.sample_multiple_batch(
            sample_keys,
            self.cfg.batch_size,
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
        self.actor.save_onnx(
            onnx_dir / "policy.onnx",
            [("B", self.actor_observation_dim)],
        )


    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        normalizer_checkpoint_dir = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint_dir.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint_dir)
