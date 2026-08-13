from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....model.jax import Network
from ....buffers.on_policy.jax_buffer import JaxBuffer
from ....buffers.on_policy.types import RolloutTransition
from ...config.ppo import PPOConfig
from ...base_agent import OnPolicyAgent
from .get_action import get_eval_action, get_value, sample_and_value
from .network import Actor, ActorCritic, Critic
from .update import make_update_ppo


def default_learner_device() -> jax.Device:
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


class PPOAgent(OnPolicyAgent):
    """JAX/NNX PPO agent with fixed-horizon rollout storage."""

    def __init__(self, envs: VectorEnv, cfg: PPOConfig) -> None:
        super().__init__(envs, cfg)

        self.learner_device = default_learner_device()
        self._action_key, self._update_key = jax.random.split(jax.random.PRNGKey(self.cfg.seed))

        self._init_train_state()
        self._init_cached_fn()

    def _init_train_state(self) -> None:
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.cfg.lr)
        compute_type = getattr(jnp, self.cfg.compute_type)
        rngs = nnx.Rngs(self.cfg.seed)
        model = ActorCritic(
            Actor(
                self.actor_observation_dim,
                self.action_dim,
                self.cfg.actor_hidden_dims,
                rngs,
                getattr(jax.nn, self.cfg.activation),
                init_std=self.cfg.init_std,
                compute_type=compute_type,
            ),
            Critic(
                self.critic_observation_dim,
                self.cfg.critic_hidden_dims,
                rngs,
                getattr(jax.nn, self.cfg.activation),
                compute_type=compute_type,
            ),
        )
        self.agent = Network(
            model,
            nnx.Optimizer(model, optimizer, wrt=nnx.Param),
            forward_name="get_mean_action",
        )
        self.replay_buffer = JaxBuffer.create(
            self.cfg.rollout_steps,
            self.num_envs,
            self.observation_space,
            self.action_space,
            device=self.learner_device,
        )

    def _init_cached_fn(self) -> None:
        self._sample_and_value_fn = nnx.cached_partial(
            sample_and_value,
            self.agent,
            self.cfg.asymmetric_obs,
        )
        self._get_eval_action_fn = nnx.cached_partial(
            get_eval_action,
            self.agent,
            self.cfg.asymmetric_obs,
        )
        self._get_value_fn = nnx.cached_partial(get_value, self.agent)
        self._update_fn = nnx.cached_partial(make_update_ppo(self.cfg), self.agent)

    def _observations(self, observations: jax.Array | np.ndarray) -> jax.Array:
        obs = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (-1, self.critic_observation_dim)
        )
        return obs


    def get_action(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        actions = self._get_eval_action_fn(self._observations(observations))
        return np.asarray(actions)

    def get_exploration_action(
        self, observations: jax.Array | np.ndarray
    ) -> np.ndarray:
        return self.sample_action_and_value(observations)[0]

    def get_value(self, observations: jax.Array | np.ndarray) -> np.ndarray:
        return np.asarray(self._get_value_fn(self._observations(observations)))

    def sample_action_and_value(
        self,
        observations: jax.Array | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._action_key, action_key = jax.random.split(self._action_key)
        actions, log_probs, values, actions_mean, actions_std = self._sample_and_value_fn(
            self._observations(observations),
            action_key,
        )
        return (
            np.asarray(actions),
            np.asarray(values),
            np.asarray(log_probs),
            np.asarray(actions_mean),
            np.asarray(actions_std),
        )

    def process_transition(self, transition: RolloutTransition) -> None:
        self.replay_buffer = self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return bool(self.replay_buffer.full and self.replay_buffer.returns_ready)

    def update(self, last_observations: jax.Array | np.ndarray) -> dict[str, float]:
        self._update_key, update_key = jax.random.split(self._update_key)
        info = self._update_fn(
            self.replay_buffer,
            self._observations(last_observations),
            update_key
        )
        self.replay_buffer = self.replay_buffer.reset()
        return {key: float(np.asarray(value)) for key, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        self.agent.save(Path(checkpoint_dir) / "agent.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.agent.load(Path(checkpoint_dir) / "agent.ckpt")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        self.agent.save_onnx(Path(onnx_dir) / "policy.onnx", [(1, self.actor_observation_dim)])


__all__ = ["PPOAgent", "default_learner_device"]
