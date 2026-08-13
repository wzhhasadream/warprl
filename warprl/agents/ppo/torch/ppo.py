from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv

from ....model.torch import Network

from ....buffers.on_policy.torch_buffer import TorchBuffer
from ....buffers.on_policy.types import RolloutTransition
from ...config.ppo import PPOConfig
from ...base_agent import OnPolicyAgent
from .get_action import get_eval_action, get_value, sample_and_value
from .network import Actor, ActorCritic, Critic
from .update import update_ppo


def default_learner_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent(OnPolicyAgent):
    """PyTorch PPO agent with the same rollout interface as the JAX agent."""

    def __init__(self, envs: VectorEnv, cfg: PPOConfig) -> None:
        super().__init__(envs, cfg)

        self.learner_device = default_learner_device()
        torch.manual_seed(self.cfg.seed)
        activation = getattr(F, self.cfg.activation)
        model = ActorCritic(
            Actor(
                self.actor_observation_dim,
                self.action_dim,
                self.cfg.actor_hidden_dims,
                activation,
                init_std=self.cfg.init_std
            ),
            Critic(
                self.critic_observation_dim,
                self.cfg.critic_hidden_dims,
                activation,
            ),
        ).to(self.learner_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, fused=True if torch.cuda.is_available() else False)
        self.agent = Network(model, optimizer, forward_name="get_mean_action")
        self.replay_buffer = TorchBuffer(
            self.cfg.rollout_steps,
            self.observation_space,
            self.action_space,
            self.num_envs,
            device=self.learner_device,
        )


    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.learner_device,
        ).reshape(-1, self.critic_observation_dim)

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def get_action(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        actions = get_eval_action(
            self.agent,
            self.asymmetric_obs,
            self._observations(observations),
        )
        return self._numpy(actions)

    def get_exploration_action(
        self, observations: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        return self.sample_action_and_value(observations)[0]

    def get_value(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        return self._numpy(get_value(self.agent, self._observations(observations)))

    def sample_action_and_value(
        self,
        observations: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions, log_probs, values, action_means, action_stds = sample_and_value(
            self.agent,
            self.asymmetric_obs,
            self._observations(observations),
        )
        return (
            self._numpy(actions),
            self._numpy(values),
            self._numpy(log_probs),
            self._numpy(action_means),
            self._numpy(action_stds),
        )

    def process_transition(self, transition: RolloutTransition) -> None:
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.full

    def update(self, last_observations: np.ndarray | torch.Tensor) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Collect a complete rollout before calling update.")
        info = update_ppo(
            self.agent,
            self.replay_buffer,
            self._observations(last_observations),
            self.cfg,
        )
        self.replay_buffer.reset()
        return {key: float(value.detach()) for key, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        self.agent.save(Path(checkpoint_dir) / "agent.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        self.agent.load(Path(checkpoint_dir) / "agent.pt")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        self.agent.save_onnx(
            Path(onnx_dir) / "policy.onnx",
            [(1, self.actor_observation_dim)],
        )


__all__ = ["PPOAgent", "default_learner_device"]
