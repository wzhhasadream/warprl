from __future__ import annotations

from abc import ABC, abstractmethod
from gymnasium.vector import VectorEnv
from typing import Any, TYPE_CHECKING, TypeAlias
import numpy as np
from pathlib import Path

from ..utils import is_image_observation
from ..buffers import RolloutTransition, Transition

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = np.ndarray | jax.Array | torch.Tensor

class BaseAgent(ABC):
    def __init__(
        self,
        envs: VectorEnv,
        cfg: Any
    ) -> None:
        self.cfg = cfg
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.num_envs = envs.num_envs
        self.observation_shape = tuple(self.observation_space.shape)
        self.action_dim = int(np.prod(np.asarray(self.action_space.shape)))
        self.critic_obs_shape = (
            self.observation_shape
            if len(self.observation_shape) > 1
            else (self.observation_shape[0], )
        )
        self.actor_obs_shape = self.critic_obs_shape
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        self.cfg.asymmetric_obs = self.asymmetric_obs
        self.cfg.image_obs = is_image_observation(self.observation_shape)
        self.image_obs = self.cfg.image_obs
        if self.asymmetric_obs:
            actor_observation_size = getattr(
                envs, "actor_observation_size", None
            )
            if actor_observation_size is None:
                raise ValueError(
                    "Asymmetric observations require actor_observation_size"
                )
            actor_observation_shape = tuple(actor_observation_size)
            self.actor_obs_shape = (
                actor_observation_shape
                if len(actor_observation_shape) > 1
                else (actor_observation_shape[0], )
            )

    @property
    def observation_debug_info(self) -> dict[str, int | tuple[int, ...] | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "image_obs": self.image_obs,
            "actor_obs_shape": self.actor_obs_shape,
            "critic_obs_shape": self.critic_obs_shape,
        }

    @property
    def can_update(self) -> bool:
        pass

    @abstractmethod
    def get_action(self, observation: Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def get_exploration_action(self, observation: Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        pass


    @abstractmethod
    def save_onnx(self, path: str | Path) -> None:
        pass



class OnPolicyAgent(BaseAgent):
    @abstractmethod
    def sample_action_and_value(self, observation: Tensor) -> tuple[Tensor, ...]:
        pass

    @abstractmethod
    def get_value(self, observation: Tensor) -> Tensor:
        pass

    @abstractmethod
    def process_transition(self, transition: RolloutTransition) -> None:
        pass

    @abstractmethod
    def update(self, last_observation: Tensor) -> dict[str, float]:
        pass




class OffPolicyAgent(BaseAgent):
    def __init__(self, envs: VectorEnv, cfg: Any) -> None:
        super().__init__(envs, cfg)
        self.cfg.target_entropy = float(
            0.5 * self.action_dim * np.log(2 * np.pi * np.e * 0.15**2)
        )

    @abstractmethod
    def process_transition(self, transition: Transition) -> None:
        pass

    @abstractmethod
    def update(self) -> dict[str, float]:
        pass
