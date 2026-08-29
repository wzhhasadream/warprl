from __future__ import annotations

from abc import ABC, abstractmethod
from gymnasium.vector import VectorEnv
from typing import Any, TYPE_CHECKING, TypeAlias
import numpy as np
from pathlib import Path
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
        self.critic_observation_dim = int(
            np.prod(np.asarray(self.observation_shape))
        )
        self.actor_observation_dim = self.critic_observation_dim
        self.asymmetric_obs = getattr(envs, 'asymmetric_obs', False)
        self.cfg.asymmetric_obs = self.asymmetric_obs
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

    @property
    def observation_debug_info(self) -> dict[str, int | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "actor_input_dim": self.actor_observation_dim,
            "critic_input_dim": self.critic_observation_dim,
        }

    @property
    def can_update(self) -> bool:
        pass

    @abstractmethod
    def get_action(self, obsveration: Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def get_exploration_action(self, obsveration: Tensor) -> np.ndarray:
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
    def update(self, last_obsveration: Tensor) -> dict[str, float]:
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
