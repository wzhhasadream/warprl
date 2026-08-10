from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gymnasium import spaces
import numpy as np

from .types import RolloutBatch, RolloutTransition


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(
            f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        # type: ignore[misc]
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported")


class BaseBuffer(ABC):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        rollout_steps: int,
        num_envs: int = 1
    ):
        if rollout_steps <= 0 or num_envs <= 0:
            raise ValueError(
                "rollout_steps and num_envs must both be positive")
        # Extract shapes from spaces
        observation_shape = get_obs_shape(observation_space)
        action_dim = get_action_dim(action_space)

        # Handle both int and tuple for obs_shape
        if isinstance(observation_shape, int):
            self.observation_shape = (observation_shape,)
        else:
            self.observation_shape = observation_shape

        self.action_shape = (action_dim,)
        self.num_envs = int(num_envs)
        self.num_steps = int(rollout_steps)
        self.rollout_steps = self.num_steps

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, transition: RolloutTransition) -> Any:
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        pass

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> RolloutBatch:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass
