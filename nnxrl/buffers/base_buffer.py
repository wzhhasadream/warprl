from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
from .__init__ import Batch, Transition

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
        action_shape_space: spaces.Space,
        max_size: int = int(1e6),
    ):
        self.max_size = max_size
        self.action_space = action_shape_space
        self.obsveration_space = observation_space
        # Extract shapes from spaces
        obsveration_shape = get_obs_shape(observation_space)
        action_dim = get_action_dim(action_shape_space)

        # Handle both int and tuple for obs_shape
        if isinstance(obsveration_shape, int):
            self.obsveration_shape = (obsveration_shape,)
        else:
            self.obsveration_shape = obsveration_shape

        self.action_shape = (action_dim,)


    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, transition: Transition) -> None:
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
