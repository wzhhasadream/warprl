from __future__ import annotations
import numpy as np
from typing import NamedTuple, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = jax.Array | torch.Tensor | np.ndarray

class Batch(NamedTuple):
    observations: "jax.Array | np.ndarray | torch.Tensor"
    actions: "jax.Array | np.ndarray | torch.Tensor"
    rewards: "jax.Array | np.ndarray | torch.Tensor"
    dones: "jax.Array | np.ndarray | torch.Tensor"
    next_observations: "jax.Array | np.ndarray | torch.Tensor"
    discounts: "jax.Array | np.ndarray | torch.Tensor"


class Transition(NamedTuple):
    observations: "jax.Array | np.ndarray | torch.Tensor"
    actions: "jax.Array | np.ndarray | torch.Tensor"
    rewards: "jax.Array | np.ndarray | torch.Tensor"
    truncations: "jax.Array | np.ndarray | torch.Tensor"
    terminations: "jax.Array | np.ndarray | torch.Tensor"
    next_observations: "jax.Array | np.ndarray | torch.Tensor"