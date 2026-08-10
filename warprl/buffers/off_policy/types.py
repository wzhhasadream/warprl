from __future__ import annotations
import numpy as np
from typing import NamedTuple, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import jax
    import torch

    Tensor: TypeAlias = jax.Array | torch.Tensor | np.ndarray

class Batch(NamedTuple):
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    next_observations: Tensor
    discounts: Tensor


class Transition(NamedTuple):
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    truncations: Tensor
    terminations: Tensor
    next_observations: Tensor