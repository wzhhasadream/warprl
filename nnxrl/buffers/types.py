from typing import TYPE_CHECKING, NamedTuple
import numpy as np
if TYPE_CHECKING:
    import torch
    import jax

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