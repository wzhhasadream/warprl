from .replaybuffer import JAXReplayBuffer, Batch, ReplayBuffer, GPUReplayBuffer
from .checkpoint import load_states, save_states
from .evaluate import evaluate_policy, evaluate_playground_policy
from .normalization import RMS
import jax
import numpy as np
from typing import Any
from .quantile_loss import quantile_loss



def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def select_actor_observations(observations: jax.Array, asymmetric_obs: bool, actor_obs_dim: int) -> jax.Array:
    if not asymmetric_obs:
        return observations
    return observations[..., : actor_obs_dim]


def replace_truncated_next_obs(
    next_obs: np.ndarray,
    truncations: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Replace autoreset observations with terminal observations for truncated envs."""
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_obs"][idx]
    return real_next_obs
