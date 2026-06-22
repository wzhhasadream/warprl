from .replaybuffer import JAXReplayBuffer, Batch, ReplayBuffer, GPUReplayBuffer
from .checkpoint import load_states, save_states
from .evaluate import evaluate_policy, evaluate_playground_policy
from .normalization import RMS, RewardNormalizer
from .zeta_dist import (
    build_truncated_zeta_cdf,
    sample_integer_from_cdf,
    sample_truncated_zeta,
)
import jax
import numpy as np
from typing import Any
from .distributional_loss import (
    quantile_loss,
    categorical_ce_loss,
    categorical_q_values,
    categorical_projection,
    make_bin_values,
    select_min_q_logits,
)
from .config import resolve_profile



def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def select_actor_observations(observations: jax.Array, asymmetric_obs: bool, actor_obs_dim: int) -> jax.Array:
    if not asymmetric_obs:
        return observations
    return observations[..., : actor_obs_dim]


def replace_done_next_obs(
    next_obs: np.ndarray,
    dones: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Restore final observations for environments that ended this step."""
    real_next_obs = next_obs.copy()

    if not np.any(dones):
        return real_next_obs

    if "final_obs" not in infos:
        raise KeyError('Expected infos["final_obs"] when any environment is done.')

    final_obs = infos["final_obs"]
    for env_idx in np.flatnonzero(dones):
        real_next_obs[env_idx] = final_obs[env_idx]

    return real_next_obs
