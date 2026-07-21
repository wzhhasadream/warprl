from .evaluate import evaluate_policy, record_video
import numpy as np
from typing import Any



def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def select_actor_observations(observations: np.ndarray, asymmetric_obs: bool, actor_obs_dim: int) -> np.ndarray:
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


