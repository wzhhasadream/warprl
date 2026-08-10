from .evaluate import evaluate_policy, record_video
import numpy as np
from typing import Any, Callable



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




def bootstrap_timeout_rewards(
    rewards: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
    next_obs: np.ndarray,
    value_fn: Callable[[np.ndarray], np.ndarray],
    infos: dict[str, Any],
) -> np.ndarray:
    """Add time-limit value bootstrapping without mutating environment rewards."""
    if not np.any(timeouts):
        return rewards

    final_obs = replace_done_next_obs(next_obs, timeouts, infos)
    final_values = value_fn(final_obs)
    rewards = rewards.copy()
    rewards[timeouts] += gamma * final_values[timeouts].reshape(-1)
    return rewards
