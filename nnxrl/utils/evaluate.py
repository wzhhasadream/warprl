from typing import Callable

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.utils import RunningMeanStd


def _as_float_array(values, *, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.shape == ():
        arr = np.full((size,), arr.item(), dtype=object)
    out = np.zeros((size,), dtype=np.float32)
    for idx, value in enumerate(arr.reshape(-1)):
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            out[idx] = value
    return out


def _extract_step_success(infos: dict, num_envs: int) -> np.ndarray:
    if "success" in infos:
        valid = np.asarray(infos.get("_success", np.ones(num_envs, dtype=bool)), dtype=bool)
        return _as_float_array(infos["success"], size=num_envs) * valid.astype(np.float32)

    final_info = infos.get("final_info")
    if isinstance(final_info, dict) and "success" in final_info:
        valid = np.asarray(infos.get("_final_info", np.ones(num_envs, dtype=bool)), dtype=bool)
        return _as_float_array(final_info["success"], size=num_envs) * valid.astype(np.float32)

    if isinstance(final_info, np.ndarray):
        out = np.zeros((num_envs,), dtype=np.float32)
        valid = np.asarray(infos.get("_final_info", np.ones(num_envs, dtype=bool)), dtype=bool)
        for idx, item in enumerate(final_info):
            if valid[idx] and isinstance(item, dict) and "success" in item:
                out[idx] = float(item["success"])
        return out

    return np.zeros((num_envs,), dtype=np.float32)


def evaluate_policy(
    envs: Callable | list[Callable] | VectorEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    eval_episodes: int = 100,
    num_envs: int = 10,
    seed: int = 0,
    rms: RunningMeanStd | None = None,
) -> dict:
    close_envs = False
    if isinstance(envs, list):
        envs = gymnasium.vector.SyncVectorEnv(envs)
        close_envs = True
    elif callable(envs):
        envs = gymnasium.vector.SyncVectorEnv([envs for _ in range(num_envs)])
        close_envs = True
    elif not isinstance(envs, VectorEnv):
        raise TypeError(f"Unsupported envs type: {type(envs)}")

    if rms is not None:
        import copy

        envs = gymnasium.wrappers.vector.NormalizeObservation(envs)
        envs.obs_rms = copy.deepcopy(rms)
        envs.update_running_mean = False

    obs, _ = envs.reset(seed=seed)

    episodic_returns = []
    episodic_success = []
    running_returns = np.zeros(envs.num_envs, dtype=np.float64)
    running_successes = np.zeros(envs.num_envs, dtype=np.float32)

    while len(episodic_returns) < eval_episodes:
        actions = np.asarray(policy(obs))
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        running_returns += np.asarray(rewards, dtype=np.float64)
        running_successes += _extract_step_success(infos, envs.num_envs)

        dones = np.logical_or(terminated, truncated)
        if dones.any():
            episodic_returns.extend(running_returns[dones].tolist())
            episodic_success.extend((running_successes[dones] > 0).astype(np.float32).tolist())
            running_returns[dones] = 0.0
            running_successes[dones] = 0.0

        obs = next_obs

    if close_envs:
        envs.close()

    result = {
        "eval/episode_return": float(np.mean(episodic_returns)),
        "eval/episode_return_std": float(np.std(episodic_returns)),
    }
    if episodic_success:
        result["eval/success_rate"] = float(np.mean(episodic_success))
    return result


def record_video(
    policy: Callable[np.ndarray],
    env: VectorEnv,
    num_episodes: int = 10,
    video_length: int = 1000,
) -> np.ndarray:
    if num_episodes == 0:
        return {}
    num_envs = env.num_envs
    num_eval_episodes_per_env = max(num_episodes // num_envs, 1)

    total_videos = []

    for _ in range(num_eval_episodes_per_env):
        videos: list[np.ndarray] = []

        observations, infos = env.reset()
        images = env.render()  # type: ignore
        dones = np.zeros(num_envs)
        while np.sum(dones) < num_envs:
            actions = policy(observations)
            actions = np.array(actions)
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions)


            # once an episode is done in a sub-environment, we assume it to be done.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            videos.append(images)  # type: ignore
            images = env.render()
            observations = next_observations

        total_videos.append(np.stack(videos, axis=1))  # (num_envs, t, c, h, w)

    # TODO: if there is termination, video length can be different
    # maybe add zero-padding depending on the max length
    total_videos = np.concatenate(total_videos, axis=0)  # (b, t, h, w, c)
    total_videos = total_videos[:, :video_length]


    env.reset()

    return total_videos
