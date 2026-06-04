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



def evaluate_playground_policy(
    env,
    policy,
    eval_episodes: int = 100,
    max_eval_steps: int = 1_000,
    seed: int = 0
) -> dict:
    import jax
    import jax.numpy as jnp
    
    num_envs = eval_episodes
    init_state = env.reset(jax.random.split(jax.random.PRNGKey(seed), num_envs))
    def cond_fn(carry):
        state, returns_buffer, count, step = carry
        return jnp.logical_and(count < eval_episodes, step < max_eval_steps)

    def body_fn(carry):
        state, returns_buffer, count, step = carry

        actions = policy(state.obs)
        next_state = env.step(state, actions)

        done = next_state.info["episode_done"].astype(bool)
        episode_returns = next_state.info["episode"]["r"]

        def write_one(write_carry, i):
            returns_buffer, count = write_carry
            valid = jnp.logical_and(done[i], count < eval_episodes)

            returns_buffer = jax.lax.cond(
                valid,
                lambda buf: buf.at[count].set(episode_returns[i]),
                lambda buf: buf,
                returns_buffer,
            )
            count = count + valid.astype(jnp.int32)
            return (returns_buffer, count), None

        (returns_buffer, count), _ = jax.lax.scan(
            write_one,
            (returns_buffer, count),
            jnp.arange(num_envs),
        )

        return next_state, returns_buffer, count, step + 1

    init_returns = jnp.zeros((eval_episodes,), dtype=jnp.float32)

    final_state, returns_buffer, count, step = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            init_state,
            init_returns,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        ),
    )

    mean_return = returns_buffer.sum() / jnp.maximum(count, 1)
    std_return = jnp.std(returns_buffer)
    return {"eval/episode_return": mean_return, "eval/episode_return_std": std_return}
