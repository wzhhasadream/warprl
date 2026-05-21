from typing import Callable
import gymnasium 
import numpy as np
from gymnasium.wrappers.utils import RunningMeanStd
from gymnasium.vector import VectorEnv

def evaluate_policy(
    envs: Callable| list[Callable]| VectorEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    eval_episodes: int = 100,
    num_envs: int = 10,
    seed: int = 0,
    rms: RunningMeanStd | None = None
) -> dict:
    if isinstance(envs, list):
        envs = gymnasium.vector.SyncVectorEnv(envs)
    elif callable(envs):
        envs = gymnasium.vector.SyncVectorEnv([envs for _ in range(num_envs)])
    elif isinstance(envs, VectorEnv):
        pass
    else:
        raise TypeError(f"Unsupported envs type: {type(envs)}")
    envs = gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)
    if rms is not None:
        import copy
        envs = gymnasium.wrappers.vector.NormalizeObservation(envs)
        envs.obs_rms = copy.deepcopy(rms)
        envs.update_running_mean = False

    obs, _ = envs.reset(seed=seed)

    episodic_returns = []
    episodic_success = []
    while len(episodic_returns) < eval_episodes:
        actions = np.asarray(policy(obs))

        next_obs, _, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        if dones.any():
            episodic_returns.extend(infos["episode"]["r"][dones].tolist())
            if "success" in infos:
                episodic_success.extend(infos["success"][dones].tolist())

        obs = next_obs

    envs.close()
    if len(episodic_success) > 0:
        success_rate = float(np.mean(episodic_success) * 100)
        return {"eval/episode_return": float(np.mean(episodic_returns)), "eval/episode_return_std": float(np.std(episodic_returns)), "eval/success_rate": success_rate}

    return {"eval/episode_return": float(np.mean(episodic_returns)), "eval/episode_return_std": float(np.std(episodic_returns))}



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

