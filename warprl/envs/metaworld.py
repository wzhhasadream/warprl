
import gymnasium as gym

METAWORLD_MT10 = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]

METAWORLD_MT50 = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
]

METAWORLD_BENCHMARK_SIZES = {
    "MT10": 10,
    "MT50": 50,
}



def make_metaworld_env(
    env_name: str,
    seed: int,
    render_mode: str | None = None,
) -> gym.Env:
    import metaworld  # noqa: F401


    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        disable_env_checker=True,
    )

    return env


def make_metaworld_benchmark_envs(
    benchmark_name: str,
    seed: int,
    num_envs: int,
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
    use_one_hot: bool = True,
    vector_strategy: str = "async",
) -> gym.vector.VectorEnv:
    import metaworld  # noqa: F401

    benchmark_name = benchmark_name.upper()
    if benchmark_name not in METAWORLD_BENCHMARK_SIZES:
        raise ValueError(
            f"Unsupported Meta-World benchmark {benchmark_name!r}; expected MT10 or MT50"
        )

    expected_num_envs = METAWORLD_BENCHMARK_SIZES[benchmark_name]
    if num_envs != expected_num_envs:
        raise ValueError(
            f"{benchmark_name} requires num_envs={expected_num_envs}, got {num_envs}"
        )


    kwargs = {
        "vector_strategy": vector_strategy,
        "autoreset_mode": "SameStep",
        "num_envs": num_envs,
        "seed": seed,
        "use_one_hot": use_one_hot,
        "render_mode": render_mode,
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    return gym.make_vec(f"Meta-World/{benchmark_name}", **kwargs)
