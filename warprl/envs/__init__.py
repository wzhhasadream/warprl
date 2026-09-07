import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, VectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RescaleAction, TimeLimit

from .wrapper import ActionRepeat, PixelObservation, wrap_vector_env

CPU_SIM = ("mujoco", "dmc", "myosuite", "humanoid_bench", "metaworld")
GPU_SIM = ("playground", "isaaclab", "maniskill", "mjlab")


def create_envs(
    env_name: str,
    env_type: str,
    seed: int,
    num_train_envs: int = 1,
    num_eval_envs: int = 1,
    num_record_envs: int = 1,
    rescale_action: bool = True,
    action_repeat: int = 1,
    max_episode_steps: int = 1000,
    clip_action: bool = True,
    render_mode: str | None = None,
    gray: bool = False,
    image_layout: str = "HWC",
) -> tuple[VectorEnv, VectorEnv, VectorEnv]:

    if env_type == "metaworld" and env_name.upper() in ["MT10", "MT50"]:
        from .metaworld import make_metaworld_benchmark_envs

        train_env = make_metaworld_benchmark_envs(
            benchmark_name=env_name,
            seed=seed,
            num_envs=num_train_envs,
            render_mode=None,
            max_episode_steps=max_episode_steps,
            use_one_hot=True,
        )
        eval_env = make_metaworld_benchmark_envs(
            benchmark_name=env_name,
            seed=seed,
            num_envs=num_eval_envs,
            render_mode=None,
            max_episode_steps=max_episode_steps,
            use_one_hot=True,
        )
        record_env = make_metaworld_benchmark_envs(
            benchmark_name=env_name,
            seed=seed,
            num_envs=num_record_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            use_one_hot=True,
        )

    elif env_type in CPU_SIM:
        train_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_train_envs,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            render_mode=None,
            action_repeat=action_repeat,
            gray=gray,
            image_layout=image_layout,
        )
        eval_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_eval_envs,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            render_mode=None,
            action_repeat=action_repeat,
            gray=gray,
            image_layout=image_layout,
        )
        record_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_record_envs,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            action_repeat=action_repeat,
            gray=gray,
            image_layout=image_layout,
        )

    elif env_type in GPU_SIM:
        if env_type == "playground":
            from .playground import make_playground_env
            train_env = make_playground_env(
                env_name=env_name, 
                seed=seed, 
                num_envs=num_train_envs,
                max_episode_steps=max_episode_steps,
                action_repeat=action_repeat)
            eval_env = make_playground_env(
                env_name=env_name, 
                seed=seed, 
                num_envs=num_eval_envs,
                max_episode_steps=max_episode_steps,
                action_repeat=action_repeat)
            record_env = make_playground_env(
                env_name=env_name,
                seed=seed,
                num_envs=num_record_envs,
                max_episode_steps=max_episode_steps,
                action_repeat=action_repeat)
            
        elif env_type == "isaaclab":
            from .isaaclab import make_isaaclab_env
            train_env = make_isaaclab_env(
                env_name=env_name, 
                seed=seed, 
                num_envs=num_train_envs,
                headless=True,
                render_mode=render_mode,
            )
            eval_env = train_env
            record_env = train_env

        elif env_type == 'maniskill':
            from .maniskill import make_maniskill_env
            train_env = make_maniskill_env(
                env_name=env_name, 
                render_mode=None, 
                num_envs=num_train_envs
                )
            eval_env = make_maniskill_env(
                env_name=env_name, 
                render_mode=None, 
                num_envs=num_eval_envs)
            record_env = make_maniskill_env(
                env_name=env_name, 
                render_mode=render_mode, 
                num_envs=num_record_envs)


        elif env_type == "mjlab":
            from .mjlab import make_mjlab_env

            train_env = make_mjlab_env(
                task_id=env_name,
                num_envs=num_train_envs,
                seed=seed,
                render_mode=None
            )
            eval_env = make_mjlab_env(
                task_id=env_name,
                num_envs=num_eval_envs,
                seed=seed,
                render_mode=None
            )
            record_env = make_mjlab_env(
                task_id=env_name,
                num_envs=num_record_envs,
                seed=seed,
                render_mode=render_mode
            )

        else:
            raise ValueError(f"Unsupported env_type: {env_type}")

    # CPU action repeat is applied inside each single environment, before pixels.
    external_action_repeat = 1 if (env_type in CPU_SIM and env_name.upper() not in ["MT10", "MT50"]) or env_type == "playground" else action_repeat
    shared_eval = eval_env is train_env
    shared_record = record_env is train_env
    train_env = wrap_vector_env(
        train_env,
        action_repeat=external_action_repeat,
        clip_action=clip_action,
    )
    eval_env = (
        train_env
        if shared_eval
        else wrap_vector_env(
            eval_env,
            action_repeat=external_action_repeat,
            clip_action=clip_action,
        )
    )
    record_env = (
        train_env
        if shared_record
        else wrap_vector_env(
            record_env,
            action_repeat=external_action_repeat,
            clip_action=clip_action,
        )
    )

    return train_env, eval_env, record_env


def create_vec_env(
    env_type: str,
    env_name: str,
    num_envs: int,
    seed: int,
    rescale_action: bool = True,
    max_episode_steps: int = 1000,
    render_mode: str | None = None,
    action_repeat: int = 1,
    gray: bool = False,
    image_layout: str = "HWC",
) -> VectorEnv:

    def make_one_env(
        env_type: str,
        env_name: str,
        seed: int,
        rescale_action: bool,
        max_episode_steps: int,
        render_mode: str | None,
        action_repeat: int,
        gray: bool,
        image_layout: str,
    ) -> gym.Env:
        is_pixel_obs = env_name.endswith("-visual")
        env_name = env_name.removesuffix("-visual")
        if is_pixel_obs:
            render_mode = "rgb_array"
        if env_type == 'dmc':
            from .dmc import make_dmc_env
            env = make_dmc_env(
                env_name, seed, render_mode=render_mode
            )
        elif env_type == 'mujoco':
            from .mujoco import make_mujoco_env
            env = make_mujoco_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'humanoid_bench':
            from .humanoid_bench import make_humanoid_env
            env = make_humanoid_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'myosuite':
            from .myosuite import make_myosuite_env
            env = make_myosuite_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'metaworld':
            from .metaworld import make_metaworld_env
            env = make_metaworld_env(env_name, seed, render_mode=render_mode)
        else:
            raise NotImplementedError

        if rescale_action:
            env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))


        # limit max_steps before action_repeat.
        env = TimeLimit(env, max_episode_steps)

        if action_repeat > 1:
            env = ActionRepeat(env, action_repeat)

        if is_pixel_obs:
            env = PixelObservation(env, gray=gray, image_layout=image_layout)

        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    env_fns = [
        (
            lambda i=i: make_one_env(
                env_type=env_type,
                env_name=env_name,
                seed=seed + i,
                rescale_action=rescale_action,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                action_repeat=action_repeat,
                gray=gray,
                image_layout=image_layout,
            )
        )
        for i in range(num_envs)
    ]
    if len(env_fns) > 1:
        envs = AsyncVectorEnv(env_fns, autoreset_mode='SameStep', context="spawn")
    else:
        envs = SyncVectorEnv(env_fns, autoreset_mode='SameStep')

    return envs
