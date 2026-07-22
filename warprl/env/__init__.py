import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, VectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RescaleAction, TimeLimit


CPU_SIM = ("mujoco", "dmc", "myosuite", "humanoid_bench")
GPU_SIM = ("playground", "isaaclab", "maniskill", "mjlab", "holosoma")







class RepeatAction(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        terminated = False
        truncated = False
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            combined_info.update(info)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, combined_info


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
    clip_action: bool = False,
    render_mode: str | None = None,
) -> tuple[VectorEnv, VectorEnv, VectorEnv]:

    if env_type in CPU_SIM:
        train_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_train_envs,
            action_repeat=action_repeat,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            clip_action=clip_action,
            render_mode=None,
        )
        eval_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_eval_envs,
            action_repeat=action_repeat,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            clip_action=clip_action,
            render_mode=None,
        )
        record_env = create_vec_env(
            env_type=env_type,
            env_name=env_name,
            seed=seed,
            num_envs=num_record_envs,
            action_repeat=action_repeat,
            rescale_action=rescale_action,
            max_episode_steps=max_episode_steps,
            clip_action=clip_action,
            render_mode=render_mode,
        )

        return train_env, eval_env, record_env

    elif env_type in GPU_SIM:
        if env_type == "playground":
            from .playground import make_playground_env
            train_env = make_playground_env(
                env_name=env_name, 
                seed=seed, 
                num_envs=num_train_envs,
                max_episode_steps=max_episode_steps,
                clip_action=clip_action,
                action_repeat=action_repeat)
            eval_env = make_playground_env(
                env_name=env_name, 
                seed=seed, 
                num_envs=num_eval_envs,
                max_episode_steps=max_episode_steps,
                clip_action=clip_action,
                action_repeat=action_repeat)
            record_env = make_playground_env(
                env_name=env_name,
                seed=seed,
                num_envs=num_record_envs,
                max_episode_steps=max_episode_steps,
                clip_action=clip_action,
                action_repeat=action_repeat)
            
            return train_env, eval_env, record_env
        if env_type == "isaaclab":
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
            return train_env, eval_env, record_env

        if env_type == "holosoma":
            from .holosoma import make_holosoma_env

            train_env = make_holosoma_env(
                task_id=env_name,
                seed=seed,
                num_envs=num_train_envs,
                headless=True,
                render_mode=render_mode,
            )
            eval_env = train_env
            record_env = train_env
            return train_env, eval_env, record_env

        if env_type == 'maniskill':
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
            return train_env, eval_env, record_env


        if env_type == "mjlab":
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

            return train_env, eval_env, record_env

        raise ValueError(f"Unsupported env_type: {env_type}")



def create_vec_env(
    env_type: str,
    env_name: str,
    num_envs: int,
    seed: int,
    rescale_action: bool = True,
    action_repeat: int = 1,
    max_episode_steps: int = 1000,
    clip_action: bool = False,
    render_mode: str | None = None,
) -> VectorEnv:

    def make_one_env(
        env_type: str,
        env_name: str,
        seed: int,
        rescale_action: bool,
        action_repeat: int,
        max_episode_steps: int,
        clip_action: bool,
        render_mode: str | None,
    ) -> gym.Env:

        if env_type == 'dmc':
            from .dmc import make_dmc_env
            env = make_dmc_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'mujoco':
            from .mujoco import make_mujoco_env
            env = make_mujoco_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'humanoid_bench':
            from .humanoid_bench import make_humanoid_env
            env = make_humanoid_env(env_name, seed, render_mode=render_mode)
        elif env_type == 'myosuite':
            from .myosuite import make_myosuite_env
            env = make_myosuite_env(env_name, seed, render_mode=render_mode)
        else:
            raise NotImplementedError

        if rescale_action:
            env = RescaleAction(env, -1.0, 1.0)

        # limit max_steps before action_repeat.
        env = TimeLimit(env, max_episode_steps)

        if action_repeat > 1:
            env = RepeatAction(env, action_repeat)

        if clip_action:
            env = gym.wrappers.ClipAction(env)

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
                action_repeat=action_repeat,
                max_episode_steps=max_episode_steps,
                clip_action=clip_action,
                render_mode=render_mode,
            )
        )
        for i in range(num_envs)
    ]
    if len(env_fns) > 1:
        envs = AsyncVectorEnv(env_fns, autoreset_mode='SameStep')
    else:
        envs = SyncVectorEnv(env_fns, autoreset_mode='SameStep')

    return envs
