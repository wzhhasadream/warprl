from typing import Any
from ..envs.mjlab import SIM2REAL_TASK_ID
CPU_ENV_TYPES = {"mujoco", "myosuite", "dmc", "humanoid_bench"}
GPU_ENV_TYPES = {"playground", "isaaclab", "maniskill", "mjlab", "holosoma"}

PROFILE_DEFAULTS = {
    "cpu_sim": dict(
        num_envs=1,
        total_timesteps=1_000_000,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=512,
        grad_step_per_interaction_step=1,
        gamma=0.99,
        decay_step=80_000,
        compute_type='float32',
        n_step=1,
        buffer_device="cpu",
        eval_episode=50,
        num_eval_envs=1
    ),
    "playground": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.97,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=1,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50
    ),
    "maniskill": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.9,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=1,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50
    ),
    "isaaclab": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.99,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=3,
        buffer_device="cuda",
        eval_episode=1024,
        # IsaacLab reuses the training vector env for evaluation, so eval envs match train envs.
        num_eval_envs=1024,
    ),
    "holosoma": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.99,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=3,
        buffer_device="cuda",
        eval_episode=1024,
        # HoloSoma reuses the training vector env for evaluation.
        num_eval_envs=1024,
    ),
    "mjlab": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.99,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=3,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
    ),
    "sim2real": dict(
        num_envs=1024,
        total_timesteps=50_000_896 * 2,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.99,
        decay_step=2_000,
        compute_type='bfloat16',
        n_step=3,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
    ),
}


def resolve_profile(args: Any) -> Any:
    profile = args.profile
    if profile == "auto":
        if args.env_type in GPU_ENV_TYPES:
            profile = args.env_type
            if args.env_type == "mjlab" and args.env_id in SIM2REAL_TASK_ID:
                profile = "sim2real"
        elif args.env_type in CPU_ENV_TYPES:
            profile = "cpu_sim"
        else:
            raise ValueError(f"Unknown env_type: {args.env_type}")

    if profile not in PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown profile {profile!r}; expected one of {tuple(PROFILE_DEFAULTS)}"
        )

    defaults = PROFILE_DEFAULTS[profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


    return args
