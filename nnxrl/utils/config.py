from typing import Any

CPU_ENV_TYPES = {"mujoco", "myosuite", "dmc", "humanoid_bench"}
GPU_ENV_TYPES = {"playground", "isaaclab", "maniskill"}

PROFILE_DEFAULTS = {
    "cpu_sim": dict(
        num_envs=1,
        total_timesteps=1_000_000,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=512,
        grad_step_per_env_step=1,
        eval_frequency=50_000,
        log_frequency=2_000,
        gamma=0.99,
        decay_step=80_000
    ),
    "gpu_sim": dict(
        num_envs=1024,
        total_timesteps=50_000_896,
        buffer_size=10_000_000,
        learning_starts=100_000,
        batch_size=2048,
        grad_step_per_env_step=2,
        eval_frequency=5_000_000,
        log_frequency=2_000_000,
        gamma=0.97,
        decay_step=1_000
    ),
}


def resolve_profile(args: Any) -> Any:
    profile = args.profile
    if profile == "auto":
        if args.env_type in GPU_ENV_TYPES:
            profile = "gpu_sim"
        elif args.env_type in CPU_ENV_TYPES:
            profile = "cpu_sim"
        else:
            raise ValueError(f"Unknown env_type: {args.env_type}")

    defaults = PROFILE_DEFAULTS[profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.env_type == 'myosuite':
        args.eval_episode = 100

    return args