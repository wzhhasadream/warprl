from typing import Any, Literal, Protocol


class WarpSACConfig(Protocol):
    """Configuration fields shared by the JAX and PyTorch WarpSAC agents."""

    seed: int
    env_type: str
    num_envs: int
    total_timesteps: int

    buffer_size: int
    learning_starts: int
    batch_size: int
    grad_step_per_interaction_step: int
    gamma: float
    decay_step: int
    n_step: int
    buffer_device: str

    compute_type: Literal["float32", "bfloat16"]
    policy_lr: float
    q_lr: float
    end_lr: float
    policy_frequency: int
    target_frequency: int
    tau: float
    target_entropy: float

    actor_hidden_dim: int
    actor_num_blocks: int
    critic_hidden_dim: int
    critic_num_blocks: int
    num_q: int
    num_head: int
    use_bias: bool
    dist_type: Literal["quantile", "ce", "scalar"]
    q_agg: Literal["mean", "min"]

    actor_normalize_parameters: bool
    critic_normalize_parameters: bool
    normalize_rewards: bool
    asymmetric_obs: bool


CPU_ENV_TYPES = {"mujoco", "myosuite", "dmc", "humanoid_bench"}
GPU_ENV_TYPES = {"playground", "isaaclab", "maniskill", "mjlab"}

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
        compute_type="float32",
        n_step=1,
        buffer_device="cpu",
        eval_episode=50,
        num_eval_envs=1,
        actor_normalize_parameters=True,
        critic_normalize_parameters=True
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
        compute_type="bfloat16",
        n_step=1,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
        actor_normalize_parameters=False,
        critic_normalize_parameters=False
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
        compute_type="bfloat16",
        n_step=1,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
        actor_normalize_parameters=False,
        critic_normalize_parameters=False
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
        compute_type="bfloat16",
        n_step=3,
        buffer_device="cuda",
        eval_episode=1024,
        num_eval_envs=1024,
        actor_normalize_parameters=False,
        critic_normalize_parameters=False
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
        compute_type="bfloat16",
        n_step=3,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
        actor_normalize_parameters=False,
        critic_normalize_parameters=False
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
        compute_type="bfloat16",
        n_step=3,
        buffer_device="cuda",
        eval_episode=50,
        num_eval_envs=50,
        actor_normalize_parameters=False,
        critic_normalize_parameters=False
    ),
}


def resolve_profile(args: Any) -> Any:
    profile = args.profile
    if profile == "auto":
        if args.env_type in GPU_ENV_TYPES:
            profile = args.env_type
            if args.env_type == "mjlab":
                from ...envs.mjlab import SIM2REAL_TASK_ID

                if args.env_id in SIM2REAL_TASK_ID:
                    profile = "sim2real"
        elif args.env_type in CPU_ENV_TYPES:
            profile = "cpu_sim"
        else:
            raise ValueError(f"Unknown env_type: {args.env_type}")

    if profile not in PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown profile {profile!r}; expected one of {tuple(PROFILE_DEFAULTS)}"
        )

    for key, value in PROFILE_DEFAULTS[profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


__all__ = ["PROFILE_DEFAULTS", "WarpSACConfig", "resolve_profile"]
