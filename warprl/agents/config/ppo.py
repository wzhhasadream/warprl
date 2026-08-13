from typing import Any, Literal, Protocol, Sequence


class PPOConfig(Protocol):
    """Configuration fields shared by the JAX and PyTorch PPO agents."""

    algo: Literal["ppo", "spo"]
    seed: int
    total_timesteps: int
    rollout_steps: int
    num_mini_batches: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    lr: float
    max_grad_norm: float
    actor_hidden_dims: Sequence[int]
    critic_hidden_dims: Sequence[int]
    activation: str
    normalize_advantages: bool
    asymmetric_obs: bool
    clip_coef: float
    value_coef: float
    entropy_coef: float
    clip_value: bool
    init_std: float
    compute_type: Literal["float32", "bfloat16"]
    asymmetric_obs: bool
    init_std: float


PPO_PROFILE_DEFAULTS = {
    "cpu_sim": {
        "num_envs": 1,
        "num_eval_envs": 50,
        "eval_episode": 50,
        "compute_type": "float32",
        "total_timesteps": 4_000_000,
        "rollout_steps": 2_048,
        "num_mini_batches": 32,
        "num_epochs": 10,
    },
    "gpu_sim": {
        "num_envs": 4_096,
        "num_eval_envs": 50,
        "eval_episode": 50,
        "compute_type": "bfloat16",
        "total_timesteps": 100_000_000,
        "rollout_steps": 24,
        "num_mini_batches": 4,
        "num_epochs": 5,
    },
}


def resolve_profile(args: Any) -> Any:
    """Fill unset PPO runtime fields from the selected profile."""
    profile = args.profile
    if profile == "auto":
        profile = "cpu_sim" if args.env_type in {
            "mujoco",
            "myosuite",
            "dmc",
            "humanoid_bench",
        } else "gpu_sim"

    if profile not in PPO_PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown PPO profile {profile!r}; "
            f"expected one of {tuple(PPO_PROFILE_DEFAULTS)}"
        )

    for key, value in PPO_PROFILE_DEFAULTS[profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


__all__ = ["PPOConfig", "PPO_PROFILE_DEFAULTS", "resolve_profile"]
