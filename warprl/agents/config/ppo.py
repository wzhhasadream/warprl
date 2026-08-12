from typing import Literal, Protocol, Sequence


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


__all__ = ["PPOConfig"]
