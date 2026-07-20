from typing import Any, Literal
import gymnasium as gym
from .types import Batch, Transition

def create_buffer(
    action_space: gym.spaces.Space,
    observation_space: gym.spaces.Space,
    buffer_type: Literal["numpy", "np", "jax", "torch", "pytorch"] = "numpy",
    num_env: int = 1,
    device: Any = "cpu",
    max_size: int = int(1e6),
    linear_decay_step: int = 0,
    min_weight: float = 0.1,
    n_step: int = 1,
    gamma: float = 0.99,
    use_approximate_sampling: bool = True,
    num_buckets: int = 2000,
) -> Any:
    buffer_type = buffer_type.lower()
    common_kwargs = dict(
        observation_space=observation_space,
        action_shape_space=action_space,
        max_size=max_size,
        linear_decay_step=linear_decay_step,
        min_weight=min_weight,
        n_step=n_step,
        gamma=gamma,
        n_envs=num_env,
        use_approximate_sampling=use_approximate_sampling,
        num_buckets=num_buckets,
    )

    if buffer_type in ("numpy", "np"):
        from .numpy_buffer import NumpyBuffer

        return NumpyBuffer(**common_kwargs)
    if buffer_type in ("torch", "pytorch"):
        from .torch_buffer import TorchBuffer

        return TorchBuffer(**common_kwargs, device=device)
    if buffer_type == "jax":
        from .jax_buffer import JaxBuffer

        return JaxBuffer.create(
            observation_space=observation_space,
            action_space=action_space,
            max_size=max_size,
            linear_decay_step=linear_decay_step,
            min_weight=min_weight,
            n_step=n_step,
            gamma=gamma,
            num_envs=num_env,
            use_approximate_sampling=use_approximate_sampling,
            num_buckets=num_buckets,
            device=device,
        )

    raise ValueError(f"Invalid buffer_type: {buffer_type}")


__all__ = ["Batch", "Transition", "create_buffer"]
