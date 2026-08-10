"""On-policy rollout storage implementations."""

from .types import RolloutBatch, RolloutTransition, Trajectory

__all__ = [
    "JaxBuffer",
    "JaxRolloutBuffer",
    "RolloutBatch",
    "RolloutTransition",
    "TorchBuffer",
    "TorchRolloutBuffer",
    "Trajectory",
]


def __getattr__(name: str):
    if name in {"JaxBuffer", "JaxRolloutBuffer"}:
        from .jax_buffer import JaxBuffer, JaxRolloutBuffer

        return {"JaxBuffer": JaxBuffer, "JaxRolloutBuffer": JaxRolloutBuffer}[name]
    if name in {"TorchBuffer", "TorchRolloutBuffer"}:
        from .torch_buffer import TorchBuffer, TorchRolloutBuffer

        return {"TorchBuffer": TorchBuffer, "TorchRolloutBuffer": TorchRolloutBuffer}[name]
    raise AttributeError(name)
