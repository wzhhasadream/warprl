"""WarpSAC agents and backend implementations."""

__all__ = ["JaxWarpSACAgent", "TorchWarpSACAgent"]


def __getattr__(name: str):
    if name == "JaxWarpSACAgent":
        from .jax import WarpSACAgent

        return WarpSACAgent
    if name == "TorchWarpSACAgent":
        from .torch import WarpSACAgent

        return WarpSACAgent
    raise AttributeError(name)
