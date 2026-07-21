# warprl/agents/__init__.py
__all__ = ["TorchWarpSACAgent", "JaxWarpSACAgent"]

def __getattr__(name: str):
    if name == "TorchWarpSACAgent":
        from .torch import WarpSACAgent
        return WarpSACAgent
    if name == "JaxWarpSACAgent":
        from .jax import WarpSACAgent
        return WarpSACAgent
    raise AttributeError(name)
