# warprl/agents/__init__.py
__all__ = [
    "JaxPPOAgent",
    "JaxWarpSACAgent",
    "TorchPPOAgent",
    "TorchWarpSACAgent",
]

def __getattr__(name: str):
    if name == "TorchWarpSACAgent":
        from .warpsac.torch import WarpSACAgent
        return WarpSACAgent
    if name == "JaxWarpSACAgent":
        from .warpsac.jax import WarpSACAgent
        return WarpSACAgent
    if name == "TorchPPOAgent":
        from .ppo.torch import PPOAgent
        return PPOAgent
    if name == "JaxPPOAgent":
        from .ppo.jax import PPOAgent
        return PPOAgent
    raise AttributeError(name)
