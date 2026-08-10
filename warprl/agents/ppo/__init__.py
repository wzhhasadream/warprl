__all__ = ["JaxPPOAgent", "TorchPPOAgent"]


def __getattr__(name: str):
    if name == "JaxPPOAgent":
        from .jax import PPOAgent

        return PPOAgent
    if name == "TorchPPOAgent":
        from .torch import PPOAgent

        return PPOAgent
    raise AttributeError(name)
