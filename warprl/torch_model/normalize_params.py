import math

import torch
from torch import nn

from .layer import (
    EnsembleBatchNorm1D,
    EnsembleLayerNorm,
    EnsembleLinear,
    EnsembleRMSNorm,
    RMSNorm
)


def normalize_linear_kernel(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize each output feature vector to unit length."""
    if weight.ndim not in (2, 3):
        raise ValueError(
            "expected Linear weight with shape [O, I] or [E, O, I], "
            f"got {tuple(weight.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    norm = torch.linalg.vector_norm(weight, dim=-1, keepdim=True)
    return weight / norm.clamp_min(eps)


def normalize_scale_bias(
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly normalize affine scale and bias to sqrt(feature_dim)."""
    if scale.shape != bias.shape:
        raise ValueError(
            f"scale and bias must have the same shape, got {tuple(scale.shape)} "
            f"and {tuple(bias.shape)}"
        )
    if scale.ndim < 1:
        raise ValueError("scale and bias must have at least one dimension")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    feature_dim = scale.shape[-1]
    sqsum = (scale.square() + bias.square()).sum(dim=-1, keepdim=True)
    factor = math.sqrt(feature_dim) * torch.rsqrt(sqsum + eps)
    return scale * factor, bias * factor


def normalize_scale(scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize an affine scale vector to sqrt(feature_dim)."""
    if scale.ndim < 1:
        raise ValueError("scale must have at least one dimension")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    feature_dim = scale.shape[-1]
    sqsum = scale.square().sum(dim=-1, keepdim=True)
    return scale * (math.sqrt(feature_dim) * torch.rsqrt(sqsum + eps))


@torch.no_grad()
def project_param(module: nn.Module) -> None:
    """Project all nested linear and affine normalization parameters."""
    norm_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        EnsembleLayerNorm,
        EnsembleBatchNorm1D,
    )

    for child in module.modules():
        if isinstance(child, (nn.Linear, EnsembleLinear)):
            child.weight.copy_(normalize_linear_kernel(child.weight))

        if isinstance(child, norm_types):
            weight = getattr(child, "weight", None)
            bias = getattr(child, "bias", None)
            if weight is not None and bias is not None:
                normalized_weight, normalized_bias = normalize_scale_bias(
                    weight,
                    bias,
                )
                weight.copy_(normalized_weight)
                bias.copy_(normalized_bias)

        if isinstance(child, (nn.RMSNorm, EnsembleRMSNorm, RMSNorm)):
            weight = getattr(child, "weight", None)
            if weight is not None:
                weight.copy_(normalize_scale(weight))
