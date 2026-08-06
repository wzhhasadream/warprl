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
    norm = torch.linalg.vector_norm(weight, dim=-1, keepdim=True)
    return weight / norm.clamp_min(eps)


def normalize_scale_bias(
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly normalize affine scale and bias to sqrt(feature_dim)."""
    feature_dim = scale.shape[-1]
    sqsum = (scale.square() + bias.square()).sum(dim=-1, keepdim=True)
    factor = math.sqrt(feature_dim) * torch.rsqrt(sqsum + eps)
    return scale * factor, bias * factor


def normalize_scale(scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize an affine scale vector to sqrt(feature_dim)."""
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
            if child.weight is not None and child.bias is not None:
                normalized_weight, normalized_bias = normalize_scale_bias(
                    child.weight,
                    child.bias,
                )
                child.weight.copy_(normalized_weight)
                child.bias.copy_(normalized_bias)

        if isinstance(child, (nn.RMSNorm, EnsembleRMSNorm, RMSNorm)):
            if child.weight is not None:
                child.weight.copy_(normalize_scale(child.weight))
