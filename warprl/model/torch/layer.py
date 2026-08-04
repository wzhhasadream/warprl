from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if bias:
            nn.init.zeros_(self.bias)
        nn.init.orthogonal_(self.weight, gain=1)




class BatchNorm1d(nn.BatchNorm1d):
    """PyTorch BatchNorm1d with an optional per-call training override."""

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        self._check_input_dim(x)
        if training is None:
            training = self.training

        use_batch_stats = training or not self.track_running_stats
        reduce_dims = (0,) + tuple(range(2, x.ndim))
        stat_shape = (1, self.num_features) + (1,) * (x.ndim - 2)

        if use_batch_stats:
            sample_count = x.shape[0]
            for dim in range(2, x.ndim):
                sample_count *= x.shape[dim]
            if sample_count <= 1:
                raise ValueError(
                    "expected more than one value per channel when using batch statistics"
                )

            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, correction=0, keepdim=True)

        if training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            assert self.running_mean is not None
            assert self.running_var is not None

            batch_mean = mean.reshape(self.num_features)
            batch_var = var.reshape(self.num_features)
            unbiased_var = batch_var * sample_count / (sample_count - 1)

            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    factor = self.num_batches_tracked.to(
                        dtype=self.running_mean.dtype
                    ).reciprocal()
                    self.running_mean.lerp_(
                        batch_mean.to(dtype=self.running_mean.dtype), factor
                    )
                    self.running_var.lerp_(
                        unbiased_var.to(dtype=self.running_var.dtype), factor
                    )
                else:
                    self.running_mean.lerp_(
                        batch_mean.to(dtype=self.running_mean.dtype), self.momentum
                    )
                    self.running_var.lerp_(
                        unbiased_var.to(dtype=self.running_var.dtype), self.momentum
                    )

        if not use_batch_stats:
            assert self.running_mean is not None
            assert self.running_var is not None
            mean = self.running_mean.view(stat_shape)
            var = self.running_var.view(stat_shape)

        output = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            assert self.weight is not None
            assert self.bias is not None
            output = output * self.weight.view(stat_shape)
            output = output + self.bias.view(stat_shape)
        return output


class RMSNorm(nn.Module):
    """RMSNorm expressed with ONNX-supported primitive operations."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        output = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (output * self.weight).to(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [
                Linear(dims[i], dims[i + 1], bias=bias)
                for i in range(len(hidden_dims))
            ]
        )
        self.layer_norm = layer_norm
        self.activation_fn = activation_fn
        if layer_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(dims[i + 1]) for i in range(len(hidden_dims))]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.norms[i](x)
            x = self.activation_fn(x)
        return x



# -------------------------------------
# Ensembled layers for Ensemble Critic
# -------------------------------------


def _add_ensemble_dim(x: torch.Tensor, num_ensemble: int):
    """
    Notice, 
    """
    if x.shape[0] != num_ensemble:
        x = torch.broadcast_to(x[None, ...], (num_ensemble, *x.shape))
        return x
    else:
        return x



class EnsembleLinear(nn.Module):
    """Independent linear layers stored in one parameter tensor.

    A shared input has shape ``[B, I]`` and returns ``[E, B, O]``. An
    ensemble-specific input has shape ``[E, B, I]`` and returns ``[E, B, O]``.
    """

    def __init__(
        self,
        num_ensemble: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_ensemble < 1:
            raise ValueError(
                f"num_ensemble must be positive, got {num_ensemble}")

        self.num_ensemble = num_ensemble
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(num_ensemble, out_features, in_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(num_ensemble, out_features)
                         ) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for member_weight in self.weight:
            nn.init.orthogonal_(member_weight, gain=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble)

        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError(
                    f"expected ensemble dimension {self.num_ensemble}, got {x.shape[0]}"
                )
        else:
            output = torch.einsum("ebi,eoi->ebo", x, self.weight)

        if self.bias is not None:
            output = output + self.bias.reshape(self.num_ensemble, 1, self.out_features)
        return output


class EnsembleBatchNorm1D(nn.Module):
    """BatchNorm1d with independent affine parameters and running stats per member.

    Inputs have shape ``[E, B, C, ...]`` or ``[B, C, ...]`` The optional
    ``training`` argument overrides ``self.training`` for one forward pass.
    """

    def __init__(
        self,
        num_ensemble: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if num_ensemble < 1:
            raise ValueError(
                f"num_ensemble must be positive, got {num_ensemble}")
        if num_features < 1:
            raise ValueError(
                f"num_features must be positive, got {num_features}")

        self.num_ensemble = num_ensemble
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_ensemble, num_features))
            self.bias = nn.Parameter(torch.zeros(num_ensemble, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_ensemble, num_features)
            )
            self.register_buffer(
                "running_var", torch.ones(num_ensemble, num_features)
            )
            self.register_buffer(
                "num_batches_tracked", torch.zeros(
                    num_ensemble, dtype=torch.int64)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble)    # [E, B, C, ...]
        if x.shape[2] != self.num_features:   
            raise ValueError(
                "expected input shape "
                f"[{self.num_ensemble}, N, {self.num_features}, ...], got {tuple(x.shape)}"
            )


        if training is None:
            training = self.training
        use_batch_stats = training or not self.track_running_stats
        reduce_dims = (1,) + tuple(range(3, x.ndim))
        stat_shape = (self.num_ensemble, 1, self.num_features) + \
            (1,) * (x.ndim - 3)

        if use_batch_stats:
            sample_count = x.shape[1]
            for dim in range(3, x.ndim):
                sample_count *= x.shape[dim]
            if sample_count <= 1:
                raise ValueError(
                    "expected more than one value per feature when using batch statistics"
                )

            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, correction=0, keepdim=True)

            if training and self.track_running_stats:
                assert self.running_mean is not None
                assert self.running_var is not None
                assert self.num_batches_tracked is not None

                batch_mean = mean.reshape(self.num_ensemble, self.num_features)
                batch_var = var.reshape(
                    self.num_ensemble, self.num_features)
                unbiased_var = batch_var * sample_count / (sample_count - 1)

                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        factor = self.num_batches_tracked.to(
                            dtype=self.running_mean.dtype
                        ).reciprocal()[:, None]
                        self.running_mean.lerp_(batch_mean.to(
                            self.running_mean.dtype), factor)
                        self.running_var.lerp_(unbiased_var.to(
                            self.running_var.dtype), factor)
                    else:
                        self.running_mean.lerp_(
                            batch_mean.to(
                                self.running_mean.dtype), self.momentum
                        )
                        self.running_var.lerp_(
                            unbiased_var.to(
                                self.running_var.dtype), self.momentum
                        )
        else:
            assert self.running_mean is not None
            assert self.running_var is not None
            mean = self.running_mean.view(stat_shape)
            var = self.running_var.view(stat_shape)

        output = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            assert self.weight is not None
            assert self.bias is not None
            affine_shape = stat_shape
            output = output * self.weight.view(affine_shape)
            output = output + self.bias.view(affine_shape)
        return output


class EnsembleLayerNorm(nn.Module):
    """LayerNorm with independent affine parameters per ensemble member.

    Inputs must have a leading ensemble dimension ``[E, ...]`` and end in
    ``normalized_shape``.
    """

    def __init__(
        self,
        num_ensemble: int,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_ensemble < 1:
            raise ValueError(
                f"num_ensemble must be positive, got {num_ensemble}")

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        if not normalized_shape or any(dim < 1 for dim in normalized_shape):
            raise ValueError(f"invalid normalized_shape: {normalized_shape}")

        self.num_ensemble = num_ensemble
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        parameter_shape = (num_ensemble, *normalized_shape)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(parameter_shape))
            self.bias = nn.Parameter(torch.zeros(
                parameter_shape)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble)
        num_normalized_dims = len(self.normalized_shape)
        if x.ndim < num_normalized_dims + 1:
            raise ValueError(
                f"expected an ensemble dimension and {num_normalized_dims} normalized dimensions, "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] != self.num_ensemble:
            raise ValueError(
                f"expected ensemble dimension {self.num_ensemble}, got {x.shape[0]}"
            )
        if tuple(x.shape[-num_normalized_dims:]) != self.normalized_shape:
            raise ValueError(
                f"expected trailing shape {self.normalized_shape}, got {tuple(x.shape)}"
            )

        reduce_dims = tuple(range(x.ndim - num_normalized_dims, x.ndim))
        mean = x.mean(dim=reduce_dims, keepdim=True)
        var = x.var(dim=reduce_dims, correction=0, keepdim=True)
        output = (x - mean) * torch.rsqrt(var + self.eps)

        if self.elementwise_affine:
            assert self.weight is not None
            affine_shape = (self.num_ensemble,) + (1,) * (
                x.ndim - num_normalized_dims - 1
            ) + self.normalized_shape
            output = output * self.weight.view(affine_shape)
            if self.bias is not None:
                output = output + self.bias.view(affine_shape)
        return output


class EnsembleRMSNorm(nn.Module):
    """RMSNorm with an independent scale vector for each ensemble member."""

    def __init__(
        self,
        num_ensemble: int,
        num_features: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if num_ensemble < 1:
            raise ValueError(
                f"num_ensemble must be positive, got {num_ensemble}"
            )
        if num_features < 1:
            raise ValueError(
                f"num_features must be positive, got {num_features}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.num_ensemble = num_ensemble
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_ensemble, num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _add_ensemble_dim(x, self.num_ensemble)
        if x.ndim < 3 or x.shape[0] != self.num_ensemble:
            raise ValueError(
                "expected input shape "
                f"[{self.num_ensemble}, ..., {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.num_features:
            raise ValueError(
                "expected input with "
                f"{self.num_features} trailing features, got {tuple(x.shape)}"
            )

        rms = x.square().mean(dim=-1, keepdim=True)
        output = x * torch.rsqrt(rms + self.eps)
        if self.weight is not None:
            weight_shape = (self.num_ensemble,) + (1,) * (x.ndim - 2) + (
                self.num_features,
            )
            output = output * self.weight.view(weight_shape)
        return output


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        num_ensemble: int,
        in_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.num_ensemble = num_ensemble
        self.layers = nn.ModuleList(
            [
                EnsembleLinear(num_ensemble, dims[i], dims[i + 1], bias=bias)
                for i in range(len(hidden_dims))
            ]
        )
        self.layer_norm = layer_norm
        self.activation_fn = activation_fn
        if layer_norm:
            self.norms = nn.ModuleList(
                [
                    EnsembleLayerNorm(num_ensemble, dims[i + 1])
                    for i in range(len(hidden_dims))
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.norms[i](x)
            x = self.activation_fn(x)
        return x
