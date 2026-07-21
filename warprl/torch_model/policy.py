from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Beta,
    Categorical,
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)


ActionBound = torch.Tensor | float | Sequence[float]


def _as_float_tensor(x: ActionBound) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def action_scale_bias(
    action_low: ActionBound,
    action_high: ActionBound,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the affine transform that maps [-1, 1] to action bounds."""
    action_low = _as_float_tensor(action_low)
    action_high = _as_float_tensor(action_high)
    return (action_high - action_low) / 2.0, (action_high + action_low) / 2.0


def _action_scale_bias_like(
    action_low: ActionBound,
    action_high: ActionBound,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, bias = action_scale_bias(action_low, action_high)
    return (
        scale.to(device=reference.device, dtype=reference.dtype),
        bias.to(device=reference.device, dtype=reference.dtype),
    )


def make_action_affine_bijector(
    action_low: ActionBound,
    action_high: ActionBound,
) -> AffineTransform:
    """Create an affine transform that maps [-1, 1] to action bounds."""
    scale, bias = action_scale_bias(action_low, action_high)
    return AffineTransform(loc=bias, scale=scale)


def unbounded_to_action(
    pre_tanh: torch.Tensor,
    *,
    action_low: ActionBound,
    action_high: ActionBound,
) -> torch.Tensor:
    """Map an unbounded action to the bounded action space."""
    scale, bias = _action_scale_bias_like(action_low, action_high, pre_tanh)
    return torch.tanh(pre_tanh) * scale + bias


def action_to_unbounded(
    action: torch.Tensor,
    *,
    action_low: ActionBound,
    action_high: ActionBound,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert the bounded tanh action transform."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    scale, bias = _action_scale_bias_like(action_low, action_high, action)
    normalized = ((action - bias) / scale).clamp(-1.0 + eps, 1.0 - eps)
    return torch.atanh(normalized)


def squash_log_std_tanh(
    log_std: torch.Tensor,
    *,
    log_std_min: float,
    log_std_max: float,
) -> torch.Tensor:
    """Squash log standard deviations to a closed finite interval."""
    if log_std_max <= log_std_min:
        raise ValueError("log_std_max must be larger than log_std_min")
    return log_std_min + 0.5 * (log_std_max - log_std_min) * (
        torch.tanh(log_std) + 1.0
    )


def squash_tanh_action(
    pre_action: torch.Tensor,
    pre_log_prob: torch.Tensor,
    action_low: ActionBound,
    action_high: ActionBound,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply tanh plus affine action scaling and correct the log probability."""
    scale, bias = _action_scale_bias_like(action_low, action_high, pre_action)
    action = torch.tanh(pre_action) * scale + bias
    logdet_tanh = (
        2.0
        * (torch.log(torch.tensor(2.0, device=pre_action.device, dtype=pre_action.dtype))
           - pre_action
           - torch.nn.functional.softplus(-2.0 * pre_action))
    ).sum(dim=-1)
    log_scale = scale.abs().log()
    if log_scale.ndim == 0:
        logdet_affine = pre_action.shape[-1] * log_scale
    else:
        logdet_affine = log_scale.sum(dim=-1)
    log_prob = pre_log_prob - logdet_tanh - logdet_affine
    return action, log_prob.unsqueeze(-1)


def diagonal_gaussian_kl(
    mu_c: torch.Tensor,
    std_c: torch.Tensor,
    mu_o: torch.Tensor,
    std_o: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(N(mu_o, std_o) || N(mu_c, std_c)) per batch item."""
    kl = (
        torch.log(std_c / std_o)
        + (std_o.square() + (mu_o - mu_c).square()) / (2.0 * std_c.square())
        - 0.5
    )
    return kl.sum(dim=-1)


def mask_logits(
    logits: torch.Tensor,
    legal_action_mask: torch.Tensor | None,
    *,
    invalid_logit: float = -1e9,
) -> torch.Tensor:
    """Mask invalid categorical actions and keep all-invalid rows finite."""
    if legal_action_mask is None:
        return logits
    mask = legal_action_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match logits {tuple(logits.shape)}"
        )
    masked = torch.where(mask, logits, torch.full_like(logits, invalid_logit))
    return torch.where(mask.any(dim=-1, keepdim=True), masked, torch.zeros_like(masked))


def _sample_normal(
    mean: torch.Tensor,
    std: torch.Tensor,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(mean)
    elif noise.shape != mean.shape:
        raise ValueError(
            f"noise shape {tuple(noise.shape)} must match mean {tuple(mean.shape)}"
        )
    elif noise.device != mean.device:
        raise ValueError(
            f"noise device {noise.device} must match mean device {mean.device}"
        )
    else:
        noise = noise.to(dtype=mean.dtype)
    return mean + std * noise


class MaskedCategoricalPolicy(nn.Module):
    """Categorical policy over discrete actions with an optional legality mask."""

    def __init__(self, invalid_logit: float = -1e9) -> None:
        super().__init__()
        self.invalid_logit = invalid_logit

    def dist(
        self,
        logits: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> Categorical:
        return Categorical(
            logits=mask_logits(
                logits,
                legal_action_mask,
                invalid_logit=self.invalid_logit,
            )
        )

    def sample_and_log_prob(
        self,
        logits: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.dist(logits, legal_action_mask)
        if noise is None:
            action = distribution.sample()
        else:
            if noise.shape != logits.shape[:-1]:
                raise ValueError(
                    "categorical noise shape "
                    f"{tuple(noise.shape)} must match batch shape "
                    f"{tuple(logits.shape[:-1])}"
                )
            if noise.device != logits.device:
                raise ValueError(
                    f"noise device {noise.device} must match logits device {logits.device}"
                )
            probabilities = distribution.probs
            cdf = probabilities.cumsum(dim=-1)
            uniform_noise = noise.to(dtype=probabilities.dtype).clamp(0.0, 1.0)
            action = torch.searchsorted(
                cdf,
                uniform_noise.unsqueeze(-1),
                right=True,
            ).squeeze(-1).clamp_max(probabilities.shape[-1] - 1)
        return action.to(dtype=torch.int64), distribution.log_prob(action).unsqueeze(-1)

    def greedy_action(
        self,
        logits: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.dist(logits, legal_action_mask).logits.argmax(dim=-1)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy without action squashing."""

    def __init__(
        self,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = False,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash_log_std = squash_log_std

    def transform_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
            )
        return log_std

    def dist(self, mean: torch.Tensor, log_std: torch.Tensor) -> Independent:
        std = self.transform_log_std(log_std).exp()
        return Independent(Normal(mean, std), 1)

    def sample_and_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.dist(mean, log_std)
        action = _sample_normal(mean, distribution.base_dist.scale, noise)
        return action, distribution.log_prob(action).unsqueeze(-1)


class SquashedTanhGaussianPolicy(nn.Module):
    """Tanh-squashed diagonal Gaussian policy for SAC-style continuous actions."""

    def __init__(
        self,
        action_low: ActionBound,
        action_high: ActionBound,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = True,
    ) -> None:
        super().__init__()
        action_low_tensor = _as_float_tensor(action_low)
        action_high_tensor = _as_float_tensor(action_high)
        scale, bias = action_scale_bias(action_low_tensor, action_high_tensor)
        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash_log_std = squash_log_std

    def transform_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
            )
        return log_std

    def _scale_bias_like(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.action_scale.to(dtype=reference.dtype),
            self.action_bias.to(dtype=reference.dtype),
        )

    def dist(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> TransformedDistribution:
        std = self.transform_log_std(log_std).exp()
        scale, bias = self._scale_bias_like(mean)
        base = Independent(Normal(mean, std), 1)
        return TransformedDistribution(
            base,
            [TanhTransform(cache_size=1), AffineTransform(loc=bias, scale=scale)],
        )

    def sample_and_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed_log_std = self.transform_log_std(log_std)
        std = transformed_log_std.exp()
        pre_tanh = _sample_normal(mean, std, noise)
        base_log_prob = Independent(Normal(mean, std), 1).log_prob(pre_tanh)
        return squash_tanh_action(
            pre_tanh,
            base_log_prob,
            self.action_low,
            self.action_high,
        )


class TanhDeterministicPolicy(nn.Module):
    """Deterministic tanh policy with arbitrary action bounds."""

    def __init__(self, action_low: ActionBound, action_high: ActionBound) -> None:
        super().__init__()
        action_low_tensor = _as_float_tensor(action_low)
        action_high_tensor = _as_float_tensor(action_high)
        scale, bias = action_scale_bias(action_low_tensor, action_high_tensor)
        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def action(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh) * self.action_scale.to(pre_tanh) + self.action_bias.to(pre_tanh)


class MultivariateBetaPolicy(nn.Module):
    """Independent Beta policy mapped from (0, 1) to arbitrary action bounds."""

    def __init__(
        self,
        action_low: ActionBound,
        action_high: ActionBound,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        action_low_tensor = _as_float_tensor(action_low)
        action_high_tensor = _as_float_tensor(action_high)
        action_scale_bias(action_low_tensor, action_high_tensor)
        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)
        self.epsilon = epsilon

    def _range_like(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.action_low.to(dtype=reference.dtype)
        high = self.action_high.to(dtype=reference.dtype)
        return low, high

    def _base_dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> Independent:
        concentration1 = torch.nn.functional.softplus(alpha) + self.epsilon
        concentration0 = torch.nn.functional.softplus(beta) + self.epsilon
        return Independent(Beta(concentration1, concentration0), 1)

    def dist(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> TransformedDistribution:
        low, high = self._range_like(alpha)
        return TransformedDistribution(
            self._base_dist(alpha, beta),
            [AffineTransform(loc=low, scale=high - low)],
        )

    def sample_and_log_prob(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self._base_dist(alpha, beta)
        raw_action = base.rsample()
        low, high = self._range_like(raw_action)
        action = raw_action * (high - low) + low
        log_prob = base.log_prob(raw_action) - (high - low).log().sum(dim=-1)
        return action, log_prob.unsqueeze(-1)
