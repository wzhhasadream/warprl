import torch
import torch.nn.functional as F
from torch import nn


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        num_bins: int,
        v_min: float,
        v_max: float,
    ) -> None:
        super().__init__()
        self.register_buffer("bins", torch.linspace(v_min, v_max, num_bins)[None, ...])

    def q_values(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.softmax(dim=-1) * self.bins.to(logits)).sum(
            dim=-1,
            keepdim=True,
        )

    def select_min_logits(self, logits: torch.Tensor) -> torch.Tensor:
        indices = self.q_values(logits).argmin(dim=0)
        return logits.gather(0, indices[None].expand(1, -1, logits.shape[-1])).squeeze(0)

    def target_probs(
        self,
        target_logits: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        bins = self.bins.to(target_logits)
        v_min = bins[..., :1]
        v_max = bins[..., -1:]
        bin_width = bins[..., 1:2] - v_min
        target_values = target_values.to(target_logits).clamp(
            v_min,
            v_max,
        ).expand_as(target_logits)
        b = (target_values - v_min) / bin_width
        lower = b.floor().long()
        upper = b.ceil().long()
        probs = target_logits.softmax(dim=-1)
        projected = torch.zeros_like(probs)
        projected.scatter_add_(
            -1,
            lower,
            probs * (upper.to(probs) + (lower == upper).to(probs) - b),
        )
        projected.scatter_add_(-1, upper, probs * (b - lower.to(probs)))
        return projected.detach()

    def _loss_one(
        self,
        logits: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def loss(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        target_probs = target_probs.detach()
        if logits.ndim == 2:
            return self._loss_one(logits, target_probs)
        return torch.vmap(self._loss_one, in_dims=(0, None))(logits, target_probs)


class QuantilePolicy(nn.Module):
    def __init__(self, num_taus: int) -> None:
        super().__init__()
        taus = (torch.arange(num_taus, dtype=torch.float32) + 0.5) / num_taus
        self.register_buffer("taus", taus)

    def q_values(self, quantiles: torch.Tensor) -> torch.Tensor:
        return quantiles.mean(dim=-1, keepdim=True)

    def _loss_one(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        kappa: float,
    ) -> torch.Tensor:
        diff = target_quantiles[:, None] - quantiles[:, :, None]
        abs_diff = diff.abs()
        huber = torch.where(
            abs_diff <= kappa,
            0.5 * diff.square(),
            kappa * (abs_diff - 0.5 * kappa),
        )
        taus = self.taus.to(quantiles)
        weight = (taus[:, None] - (diff < 0).to(diff.dtype)).abs()
        return (weight * huber / kappa).sum(dim=1).mean()

    def loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        if quantiles.ndim == 2:
            return self._loss_one(quantiles, target_quantiles, kappa)
        return torch.vmap(self._loss_one, in_dims=(0, None, None))(
            quantiles,
            target_quantiles,
            kappa,
        )
