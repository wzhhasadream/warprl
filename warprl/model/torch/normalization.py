from collections.abc import Sequence

import torch
import torch.nn as nn


def _reshape_to_samples(
    batch: torch.Tensor,
    obs_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    batch = torch.as_tensor(batch, dtype=torch.float32, device=device)
    return batch.reshape((-1,) + obs_shape)


class RMS(nn.Module):
    """Running mean and population variance."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        obs_shape: int | Sequence[int],
        epsilon: float = 1e-8,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.register_buffer(
            "mean", torch.zeros(obs_shape, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "var", torch.ones(obs_shape, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "count", torch.tensor(epsilon, dtype=torch.float32, device=device)
        )
        self.epsilon = float(epsilon)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        batch = _reshape_to_samples(
            batch, tuple(self.mean.shape), self.mean.device
        )
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total_count
        )

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
        self.count.copy_(total_count)

    def normalize(self, batch: torch.Tensor, update: bool = True) -> torch.Tensor:
        normalized_batch = (batch - self.mean) * torch.rsqrt(self.var + self.epsilon)
        if update:
            self.update(batch)
        return normalized_batch


class OnPolicyRMS(nn.Module):
    """RMS normalizer for on-policy rollouts with a frozen forward snapshot."""

    def __init__(
        self,
        obs_shape: int | Sequence[int],
        epsilon: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        # Updated from rollout observations.
        self.rms = RMS(obs_shape, epsilon, device=device)
        # Used by policy/value forward passes until the next sync.
        self.frozen_rms = RMS(obs_shape, epsilon, device=device)

    def normalize(self, batch: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(batch)
        return self.frozen_rms.normalize(batch, update=False)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Update live rollout statistics without changing forward normalization."""
        self.rms.update(batch)

    @torch.no_grad()
    def sync(self) -> None:
        """Freeze the latest rollout statistics for subsequent forward passes."""
        self.frozen_rms.mean.copy_(self.rms.mean)
        self.frozen_rms.var.copy_(self.rms.var)
        self.frozen_rms.count.copy_(self.rms.count)

    def forward(self, batch: torch.Tensor, update: bool = False) -> torch.Tensor:
        return self.normalize(batch, update)

class RewardNormalizer(nn.Module):
    """Reward normalizer based on discounted-return statistics."""

    def __init__(
        self,
        num_envs: int | None = None,
        gamma: float = 0.99,
        g_max: float = 5.0,
        epsilon: float = 1e-8,
        use_max_bound: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        shape = () if num_envs is None else (num_envs,)
        self.gamma = float(gamma)
        self.g_max = float(g_max)
        self.epsilon = float(epsilon)
        self.use_max_bound = bool(use_max_bound)
        self.register_buffer(
            "g", torch.zeros(shape, dtype=torch.float32, device=device)
        )
        self.g_rms = RMS((), epsilon=epsilon, device=device)
        self.register_buffer(
            "g_abs_max", torch.zeros((), dtype=torch.float32, device=device)
        )

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, episode_dones: torch.Tensor) -> None:
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.g.device
        ).reshape_as(self.g)
        dones = torch.as_tensor(
            episode_dones, dtype=torch.float32, device=self.g.device
        ).reshape_as(self.g)
        discounted_returns = self.gamma * (1.0 - dones) * self.g + rewards

        self.g.copy_(discounted_returns)
        self.g_rms.update(discounted_returns)
        self.g_abs_max.copy_(
            torch.maximum(self.g_abs_max, discounted_returns.abs().max())
        )

    def denominator(self) -> torch.Tensor:
        variance_scale = torch.sqrt(self.g_rms.var + self.epsilon)
        if not self.use_max_bound:
            return variance_scale
        max_scale = self.g_abs_max / max(self.g_max, self.epsilon)
        return torch.maximum(variance_scale, max_scale)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.g.device
        )
        return rewards / self.denominator()

    def update_and_normalize(
        self,
        rewards: torch.Tensor,
        episode_dones: torch.Tensor,
    ) -> torch.Tensor:
        self.update(rewards, episode_dones)
        return self.normalize(rewards)
