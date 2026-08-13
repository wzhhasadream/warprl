from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ....model.torch import Linear, MLP, OnPolicyRMS
from ....model.torch.policy import GaussianPolicy


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.elu,
        init_std: float = 1
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_norm = OnPolicyRMS(obs_dim)
        self.encoder = MLP(
            obs_dim,
            hidden_dims,
            layer_norm=True,
            activation_fn=activation,
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(init_std))
        self.mean_head = Linear(hidden_dims[-1], action_dim)
        self.policy = GaussianPolicy()

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        return self.encoder(self.obs_norm(obs, update_rms))

    def sync_rms(self) -> None:
        self.obs_norm.sync()

    def get_action(
        self,
        obs: torch.Tensor,
        update_rms: bool = True,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_head(self(obs, update_rms))
        log_std = self.log_std.expand_as(mean)
        dist = self.policy.dist(mean, log_std)
        if actions is None:
            actions = dist.sample()
        log_probs = dist.log_prob(actions).reshape(-1, 1)
        entropy = dist.entropy().reshape(-1, 1)
        return actions, log_probs, entropy, mean, dist.base_dist.scale

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self(obs, update_rms=False))


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.elu,
    ) -> None:
        super().__init__()
        self.obs_norm = OnPolicyRMS(obs_dim)
        self.encoder = MLP(
            obs_dim,
            hidden_dims,
            layer_norm=True,
            activation_fn=activation,
        )
        self.value_head = Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor, update_rms: bool = False) -> torch.Tensor:
        return self.value_head(self.encoder(self.obs_norm(obs, update_rms)))

    def sync_rms(self) -> None:
        self.obs_norm.sync()


class ActorCritic(nn.Module):
    def __init__(self, actor: Actor, critic: Critic) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.get_mean_action(obs)

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.get_mean_action(obs)

    def sync_rms(self) -> None:
        self.actor.sync_rms()
        self.critic.sync_rms()
