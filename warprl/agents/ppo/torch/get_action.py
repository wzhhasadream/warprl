import torch

from ....model.torch import Network
from ....utils import select_actor_observations
from .network import ActorCritic

@torch.no_grad()
@torch.compile(mode="max-autotune")
def get_eval_action(
    agent: Network[ActorCritic],
    asymmetric_obs: bool,
    observations: torch.Tensor,
) -> torch.Tensor:
    actor_obs = select_actor_observations(
        observations,
        asymmetric_obs,
        agent.model.actor.obs_dim,
    )
    return agent(actor_obs)

@torch.no_grad()
@torch.compile(mode="max-autotune")
def get_value(
    agent: Network[ActorCritic],
    observations: torch.Tensor,
) -> torch.Tensor:
    return agent.model.critic(observations, update_rms=False)


@torch.no_grad()
@torch.compile(mode="max-autotune")
def sample_and_value(
    agent: Network[ActorCritic],
    asymmetric_obs: bool,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actor = agent.model.actor
    actor_obs = select_actor_observations(
        observations,
        asymmetric_obs,
        actor.obs_dim,
    )
    actions, log_probs, _, action_means, action_stds = actor.get_action(
        actor_obs,
        update_rms=True,
    )
    values = agent.model.critic(observations, update_rms=True)
    return actions, log_probs, values, action_means, action_stds
