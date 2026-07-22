from ...torch_model import FlashSACActor, RewardNormalizer
import torch
from ...utils import select_actor_observations
from .zeta_dist import sample_truncated_zeta

compile_mode = "max-autotune"

@torch.no_grad()
@torch.compile(mode=compile_mode, fullgraph=True)
def get_eval_action(
    actor: FlashSACActor,
    asymmetric_obs: bool,
    observations: torch.Tensor,
) -> torch.Tensor:
    obs = select_actor_observations(
        observations, asymmetric_obs, actor.obs_dim
    )
    return actor.get_mean_action(obs)


@torch.no_grad()
@torch.compile(mode=compile_mode, fullgraph=True)
def get_exploration_action(
    actor: FlashSACActor,
    asymmetric_obs: bool,
    observations: torch.Tensor,
    repeat_n: torch.Tensor,
    repeat_count: torch.Tensor,
    cached_noise: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = select_actor_observations(
        observations, asymmetric_obs, actor.obs_dim
    )
    refresh = torch.logical_or(repeat_count == 0, repeat_count >= repeat_n)
    true_action_noise = torch.where(refresh, noise, cached_noise)
    actions = actor.get_action(obs, noise=true_action_noise, training=False)[0]

    new_repeat_n = torch.where(refresh, sample_truncated_zeta(), repeat_n)
    new_repeat_count = torch.where(refresh, 1, repeat_count + 1)
    return true_action_noise, actions, new_repeat_n, new_repeat_count


@torch.no_grad()
@torch.compile(mode=compile_mode, fullgraph=True)
def update_reward_normalizer(
    reward_normalizer: RewardNormalizer | None,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> None:
    if reward_normalizer is not None:
        reward_normalizer.update(rewards, terminations.bool() | truncations.bool())

