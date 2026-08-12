from collections.abc import Callable
import torch
import torch.nn.functional as functional

from ....buffers.off_policy import Batch
from ...config.warpsac import WarpSACConfig
from ....model.torch import (
    Alpha,
    CategoricalPolicy,
    Network,
    QuantilePolicy,
    RewardNormalizer,
)
from .network import FlashSACActor, FlashSACDoubleCritic
from ....utils import select_actor_observations
import torch.utils._pytree as pytree


compile_mode = "max-autotune"
def _scalar_loss(
    q_values: torch.Tensor,
    next_values: torch.Tensor,
    batch: Batch,
    alpha_value: torch.Tensor,
    next_log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch.rewards + (1.0 - batch.dones) * batch.discounts * (
        next_values.min(dim=0).values - alpha_value * next_log_pi
    )
    return functional.mse_loss(q_values, targets.expand_as(q_values)), q_values.mean()


def _quantile_loss(
    policy: QuantilePolicy,
    q_values: torch.Tensor,
    next_values: torch.Tensor,
    batch: Batch,
    alpha_value: torch.Tensor,
    next_log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch.rewards + (1.0 - batch.dones) * batch.discounts * (
        next_values.min(dim=0).values - alpha_value * next_log_pi
    )
    return policy.loss(q_values, targets).mean(), q_values.mean()


def _categorical_loss(
    policy: CategoricalPolicy,
    q_logits: torch.Tensor,
    next_logits: torch.Tensor,
    batch: Batch,
    alpha_value: torch.Tensor,
    next_log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_logits = policy.select_min_logits(next_logits)
    target_bins = batch.rewards + (1.0 - batch.dones) * batch.discounts * (
        policy.bins - alpha_value * next_log_pi
    )
    targets = policy.target_probs(next_logits, target_bins)
    return policy.loss(q_logits, targets).mean(), policy.q_values(q_logits).mean()


@torch.compile(mode=compile_mode)
def critic_loss(
    actor: Network[FlashSACActor],
    critic: Network[FlashSACDoubleCritic],
    alpha: Network[Alpha],
    target_critic: Network[FlashSACDoubleCritic],
    config: WarpSACConfig,
    batch: Batch,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        actor_next_observations = select_actor_observations(batch.next_observations, config.asymmetric_obs, actor.model.obs_dim)
        next_actions, next_log_pi = actor.model.get_action(
            actor_next_observations, training=False
        )
        observations = torch.cat(
            (batch.observations, batch.next_observations), dim=0
        )
        actions = torch.cat((batch.actions, next_actions), dim=0)
        next_values = target_critic.model(observations, actions, training=True)[
            :, batch.observations.shape[0] :
        ]

    q_values = critic.model(observations, actions, training=True)[
        :, : batch.observations.shape[0]
    ]
    alpha_value = alpha.model().detach()
    policy = critic.model.dist
    if isinstance(policy, CategoricalPolicy):
        loss, q_mean = _categorical_loss(
            policy,
            q_values,
            next_values,
            batch,
            alpha_value,
            next_log_pi,
        )
    elif isinstance(policy, QuantilePolicy):
        loss, q_mean = _quantile_loss(
            policy,
            q_values,
            next_values,
            batch,
            alpha_value,
            next_log_pi,
        )
    else:
        loss, q_mean = _scalar_loss(
            q_values,
            next_values,
            batch,
            alpha_value,
            next_log_pi,
        )

    return loss, {"training/q_loss": loss.detach(), "training/q_mean": q_mean.detach()}


def update_critic(
    actor: Network[FlashSACActor],
    critic: Network[FlashSACDoubleCritic],
    alpha: Network[Alpha],
    target_critic: Network[FlashSACDoubleCritic],
    config: WarpSACConfig,
    batch: Batch,
):
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, config.compute_type)):
        loss, info = critic_loss(actor, critic, alpha, target_critic, config, batch)

    critic.grad_step(loss)
    info = pytree.tree_map(lambda x: x.detach().clone(), info)
    if config.critic_normalize_parameters:
        critic.project_param()

    return info

@torch.compile(mode=compile_mode)
def actor_loss(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
) -> dict[str, torch.Tensor]:
    actor_observations = select_actor_observations(
        batch.observations, config.asymmetric_obs, actor.model.obs_dim
    )
    next_actor_observations = select_actor_observations(
        batch.next_observations, config.asymmetric_obs, actor.model.obs_dim
    )
    actions, log_pi = actor.model.get_action(
        torch.cat((actor_observations, next_actor_observations), dim=0),
        training=True,
    )
    actions, log_pi = actions[: batch.observations.shape[0]], log_pi[: batch.observations.shape[0]]
    q_values = critic.model.q_values(batch.observations, actions, training=False)
    def q_agg(
        mode: Literal["mean", "min"],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return {
            "mean": lambda values: values.mean(dim=0),
            "min": lambda values: values.amin(dim=0),
        }[mode]
    q_values = q_agg(config.q_agg)(q_values)
    alpha_value = alpha.model().detach()
    loss = -(q_values - alpha_value * log_pi).mean()

    return loss, {
        "training/actor_loss": loss.detach(),
        "training/entropy": -log_pi.detach().mean(),
    }


@torch.compile(mode=compile_mode)
def alpha_loss(
    alpha: Network[Alpha],
    entropy: torch.Tensor,
    config: WarpSACConfig,
) -> dict[str, torch.Tensor]:
    alpha_value = alpha.model()
    loss = (-alpha_value * (config.target_entropy - entropy.detach())).mean()
    return loss, {
        "training/alpha_loss": loss.detach(),
        "training/alpha_value": alpha_value,
    }


def update_policy(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    config: WarpSACConfig,
    batch: Batch,
) -> dict[str, torch.Tensor]:
    critic.model.requires_grad_(False)
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", getattr(torch, config.compute_type)):
        loss, actor_info = actor_loss(critic, actor, alpha, config, batch)
    actor.grad_step(loss)
    critic.model.requires_grad_(True)
    actor_info = pytree.tree_map(lambda x : x.clone(), actor_info)
    if config.actor_normalize_parameters:
        actor.project_param()
    loss, alpha_info = alpha_loss(alpha, actor_info["training/entropy"], config)
    alpha.grad_step(loss)
    alpha_info = pytree.tree_map(lambda x : x.clone(), alpha_info)
    return {
        **actor_info,
        **alpha_info,
    }




def update_warpsac(
    critic: Network[FlashSACDoubleCritic],
    actor: Network[FlashSACActor],
    alpha: Network[Alpha],
    target_critic: Network[FlashSACDoubleCritic],
    reward_normalizer: RewardNormalizer | None,
    do_policy: bool,
    do_target: bool,
    batch: Batch,
    config: WarpSACConfig
) -> dict:

    if reward_normalizer is not None:
        batch = batch._replace(
            rewards=reward_normalizer.normalize(batch.rewards)
        )
    policy_info = {}
    if do_policy:
        policy_info = update_policy(critic, actor, alpha, config, batch)
    critic_info = update_critic(actor, critic, alpha, target_critic, config, batch)
    if do_target:
        target_critic.soft_update()
    info = {**policy_info, **critic_info}
    return info
    
