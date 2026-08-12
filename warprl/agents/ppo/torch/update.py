import torch
from ....buffers.on_policy.torch_buffer import TorchBuffer
from ....buffers.on_policy.types import RolloutBatch
from ...config.ppo import PPOConfig
from ....utils import select_actor_observations
from .network import ActorCritic
from ....model.torch import Network
from .utils import adapt_lr, diagonal_gaussian_kl
from torch import nn

@torch.compile(mode="max-autotune")
def ppo_loss(
    agent: Network[ActorCritic],
    batch: RolloutBatch,
    cfg: PPOConfig,
):
    actor, critic = agent.model.actor, agent.model.critic
    actor_obs = select_actor_observations(
        batch.observations,
        cfg.asymmetric_obs,
        actor.obs_dim,
    )
    _, new_log_probs, entropy, new_means, new_stds = actor.get_action(
        actor_obs,
        update_rms=False,
        actions=batch.actions,
    )
    values = critic(batch.observations, update_rms=False)
    ratio = torch.exp(new_log_probs - batch.old_log_probs)

    if cfg.algo == "ppo":
        pg_loss = -torch.minimum(
            ratio * batch.advantages,
            ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            * batch.advantages,
        ).mean()
    elif cfg.algo == "spo":
        pg_loss = -(
            batch.advantages * ratio
            - batch.advantages.abs() * (ratio - 1.0).square() / (2.0 * cfg.clip_coef)
        ).mean()
    else:
        raise ValueError(f"Unsupported PPO objective: {cfg.algo}")

    if cfg.clip_value:
        clipped_values = batch.values + (values - batch.values).clamp(
            -cfg.clip_coef,
            cfg.clip_coef,
        )
        value_loss = 0.5 * torch.maximum(
            (values - batch.returns).square(),
            (clipped_values - batch.returns).square(),
        ).mean()
    else:
        value_loss = 0.5 * (values - batch.returns).square().mean()

    entropy_mean = entropy.mean()
    loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
    kl = diagonal_gaussian_kl(
        new_means,
        new_stds,
        batch.actions_mean,
        batch.actions_std,
    ).mean()
    info = {
        "training/loss": loss.detach(),
        "training/pg_loss": pg_loss.detach(),
        "training/value_loss": value_loss.detach(),
        "training/entropy": entropy_mean.detach(),
        "training/kl": kl.detach(),
        "training/clipfrac": (ratio.sub(1.0).abs() > cfg.clip_coef).float().mean().detach(),
    }

    return loss, info


def update_ppo_minibatch(
    agent: Network[ActorCritic],
    batch: RolloutBatch,
    cfg: PPOConfig,
) -> dict[str, torch.Tensor]:
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, cfg.compute_type)):
        loss, info = ppo_loss(agent, batch, cfg)
    info = {key: value.clone() for key, value in info.items()}
    if cfg.algo == "ppo":
        for group in agent.opt.param_groups:
            group["lr"] = adapt_lr(float(group["lr"]), info["training/kl"])
    agent.opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.model.actor.parameters(), cfg.max_grad_norm)
    nn.utils.clip_grad_norm_(agent.model.critic.parameters(), cfg.max_grad_norm)
    agent.opt.step()
    return info


def update_ppo(
    agent: Network[ActorCritic],
    buffer: TorchBuffer,
    last_observations: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        last_values = agent.model.critic(last_observations, update_rms=False)
    buffer.compute_returns_and_advantages(last_values, cfg.gamma, cfg.gae_lambda)
    if cfg.normalize_advantages:
        buffer.normalize_advantages()

    infos = [
        update_ppo_minibatch(agent, batch, cfg)
        for batch in buffer.sample(cfg.num_mini_batches, cfg.num_epochs)
    ]
    agent.model.sync_rms()
    return {key: torch.stack([info[key] for info in infos]).mean() for key in infos[0]}


__all__ = ["adapt_lr", "diagonal_gaussian_kl", "update_ppo", "update_ppo_minibatch"]
