from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ...buffers import Transition
from ...buffers.torch_buffer import TorchBuffer
from ...torch_model import (
    Alpha,
    Network,
    RewardNormalizer,
)
from .warpsac_network import FlashSACActor, FlashSACDoubleCritic
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import WarpSACConfig, update_warpsac
import torch.utils._pytree as pytree

class WarpSACAgent:
    def __init__(self, envs: VectorEnv, cfg: WarpSACConfig) -> None:
        self.cfg = cfg
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.num_envs = envs.num_envs
        self.observation_shape = tuple(self.observation_space.shape)
        self.critic_observation_dim = int(np.prod(self.observation_shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.asymmetric_obs = getattr(envs, "asymmetric_obs", False)
        self.actor_observation_dim = self.critic_observation_dim
        if self.asymmetric_obs:
            self.actor_observation_dim = int(
                np.prod(getattr(envs, "actor_observation_size"))
            )

        self.cfg.asymmetric_obs = self.asymmetric_obs
        self.cfg.target_entropy = float(
            0.5 * self.action_dim * np.log(2 * np.pi * np.e * 0.15**2)
        )
        self.learner_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self.dtype = getattr(torch, cfg.compute_type)
        torch.manual_seed(cfg.seed)
        self._init_train_state()
        self._update_fn = update_warpsac
        self.repeat_count = torch.tensor(0, device=self.learner_device)
        self.repeat_n = torch.tensor(0, device=self.learner_device)
        self.cached_noise = torch.randn((self.num_envs, self.action_dim), device=self.learner_device)
        self.critic_grad_updates = 0

    @property
    def observation_debug_info(self) -> dict[str, int | bool]:
        return {
            "asymmetric_obs": self.asymmetric_obs,
            "actor_input_dim": self.actor_observation_dim,
            "critic_input_dim": self.critic_observation_dim,
        }

    def _init_train_state(self) -> None:
        num_updates = int(
            self.cfg.total_timesteps
            / self.cfg.num_envs
            * self.cfg.grad_step_per_interaction_step
        )
        self.replay_buffer = TorchBuffer(
            observation_space=self.observation_space,
            action_space=self.action_space,
            max_size=self.cfg.buffer_size,
            linear_decay_step=self.cfg.decay_step,
            n_step=self.cfg.n_step,
            gamma=self.cfg.gamma,
            num_envs=self.num_envs,
            use_approximate_sampling=self.cfg.buffer_device == "cpu",
            device=self.cfg.buffer_device,
        )
        actor_model = FlashSACActor(
            self.actor_observation_dim,
            self.action_dim,
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.actor_num_blocks,
            action_low=-1.0,
            action_high=1.0,
            use_bias=self.cfg.use_bias,
        ).to(device=self.learner_device)
        critic_model = FlashSACDoubleCritic(
            self.critic_observation_dim,
            self.action_dim,
            num_q=self.cfg.num_q,
            hidden_dim=self.cfg.critic_hidden_dim,
            num_blocks=self.cfg.critic_num_blocks,
            num_head=self.cfg.num_head,
            dist_type=(
                "scalar"
                if self.cfg.num_head == 1
                else self.cfg.dist_type
            ),
            use_bias=self.cfg.use_bias,
        ).to(device=self.learner_device)
        use_fused_adam = torch.cuda.is_available()
        actor_optimizer = Adam(
            actor_model.parameters(), lr=self.cfg.policy_lr, fused=use_fused_adam
        )
        critic_optimizer = Adam(
            critic_model.parameters(), lr=self.cfg.q_lr, fused=use_fused_adam
        )
        alpha_model = Alpha().to(self.learner_device)
        alpha_optimizer = Adam(
            alpha_model.parameters(), lr=self.cfg.policy_lr, fused=use_fused_adam
        )
        self.actor = Network(
            actor_model,
            actor_optimizer,
            scheduler=CosineAnnealingLR(
                actor_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
            forward_name="get_mean_action",  # for onnx export
        )
        self.critic = Network(
            critic_model,
            critic_optimizer,
            scheduler=CosineAnnealingLR(
                critic_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
        )
        self.alpha = Network(
            alpha_model,
            alpha_optimizer,
            scheduler=CosineAnnealingLR(
                alpha_optimizer, num_updates, eta_min=self.cfg.end_lr
            ),
        )
        if self.cfg.actor_normalize_parameters:
            self.actor.project_param()
        if self.cfg.critic_normalize_parameters:
            self.critic.project_param()
        target_critic_model = deepcopy(critic_model)
        target_critic_model.requires_grad_(False)
        self.target_critic = Network(
            target_critic_model, source_model=critic_model, tau=self.cfg.tau
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(self.num_envs, self.cfg.gamma, device=self.learner_device))
            if self.cfg.normalize_rewards
            else None
        )

    def _observations(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            observations, device=self.learner_device, dtype=torch.float32
        ).reshape(-1, self.critic_observation_dim)

    def get_action(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        actions = get_eval_action(
            self.actor.model, self.asymmetric_obs, self._observations(observations)
        )
        return actions.cpu().numpy()

    def get_exploration_action(
        self, observations: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        observations = self._observations(observations)
        noise = torch.randn((self.num_envs, self.action_dim), device=self.learner_device)
        cached_noise, actions, repeat_n, repeat_count = get_exploration_action(
            self.actor.model,
            self.asymmetric_obs,
            observations,
            self.repeat_n,
            self.repeat_count,
            self.cached_noise,
            noise,
        )

        self.cached_noise = cached_noise.clone()
        self.repeat_n = repeat_n.clone()
        self.repeat_count = repeat_count.clone()

        return actions.cpu().numpy()

    def process_transition(self, transition: Transition) -> None:
        update_reward_normalizer(
            self.reward_normalizer.model,
            torch.as_tensor(transition.rewards, device=self.learner_device),
            torch.as_tensor(transition.terminations, device=self.learner_device),
            torch.as_tensor(transition.truncations, device=self.learner_device),
        )
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample()

    def update(self) -> dict[str, float]:
        for _ in range(self.cfg.grad_step_per_interaction_step):
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            batch = pytree.tree_map(lambda x: x.to(self.learner_device), batch)
            do_policy = self.critic_grad_updates % self.cfg.policy_frequency == 0
            self.critic_grad_updates += 1
            do_target = self.critic_grad_updates % self.cfg.target_frequency == 0
            info = self._update_fn(
                self.critic,
                self.actor,
                self.alpha,
                self.target_critic,
                self.reward_normalizer.model if self.reward_normalizer is not None else None,
                do_policy,
                do_target,
                batch,
                self.cfg
            )
            info = pytree.tree_map(lambda x: x.detach().cpu().numpy(), info)
        return info

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.pt")
        self.critic.save(checkpoint_dir / "critic.pt")
        self.target_critic.save(checkpoint_dir / "target_critic.pt")
        self.alpha.save(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.pt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.pt")
        self.critic.load(checkpoint_dir / "critic.pt")
        self.target_critic.load(checkpoint_dir / "target_critic.pt")
        self.alpha.load(checkpoint_dir / "alpha.pt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.load(checkpoint_dir / "reward_normalizer.pt")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        onnx_dir = Path(onnx_dir)
        self.actor.save_onnx(
            onnx_dir / "policy.onnx",
            [(1, self.actor_observation_dim)],      
        )
