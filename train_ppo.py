from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import warprl.utils.logger as wandb
from warprl.envs import create_envs
from warprl.buffers.on_policy.types import RolloutTransition
from warprl.utils import (
    bootstrap_timeout_rewards,
    evaluate_policy,
    record_video
)
import tqdm
import dataclasses


@dataclass
class Args:
    profile: Literal["auto", "cpu_sim", "gpu_sim"] = "auto"

    env_id: str = "Hopper-v4"
    env_type: Literal['mujoco', 'myosuite', 'dmc',
                      'humanoid_bench', 'playground', 'maniskill', 'isaaclab', 'mjlab'] = 'mujoco'
    backend: Literal["jax", "torch"] = "torch"
    algo: Literal["ppo", "spo"] = "ppo"
    seed: int = 0
    ##############    depends on env_type ###################
    num_envs: int | None = None
    num_eval_envs: int | None = None
    eval_episode: int | None = None
    compute_type: Literal["float32", "bfloat16"] | None = None
    total_timesteps: int | None = None
    rollout_steps: int | None = None
    num_mini_batches: int | None = None
    num_epochs: int | None = None
    ##########################################################
    init_std: float = 1
    action_repeat: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-3
    max_grad_norm: float = 1
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    normalize_advantages: bool = True
    save_onnx: bool = False
    record_video: bool = False
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_value: bool = False


    eval_frequency: int | None = None
    log_frequency: int | None = None
    num_rollout: int | None = None
    render_mode: str | None = dataclasses.field(init=False, default=None)
    asymmetric_obs: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        from warprl.agents.config.ppo import resolve_profile
        resolve_profile(self)
        self.num_rollout = self.total_timesteps // (self.num_envs * self.rollout_steps)

        self.render_mode = "rgb_array" if self.record_video else None
    
        if self.eval_frequency is None:
            self.eval_frequency = self.num_rollout // 20

        if self.log_frequency is None:
            self.log_frequency = self.num_rollout // 50

    


def main() -> None:
    args = tyro.cli(Args)
    np.random.seed(args.seed)
    train_env, eval_env, record_env = create_envs(
        env_name=args.env_id,
        env_type=args.env_type,
        seed=args.seed,
        num_train_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        render_mode=args.render_mode,
    )
    if args.backend == "jax":
        from warprl.agents.ppo.jax import PPOAgent
    else:
        from warprl.agents.ppo.torch import PPOAgent

    agent = PPOAgent(train_env, args)
    wandb.init(
        project=args.env_type,
        name=args.env_id,
        config={**vars(args), **agent.observation_debug_info},
        dir="Results/ppo"
    )

    obs, _ = train_env.reset(seed=args.seed)

    def eval_and_log(agent, rollout_idx):
        info = evaluate_policy(
            agent.get_action,
            eval_env,
            args.eval_episode, 
            args.env_type
        )
        wandb.log(info, rollout_idx * args.num_envs * args.rollout_steps)
        if args.record_video:
            video = record_video(agent.get_action, record_env, args.env_type)
            wandb.video(video, rollout_idx * args.num_envs * args.rollout_steps)
        if args.save_onnx:
            wandb.save_onnx(agent, rollout_idx * args.num_envs * args.rollout_steps)

    eval_and_log(agent, 0)
    for rollout_idx in tqdm.tqdm(range(1, args.num_rollout + 1), smoothing=0.1, mininterval=0.5):
        for _ in range(args.rollout_steps):
            actions, values, log_probs, actions_mean, actions_std = agent.sample_action_and_value(obs)
            next_obs, rewards, terminated, truncated, infos = train_env.step(actions)
            timeouts = np.logical_and(truncated, ~terminated)
            rewards = bootstrap_timeout_rewards(rewards, timeouts, args.gamma, next_obs, agent.get_value, infos)

            agent.process_transition(
                RolloutTransition(
                    observations=obs,
                    actions=actions,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    values=values,
                    log_probs=log_probs,
                    actions_mean=actions_mean,
                    actions_std=actions_std
                )
            )

            obs = next_obs
        info = agent.update(obs)
        if rollout_idx % args.log_frequency == 0:
            wandb.log(info, rollout_idx * args.num_envs * args.rollout_steps)
        if rollout_idx % args.eval_frequency == 0:
            eval_and_log(agent, rollout_idx)

    eval_and_log(agent, args.num_rollout)
    wandb.finish()
    if record_env is not train_env and record_env is not eval_env:
        record_env.close()
    if eval_env is not train_env:
        eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
