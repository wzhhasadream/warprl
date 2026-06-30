import nnxrl.utils.logger as wandb
from typing import Literal
from nnxrl.agents import RainbowSACAgent
from nnxrl.env import create_envs
from nnxrl.utils import (
    evaluate_policy, 
    replace_done_next_obs, 
    resolve_profile, 
    record_video
)
from nnxrl.buffer import Transition
import numpy as np
import tyro
import dataclasses

@dataclasses.dataclass
class Args:
    profile: Literal["auto", "cpu_sim", "playground", "maniskill", 'playground'] = "auto"

    env_id: str = "PickSingleYCB-v1"
    env_type: Literal['mujoco', 'myosuite', 'dmc',
                      'humanoid_bench', 'playground', 'maniskill', 'isaaclab'] = 'maniskill'

    seed: int = 1

    ##############    depends on env_type ###################
    num_envs: int | None = None
    total_timesteps: int | None = None
    buffer_size: int | None = None
    learning_starts: int | None = None        
    batch_size: int | None = None
    grad_step_per_env_step: int | None = None
    eval_frequency: int | None = None
    log_frequency: int | None = None
    gamma: float | None = None
    decay_step: int | None = None
    compute_type: Literal["float32", "bfloat16"] | None = None
    n_step: int | None = None
    buffer_type: Literal["jax", "numpy"] | None = None
    #######################################################

    policy_frequency: int = 2
    target_frequency: int = 1
    eval_episode: int = 10
    tau: float = 1e-2
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    end_lr: float = 1.5e-4
    target_entropy: float = 0  # will be set automatically
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    num_q: int = 2
    num_head: int = 101
    normalize_parameters: Literal[True, False] = True
    normalize_rewards: Literal[True, False] = True
    asymmetric_obs: Literal[True, False] = False   # will be set automatically
    use_bias: Literal[True, False] = False
    record_video: Literal[True, False] = False
    save_agent: Literal[True, False] = False
    loss_type: Literal["quantile_loss", "ce_loss"] = "ce_loss"
    log_path: str = "final"
    action_repeat: int = 1
    coupled_flow: Literal[True, False] = False 
    num_ode: int = 1
    num_step: int = 1



def main():
    print("🚀 RainBowsac training")
    print("=" * 60)

    args = tyro.cli(Args)
    args = resolve_profile(args)
    np.random.seed(args.seed)

    envs, eval_envs, record_envs = create_envs(
        args.env_id, args.env_type, num_train_envs=args.num_envs, action_repeat=args.action_repeat, seed=args.seed)

    agent = RainbowSACAgent(envs, args)

    def eval_and_log(agent, global_step):
        def policy(obs):
            return agent.get_action(obs)
        info = evaluate_policy(eval_envs, policy, args.env_type, args.eval_episode)
        wandb.log(info, global_step)
        if args.record_video:
            videos = record_video(policy, record_envs)
            wandb.video(videos, global_step)
        if args.save_agent:
            wandb.save_agent(agent, global_step)


    for global_step in range(0, args.total_timesteps, args.num_envs):
        if global_step % args.eval_frequency < args.num_envs:
            eval_and_log(agent, global_step)
        if agent.can_update:
            actions = envs.action_space.sample()
        else:
            actions = agent.get_action(obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        dones = np.logical_or(terminations, truncations)

        mask = truncations if args.env_type == 'myosuite' else dones
        real_next_obs = replace_done_next_obs(next_obs, mask, infos)

        transitions = Transition(
            observations=obs,
            actions=actions,
            rewards=rewards,
            truncations=truncations,
            terminations=terminations,
            next_observations=real_next_obs
        )

        agent.process_transition(transitions)
        if agent.can_update:
            info = agent.update()
            if global_step % args.log_frequency < args.num_envs:
                wandb.log(info, global_step)

        obs = next_obs

    envs.close()
    eval_and_log(agent, args.total_timesteps)
    eval_envs.close()
    record_envs.close()
    wandb.finish()


if __name__ == "__main__":
    main()
