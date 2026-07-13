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
from nnxrl.buffers import Transition
import numpy as np
import tyro
import dataclasses
import tqdm
@dataclasses.dataclass
class Args:
    profile: Literal["auto", "cpu_sim", "playground", "maniskill", 'isaaclab', 'mjlab'] = "auto"

    env_id: str = "humanoid-run"
    env_type: Literal['mujoco', 'myosuite', 'dmc',
                      'humanoid_bench', 'playground', 'maniskill', 'isaaclab', 'mjlab'] = 'dmc'

    seed: int = 1

    ##############    depends on env_type ###################
    num_envs: int | None = None
    total_timesteps: int | None = None
    buffer_size: int | None = None
    learning_starts: int | None = None        
    batch_size: int | None = None
    grad_step_per_interaction_step: int | None = None
    gamma: float | None = None
    decay_step: int | None = None
    compute_type: Literal["float32", "bfloat16"] | None = None
    n_step: int | None = None
    buffer_device: Literal["cpu", "cuda"] | None = None
    eval_episode: int | None = None
    num_eval_envs: int | None = None
    #######################################################
    eval_frequency: int | None = None
    log_frequency: int | None = None
    num_interaction_steps: int | None = None
    policy_frequency: int = 2
    target_frequency: int = 1
    tau: float = 1e-2
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    end_lr: float = 1.5e-4
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    num_q: int = 2
    num_head: int = 101
    normalize_parameters: Literal[True, False] = True
    normalize_rewards: Literal[True, False] = True
    use_bias: Literal[True, False] = False
    record_video: Literal[True, False] = True
    save_agent: Literal[True, False] = False
    save_onnx: Literal[True, False] = True
    loss_type: Literal["quantile_loss", "ce_loss"] = "ce_loss"
    log_path: str = "nnxrl"
    action_repeat: int = 1

    def __post_init__(self):
        resolve_profile(self)
        self.num_interaction_steps = self.total_timesteps // self.num_envs
        if self.eval_frequency is None:
            if self.env_type == 'isaaclab':
                # IsaacLab reuses train_envs for evaluation, so evaluate less frequently.
                self.eval_frequency = self.num_interaction_steps // 10
            else:
                self.eval_frequency = self.num_interaction_steps // 20

        if self.log_frequency is None:
            self.log_frequency = self.num_interaction_steps // 50




def main():
    print("🚀 RainBowsac training")
    print("=" * 60)

    args = tyro.cli(Args)
    np.random.seed(args.seed)

    wandb.init(project=args.log_path, name=f"{args.env_id}", config=vars(args))

    render_mode = "rgb_array" if args.record_video else None

    train_envs, eval_envs, record_envs = create_envs(
        args.env_id,
        args.env_type,
        num_train_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        action_repeat=args.action_repeat,
        seed=args.seed,
        render_mode=render_mode,
    )

    agent = RainbowSACAgent(train_envs, args)


    def eval_and_log(agent, global_step):
        def policy(obs):
            return agent.get_action(obs)
        info = evaluate_policy(policy, eval_envs, args.eval_episode, args.env_type)
        wandb.log(info, global_step)
        if args.record_video:
            videos = record_video(policy, record_envs, args.env_type)
            wandb.video(videos, global_step)
        if args.save_agent:
            wandb.save_agent(agent, global_step)
        if args.save_onnx:
            wandb.save_onnx(agent, global_step)

    eval_and_log(agent, 0)
    obs, info = train_envs.reset(seed=args.seed)
    for interaction_step in tqdm.tqdm(range(1, int(args.num_interaction_steps + 1)), smoothing=0.1, mininterval=0.5):
        if not agent.can_update:
            actions = train_envs.action_space.sample()
        else:
            actions = agent.get_exploration_action(obs)

        next_obs, rewards, terminations, truncations, infos = train_envs.step(
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
            if interaction_step % args.log_frequency == 0:
                wandb.log(info, interaction_step * args.num_envs)
        
            if interaction_step % args.eval_frequency == 0:
                eval_and_log(agent, interaction_step * args.num_envs)

        obs = next_obs

    eval_and_log(agent, args.total_timesteps)
    wandb.finish()
    train_envs.close()
    if eval_envs is not train_envs:
        eval_envs.close()
    if record_envs is not train_envs and record_envs is not eval_envs:
        record_envs.close()


if __name__ == "__main__":
    main()
