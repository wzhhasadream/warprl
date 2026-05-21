import nnxrl.utils.logger as wandb
from flax import nnx
from typing import Literal
from nnxrl.agents.rainbowsac import TrainState
from nnxrl.model import (
    Alpha,
    FlashSACDoubleCritic,
    FlashSACActor,
    project_normalized_parameters,
)
from nnxrl.env import make_venv_env
from nnxrl.utils import ReplayBuffer, evaluate_policy
import time
import numpy as np
import jax
import optax
import tyro
import gymnasium as gym
import dataclasses


@dataclasses.dataclass
class Args:
    env_id: str = "Ant-v4"
    env_type: Literal['mujoco', 'myosuite', 'dmc',
                      'humanoid_bench'] = 'mujoco'
    seed: int = 1
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    target_frequency: int = 1
    learning_starts: int = int(5e3)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 512
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    target_entropy: float = 0  # will be set automatically
    target_sigma: float = 0.15
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    num_q: int = 2
    num_head: int = 100
    normalize_parameters: Literal[True, False] = False

    action_repeat: int = 1
    grad_step_per_env_step: int = 1

    eval_frequency: int = 1e4
    eval_episode: int = 10

    decay_step: int = 80_000


def main():
    print("🚀 sac training")
    print("=" * 60)

    args = tyro.cli(Args)
    if args.env_type == 'mujoco':
        args.action_repeat = 1
    np.random.seed(args.seed)

    envs, eval_envs = make_venv_env(args.env_id, args.env_type, args.num_envs, action_repeat=args.action_repeat, seed=args.seed)

    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))
    actor_obs_dim = obs_dim
    if getattr(envs, 'asymmetric_obs', False):
        actor_obs_dim = envs.actor_observation_size
    obs, _ = envs.reset(seed=args.seed)
    args.target_entropy = 0.5 * action_dim * np.log(
        2.0 * np.pi * np.e * args.target_sigma ** 2
    )

    wandb.init(project='rainbowsac', config=vars(args), name=f'{args.env_id}')

    rngs = nnx.Rngs(args.seed)
    actor = FlashSACActor(
        actor_obs_dim, action_dim, rngs.fork(),
            hidden_dim=args.actor_hidden_dim,
            num_blocks=args.actor_num_blocks,
            action_high=envs.single_action_space.high,
            action_low=envs.single_action_space.low,
        )
    critic = FlashSACDoubleCritic(
        obs_dim,
        action_dim,
        rngs.fork(split=args.num_q),
        hidden_dim=args.critic_hidden_dim,
        num_blocks=args.critic_num_blocks,
        num_head=args.num_head
    )
    if args.normalize_parameters:
        project_normalized_parameters(actor)
        project_normalized_parameters(critic)
    alpha = Alpha() 
    actor_opt = nnx.Optimizer(actor, optax.adam(args.policy_lr))
    critic_opt = nnx.Optimizer(critic, optax.adam(args.q_lr))
    alpha_opt = nnx.Optimizer(alpha, optax.adam(
        args.policy_lr)) 

    rb = ReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        args.buffer_size,
        n_envs=args.num_envs,
        linear_decay_steps=args.decay_step
    )


    ts = TrainState.create(actor, critic, actor_opt,
                           critic_opt, alpha=alpha, alpha_opt=alpha_opt)
    start_time = time.time()

    jit_update = ts.make_update_fn(args)
    action_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed))

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(args.num_envs)])
        else:
            actions = ts.get_exploration_action(
                obs=obs, key=jax.random.fold_in(action_key, global_step))
            actions = np.asarray(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        real_next_obs = next_obs.copy()
        if truncations.any():
            real_next_obs[truncations] = infos["final_obs"][truncations]

        rb.add(
            obs,
            actions,
            rewards,
            real_next_obs,
            terminations,
        )

        if global_step >= args.learning_starts:
            big_batch = rb.sample(
                args.batch_size * args.grad_step_per_env_step)
            ts, info = jit_update(
                ts, big_batch, jax.random.fold_in(
                    update_key, global_step)
            )
            if global_step % args.eval_frequency == 0:
                def policy(obs): return ts.get_action(obs)
                wall_time = time.time() - start_time
                eval_info = evaluate_policy(eval_envs, policy, args.eval_episode)
                wandb.log(
                    {**info, **eval_info, "eval/wall_time": wall_time}, global_step)
        obs = next_obs

    envs.close()
    def policy(obs): return ts.get_action(obs)
    final_info = evaluate_policy(eval_envs, policy, args.eval_episode)
    wall_time = time.time() - start_time
    wandb.log({**final_info, "eval/wall_time": wall_time},
              args.total_timesteps)
    wandb.finish()


if __name__ == "__main__":
    main()
