import nnxrl.utils.logger as wandb
from flax import nnx
from typing import Literal
from nnxrl.agents.rainbowsac import TrainState
from nnxrl.model import (
    Alpha,
    FlashSACDoubleCritic,
    CoupleFlowActor,
    FlashSACActor,
    project_param
)
from nnxrl.env import create_envs
from nnxrl.utils import ReplayBuffer, evaluate_policy, replace_truncated_next_obs, RewardNormalizer
import time
import numpy as np
import jax
from optax import adam, cosine_decay_schedule
import tyro
import dataclasses

@dataclasses.dataclass
class Args:
    env_id: str = "h1-run-v0"
    env_type: Literal['mujoco', 'myosuite', 'dmc',
                      'humanoid_bench', 'playground'] = 'humanoid_bench'
    seed: int = 1
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    target_frequency: int = 1
    learning_starts: int = int(1e4)
    gamma: float = 0.99
    tau: float = 1e-2
    batch_size: int = 512
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
    exploration_noise: float = 0.7
    normalize_parameters: Literal[True, False] = False
    normalize_rewards: Literal[True, False] = True
    asymmetric_obs: Literal[True, False] = False
    loss_type: Literal["quantile_loss", "ce_loss"] = "ce_loss"

    action_repeat: int = 1
    grad_step_per_env_step: int = 1

    eval_frequency: int = 2e4
    eval_episode: int = 10
    log_frequency: int = 2000

    decay_step: int = 80_000
    coupled_flow: Literal[True, False] = False 
    num_ode: int = 1
    num_step: int = 1



def main():
    print("🚀 sac training")
    print("=" * 60)

    args = tyro.cli(Args)
    assert args.num_envs == 1, "num_envs only supports 1 for cpu simulation"
    if args.env_type == 'myosuite':
        args.eval_episode = 100
    np.random.seed(args.seed)

    num_critic_updates = int(args.total_timesteps / args.num_envs * args.grad_step_per_env_step)

    envs, eval_envs = create_envs(
        args.env_id, args.env_type, num_train_envs=args.num_envs, action_repeat=args.action_repeat, seed=args.seed)

    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))
    actor_obs_dim = obs_dim
    asymmetric_obs = False
    if getattr(envs, 'asymmetric_obs', False):
        actor_obs_dim = envs.actor_observation_size
        asymmetric_obs = True
    args.asymmetric_obs = asymmetric_obs
    obs, _ = envs.reset(seed=args.seed)
    args.target_entropy = 0.5 * action_dim * np.log(2 * np.pi * np.e * 0.15 ** 2) 

    wandb.init(project='rainbowsac_falsh', config=vars(args), name=f'{args.env_id}')

    rngs = nnx.Rngs(args.seed)
    if args.coupled_flow:
        actor = CoupleFlowActor(
        actor_obs_dim, action_dim, rngs.fork(),
            hidden_dim=args.actor_hidden_dim,
            num_blocks=args.actor_num_blocks,
            action_high=envs.single_action_space.high,
            action_low=envs.single_action_space.low,
            num_ode=args.num_ode,
            num_step=args.num_step
        )
    else:
        actor = FlashSACActor(
        actor_obs_dim, action_dim, rngs.fork(),
            hidden_dim=args.actor_hidden_dim,
            num_blocks=args.actor_num_blocks,
            action_high=envs.single_action_space.high,
            action_low=envs.single_action_space.low
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
        project_param(critic)
        project_param(actor)
    alpha = Alpha() 
    actor_opt = nnx.Optimizer(actor, adam(cosine_decay_schedule(args.policy_lr, num_critic_updates, args.policy_lr / args.end_lr)))
    critic_opt = nnx.Optimizer(critic, adam(cosine_decay_schedule(args.q_lr, num_critic_updates, args.q_lr / args.end_lr)))
    alpha_opt = nnx.Optimizer(alpha, adam(cosine_decay_schedule(args.policy_lr, num_critic_updates, args.policy_lr / args.end_lr))) 
    reward_normalizer = (
        RewardNormalizer.create(args.num_envs, args.gamma)
        if args.normalize_rewards
        else None
    )
    rb = ReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        args.buffer_size,
        n_envs=args.num_envs,
        linear_decay_steps=args.decay_step
    )


    ts = TrainState.create(actor, critic, actor_opt,
                           critic_opt, alpha=alpha, alpha_opt=alpha_opt, pre_noise=jax.random.normal(jax.random.PRNGKey(args.seed), (args.num_envs, action_dim)), reward_normalizer=reward_normalizer)
    start_time = time.time()

    def eval_and_log(ts, global_step):
        def policy(obs):
            return ts.get_action(obs)
        wall_time = time.time() - start_time
        info = evaluate_policy(eval_envs, policy, args.eval_episode)
        wandb.log({**info, "eval/wall_time":wall_time}, global_step)

    jit_update = ts.make_update_fn(args)
    action_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)

    for global_step in range(0, args.total_timesteps):
        if global_step % args.eval_frequency == 0:
            eval_and_log(ts, global_step)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(args.num_envs)])
        else:
            ts, actions = ts.get_exploration_action(
                obs=obs,
                key=jax.random.fold_in(action_key, global_step),
                exploration_noise=args.exploration_noise
            )
            actions = np.asarray(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        if args.normalize_rewards:
            ts = ts.update_reward_normalizer(
                rewards, np.logical_or(terminations, truncations)
            )

        real_next_obs = replace_truncated_next_obs(next_obs, truncations, infos)

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
            if global_step % args.log_frequency == 0:
                wandb.log(info, global_step)
        obs = next_obs

    envs.close()
    eval_and_log(ts, args.total_timesteps)
    wandb.finish()


if __name__ == "__main__":
    main()
