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
from nnxrl.env import make_venv_env
from nnxrl.utils import evaluate_policy, replace_truncated_next_obs, GPUReplayBuffer, Batch
import time
import numpy as np
import jax
from optax import adam, cosine_decay_schedule
import tyro
import dataclasses


@dataclasses.dataclass
class Args:
    env_id: str = "BallInCup"
    env_type: Literal['playground', 'issaclab', 'maniskill'] = 'playground'
    seed: int = 1
    num_envs: int = 1024
    total_timesteps: int = 50_000_896
    buffer_size: int = int(10e6)
    policy_frequency: int = 2
    target_frequency: int = 1
    learning_starts: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 2048
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    target_entropy: float = 0  # will be set automatically
    target_sigma: float = 0.15
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    num_q: int = 2
    num_head: int = 101
    exploration_noise: float = 0.5
    normalize_parameters: Literal[True, False] = False
    asymmetric_obs: Literal[True, False] = False

    action_repeat: int = 1
    grad_step_per_env_step: int = 2

    eval_frequency: int = 5_496_832
    eval_episode: int = 50

    decay_step: int = 40_000
    coupled_flow: Literal[True, False] = False
    num_ode: int = 1
    num_step: int = 1


def main():
    print("🚀 sac training")
    print("=" * 60)

    args = tyro.cli(Args)
    if args.env_type == 'myosuite':
        args.eval_episode = 100
    np.random.seed(args.seed)

    envs, eval_envs = make_venv_env(
        args.env_id, args.env_type, args.num_envs, action_repeat=args.action_repeat, seed=args.seed)

    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))
    actor_obs_dim = obs_dim
    asymmetric_obs = False
    if getattr(envs, 'asymmetric_obs', False):
        actor_obs_dim = envs.actor_observation_size
        asymmetric_obs = True
    args.asymmetric_obs = asymmetric_obs
    obs, _ = envs.reset(seed=args.seed)
    args.target_entropy = 0.5 * action_dim * np.log(
        2.0 * np.pi * np.e * args.target_sigma ** 2
    )

    wandb.init(project='rainbowsac_gpu',
               config=vars(args), name=f'{args.env_id}')

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
        project_param(critic)
        project_param(actor)
    alpha = Alpha()
    actor_opt = nnx.Optimizer(actor, adam(args.policy_lr))
    critic_opt = nnx.Optimizer(critic, adam(cosine_decay_schedule(args.q_lr, (args.total_timesteps * args.grad_step_per_env_step) / args.num_envs)))
    alpha_opt = nnx.Optimizer(alpha, adam(args.policy_lr))

    rb = GPUReplayBuffer.create(
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        args.buffer_size,
        n_envs=args.num_envs,
        linear_decay_steps=args.decay_step
    )

    ts = TrainState.create(actor, critic, actor_opt,
                           critic_opt, alpha=alpha, alpha_opt=alpha_opt, pre_noise=jax.random.normal(jax.random.PRNGKey(args.seed), (args.num_envs, action_dim)))
    start_time = time.time()

    def eval_and_log(ts, global_step):
        def policy(obs):
            return ts.get_action(obs)
        wall_time = time.time() - start_time
        info = evaluate_policy(eval_envs, policy, args.eval_episode)
        wandb.log({**info, "eval/wall_time": wall_time}, global_step)

    jit_train = ts.make_train_step(args)
    action_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)

    for global_step in range(0, args.total_timesteps, args.num_envs):
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

        real_next_obs = replace_truncated_next_obs(
            next_obs, truncations, infos)

        transition = Batch(obs, actions, rewards, terminations, real_next_obs)
        ts, rb, info = jit_train(ts, rb, transition, jax.random.fold_in(update_key, global_step))
        obs = next_obs

    envs.close()
    eval_and_log(ts, args.total_timesteps)
    wandb.finish()


if __name__ == "__main__":
    main()
