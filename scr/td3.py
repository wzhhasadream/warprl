import nnxrl.utils.logger as wandb
from flax import nnx
from typing import Literal, Sequence
from nnxrl.utils import ReplayBuffer, RMS, evaluate_policy
from nnxrl.agents.td3 import update_td3, TrainState
from nnxrl.model import EnsembleCritic, TanhDetActor, project_normalized_parameters
from nnxrl.env import load_env, replace_truncated_next_obs
import time
import numpy as np
import jax.numpy as jnp
import jax
import optax
import tyro
import gymnasium as gym
import dataclasses


@dataclasses.dataclass
class Args:
    env_id: str = "Ant-v4"
    env_type: Literal['mujoco', 'dmc', 'humanoid_bench', 'myosuite'] = 'mujoco'
    seed: int = 0
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    learning_starts: int = int(25e3)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    critic_hidden_dim: Sequence[int] = (512, 512, 512)
    critic_ln: Literal[True, False] = True
    actor_hidden_dim: Sequence[int] = (512, 512, 512)
    actor_ln: Literal[True, False] = True
    num_q: int = 2
    num_head: int = 100
    normalize_parameters: Literal[True, False] = True

    action_repeat: int = 2
    normalize_observation: Literal[True, False] = True
    simba: Literal[True, False] = False
    grad_step_per_env_step: int = 1

    eval_frequency: int = int(1e4)
    eval_episode: int = 10

    decay_step: int = 0


def main():
    print("🚀 td3 training")
    print("=" * 60)

    args = tyro.cli(Args)
    if args.env_type == 'mujoco':
        args.action_repeat = 1

    np.random.seed(args.seed)
    env = [load_env(args.env_id, args.env_type, args.action_repeat, args.seed + i)
           for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env, autoreset_mode='SameStep')
    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))

    wandb.init(project='td3', config=vars(args), name=f'{args.env_id}')

    rngs = nnx.Rngs(args.seed)
    actor = TanhDetActor(
        obs_dim,
        action_dim,
        rngs.fork(),
        hidden_dim=args.actor_hidden_dim,
        action_high=envs.single_action_space.high,
        action_low=envs.single_action_space.low,
        simba_encoder=args.simba,
        layer_norm=args.actor_ln)

    critic = EnsembleCritic(
        obs_dim,
        action_dim,
        rngs.fork(split=args.num_q),
        hidden_dim=args.critic_hidden_dim,
        simba_encoder=args.simba,
        layer_norm=args.critic_ln)

    if args.normalize_parameters:
        project_normalized_parameters(actor)
        project_normalized_parameters(critic)

    actor_opt = nnx.Optimizer(actor, optax.adam(args.policy_lr))
    critic_opt = nnx.Optimizer(critic, optax.adam(args.q_lr))

    rb = ReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        args.buffer_size,
        n_envs=args.num_envs,
        linear_decay_steps=args.decay_step
    )
    if args.normalize_observation:
        rms = RMS.create(envs.single_observation_space.shape)
    else:
        rms = None

    ts = TrainState.create(
        actor=actor,
        critic=critic,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        rms=rms,
    )
    start_time = time.time()


    jit_update = nnx.jit(lambda ts, big_batch, key: update_td3(
        ts, args, key, big_batch), donate_argnums=0)

    obs, _ = envs.reset(seed=args.seed)
    step_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(args.num_envs)])
        else:
            ts, actions = ts.get_exploration_action(obs, args.exploration_noise, jax.random.fold_in(step_key, global_step))
            actions = np.asarray(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        real_next_obs = replace_truncated_next_obs(next_obs, truncations, infos)

        rb.add(
            obs,
            actions,
            rewards,
            real_next_obs,
            terminations,
            truncations,
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
                eval_info = evaluate_policy(load_env(
                    args.env_id, args.env_type, args.action_repeat, args.seed + 100), policy, args.eval_episode)
                wandb.log({**info, **eval_info, "eval/wall_time": time.time() - start_time}, global_step)
        obs = next_obs

    envs.close()
    def policy(obs): return ts.get_action(obs)
    final_info = evaluate_policy(load_env(
        args.env_id, args.env_type, args.action_repeat, args.seed + 100), policy, args.eval_episode)
    wandb.log({**final_info, "eval/wall_time": time.time() - start_time}, args.total_timesteps)
    wandb.finish()


if __name__ == "__main__":
    main()
