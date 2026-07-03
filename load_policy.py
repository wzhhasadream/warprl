import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import gymnasium as gym
import numpy as np
import tyro

from nnxrl.agents import RainbowSACAgent


Policy = Callable[[np.ndarray], np.ndarray]

DEFAULT_ENV_ID = "Isaac-Velocity-Flat-G1-v0"
DEFAULT_OBS_DIM = 123
DEFAULT_ACTION_DIM = 37
DEFAULT_TRAIN_NUM_ENVS = 1024
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = {
    "rainbowsac": PROJECT_ROOT / "ckpt/rainbowsac/7498752_ckpt",
    "flashsac": PROJECT_ROOT / "ckpt/flashsac/7498752_ckpt",
}


@dataclasses.dataclass
class Args:
    checkpoint: Literal["rainbowsac", "flashsac", "both"] = "both"
    checkpoint_path: str | None = None
    env_id: str = DEFAULT_ENV_ID
    seed: int = 0


class _PolicySpecEnv:
    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        num_envs: int = DEFAULT_TRAIN_NUM_ENVS,
    ):
        self.num_envs = num_envs
        self.asymmetric_obs = False
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )


def _make_policy_config(env_id: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        env_id=env_id,
        env_type="isaaclab",
        seed=seed,
        num_envs=DEFAULT_TRAIN_NUM_ENVS,
        total_timesteps=50_000_896,
        buffer_size=DEFAULT_TRAIN_NUM_ENVS,
        learning_starts=1,
        batch_size=2048,
        grad_step_per_interaction_step=2,
        gamma=0.99,
        decay_step=2_000,
        compute_type="bfloat16",
        n_step=1,
        buffer_device="cpu",
        eval_episode=1,
        policy_frequency=2,
        target_frequency=1,
        tau=1e-2,
        policy_lr=3e-4,
        q_lr=3e-4,
        end_lr=1.5e-4,
        critic_hidden_dim=256,
        critic_num_blocks=2,
        actor_hidden_dim=128,
        actor_num_blocks=2,
        num_q=2,
        num_head=101,
        normalize_parameters=True,
        normalize_rewards=True,
        use_bias=False,
        loss_type="ce_loss",
    )


def _resolve_checkpoint_path(checkpoint: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve()

    if checkpoint not in DEFAULT_CHECKPOINTS:
        raise ValueError(
            f"checkpoint must be one of {tuple(DEFAULT_CHECKPOINTS)}, got {checkpoint!r}"
        )

    return DEFAULT_CHECKPOINTS[checkpoint].resolve()


def _load_agent(checkpoint_path: Path, env_id: str, seed: int) -> RainbowSACAgent:
    cfg = _make_policy_config(env_id, seed)
    env = _PolicySpecEnv()
    agent = RainbowSACAgent(env, cfg)  # type: ignore[arg-type]
    agent.load(str(checkpoint_path))
    return agent


def load_policy(
    checkpoint: Literal["rainbowsac", "flashsac"] = "rainbowsac",
    *,
    checkpoint_path: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
) -> Policy:
    checkpoint_path_obj = _resolve_checkpoint_path(checkpoint, checkpoint_path)
    agent = _load_agent(checkpoint_path_obj, env_id, seed)

    def policy(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        if obs.shape[-1] != DEFAULT_OBS_DIM:
            raise ValueError(
                f"Expected observations with last dimension {DEFAULT_OBS_DIM}, got {obs.shape}"
            )
        actions = np.asarray(agent.get_action(obs), dtype=np.float32)
        return actions[0] if squeeze else actions

    return policy


def load_policies(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
) -> dict[str, Policy]:
    return {
        name: load_policy(name, env_id=env_id, seed=seed)  # type: ignore[arg-type]
        for name in DEFAULT_CHECKPOINTS
    }


def main() -> None:
    args = tyro.cli(Args)

    if args.checkpoint == "both":
        policies = load_policies(env_id=args.env_id, seed=args.seed)
        for name, policy in policies.items():
            action = policy(np.zeros(DEFAULT_OBS_DIM, dtype=np.float32))
            print(f"{name}: action_shape={action.shape}")
        return

    policy = load_policy(
        args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        env_id=args.env_id,
        seed=args.seed,
    )
    action = policy(np.zeros(DEFAULT_OBS_DIM, dtype=np.float32))
    print(f"{args.checkpoint}: action_shape={action.shape}")


if __name__ == "__main__":
    main()

# from load_policy import load_policy, load_policies

# policy = load_policy("rainbowsac")
# action = policy(obs)  # obs shape: (123,) or (B, 123), action shape: (37,) or (B, 37)

# policies = load_policies()
# rainbow_action = policies["rainbowsac"](obs)
# flash_action = policies["flashsac"](obs)