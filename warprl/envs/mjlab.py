from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import RenderFrame
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from .types import F32NDArray, NDArray

SIM2REAL_TASK_ID = (
    "Unitree-Go2-Flat",
    "Unitree-Go2-Rough",
    "Unitree-A2-Flat",
    "Unitree-A2-Rough",
    "Unitree-As2-Flat",
    "Unitree-As2-Rough",
    "Unitree-G1-Flat",
    "Unitree-G1-Rough",
    "Unitree-G1-23Dof-Flat",
    "Unitree-G1-23Dof-Rough",
    "Unitree-H1_2-Flat",
    "Unitree-H1_2-Rough",
    "Unitree-H2-Flat",
    "Unitree-H2-Rough",
    "Unitree-R1-Flat",
    "Unitree-R1-Rough",
)



def _register_unitree_rl_mjlab_tasks(task_id: str) -> None:
    """Register vendored Unitree RL MjLab tasks on demand."""
    if not task_id.startswith("Unitree-"):
        return

    project_root = Path(__file__).resolve().parents[2]
    unitree_root = project_root / "third_party" / "unitree_rl_mjlab"
    if not unitree_root.is_dir():
        raise FileNotFoundError(
            "Unitree RL MjLab tasks require "
            f"{unitree_root}, but that directory does not exist."
        )

    unitree_root_str = str(unitree_root)
    if unitree_root_str not in sys.path:
        sys.path.insert(0, unitree_root_str)

    try:
        importlib.import_module("src.tasks")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to register Unitree RL MjLab tasks. The vendored project "
            "requires mjlab==1.2.0 and mujoco-warp==3.5.0; use its matching "
            "environment or port the tasks before using Unitree-* task IDs."
        ) from exc


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data



class MjlabVectorEnv(VectorEnv[F32NDArray, F32NDArray, F32NDArray]):
    """Gymnasium VectorEnv wrapping mjlab's ManagerBasedRlEnv for FlashSAC.

    Uses auto_reset=False so we can capture the true terminal observation before
    resetting. This populates infos["final_obs"] correctly for off-policy TD
    bootstrapping on truncated episodes — fixing the known limitation in the
    IsaacLab wrapper where terminal obs is unavailable.

    Observations are flattened from mjlab's dict format:
    - If both "actor" and "critic" groups exist: concatenated as [actor | critic],
      with actor_observation_size exposing the actor-visible prefix length.
    - Otherwise: the single group is used as-is.

    Actions are passed through unchanged (mjlab action terms handle scaling internally).
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        seed: int,
        device: str = "cuda:0",
        to_numpy: bool = True,
        render_mode: str | None = None
    ) -> None:
        import mjlab.tasks  # noqa: F401  # populates the task registry via side effects
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        _register_unitree_rl_mjlab_tasks(task_id)
        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.auto_reset = False  # we handle resets to preserve the terminal obs

        self._env = ManagerBasedRlEnv(
            cfg=env_cfg,
            device=device,
            render_mode=render_mode,
        )
        self._device = device
        self._to_numpy = to_numpy
        self.render_mode = render_mode
        self.num_envs = num_envs
        self._configure_spaces()

    def _configure_spaces(self) -> None:
        """Configure the flattened observation and action spaces."""
        spaces = self._env.single_observation_space.spaces
        if "actor" in spaces:
            self._actor_obs_key = "actor"
        elif len(spaces) == 1:
            self._actor_obs_key = next(iter(spaces))
        else:
            raise ValueError(
                "MJLab observations must contain an 'actor' group when "
                f"multiple groups are present, got {tuple(spaces)}"
            )

        actor_space = spaces[self._actor_obs_key]
        if actor_space.shape is None:
            raise ValueError("MJLab actor observations must have a fixed shape")

        self.actor_observation_size = int(np.prod(actor_space.shape))
        self.asymmetric_obs = (
            self._actor_obs_key == "actor" and "critic" in spaces
        )

        critic_group_size = 0
        if self.asymmetric_obs:
            critic_space = spaces["critic"]
            if critic_space.shape is None:
                raise ValueError("MJLab critic observations must have a fixed shape")
            critic_group_size = int(np.prod(critic_space.shape))

        flat_obs_size = self.actor_observation_size + critic_group_size
        self.critic_observation_size = flat_obs_size
        self.obs_size = (flat_obs_size,)
        self.action_size = tuple(self._env.single_action_space.shape)

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_size, dtype=np.float32
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        # NOTE: The Gym action space is unbounded. The policy emits actions in
        # [-1, 1], which this wrapper forwards to MJLab without rescaling or clipping.
        self.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = dict(getattr(self._env, "metadata", {}))
        self.metadata["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP

    def _flatten_obs(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> Union[F32NDArray, torch.Tensor]:
        actor_obs = obs_dict[self._actor_obs_key].reshape(self.num_envs, -1)
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"].reshape(self.num_envs, -1)
            flat_obs = torch.cat((actor_obs, critic_obs), dim=-1)
        else:
            flat_obs = actor_obs

        if self._to_numpy:
            return np.asarray(
                recursive_to_numpy(flat_obs), dtype=np.float32
            )
        return flat_obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Union[F32NDArray, torch.Tensor], dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        env_info: dict[str, Any] = {
            "actor_observation_size": self.actor_observation_size,
            "asymmetric_obs": self.asymmetric_obs,
        }
        return self._flatten_obs(obs_dict), env_info

    def step(
        self,
        actions: Union[F32NDArray, torch.Tensor],
    ) -> tuple[
        Union[F32NDArray, torch.Tensor],
        Union[F32NDArray, torch.Tensor],
        Union[NDArray, torch.Tensor],
        Union[NDArray, torch.Tensor],
        dict[str, Any],
    ]:
        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions).float().to(self._device)
        else:
            actions_t = actions.to(self._device)

        obs_dict, rewards, terminateds, truncateds, extras = self._env.step(
            actions_t)

        rewards_np = np.asarray(
            recursive_to_numpy(rewards), dtype=np.float32
        )

        # Capture terminal obs BEFORE resetting done envs
        terminal_obs = self._flatten_obs(obs_dict)

        # Reset done envs; mjlab raises RuntimeError on the next step() if we skip this.
        # reset() recomputes obs for ALL envs: done envs get fresh state, non-done envs
        # are unchanged — so the returned buf is already the correct next obs.
        dones = terminateds | truncateds
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            reset_obs_dict, _ = self._env.reset(env_ids=done_ids)
            next_obs = self._flatten_obs(reset_obs_dict)
        else:
            next_obs = terminal_obs

        infos: dict[str, Any] = {
            "final_obs": terminal_obs,
            "actor_observation_size": self.actor_observation_size,
            "asymmetric_obs": self.asymmetric_obs,
        }

        # Preserve mjlab's per-reward-term logs; eval_env owns episode return/length statistics.
        raw_log = extras.get("log") or {}
        episode_info: dict[str, Any] = {
            k: float(v.mean().item()) if isinstance(v, torch.Tensor) else v
            for k, v in raw_log.items()
        }
        if episode_info:
            infos["episode_info"] = episode_info

        if self._to_numpy:
            terminateds_out, truncateds_out, infos = recursive_to_numpy(
                (terminateds, truncateds, infos)
            )
            rewards_out: Union[F32NDArray, torch.Tensor] = rewards_np
        else:
            terminateds_out = terminateds
            truncateds_out = truncateds
            rewards_out = rewards

        return (
            next_obs,
            rewards_out,
            terminateds_out,
            truncateds_out,
            infos,
        )

    def close(self, **kwargs: Any) -> None:
        if hasattr(self, "_env"):
            self._env.close()

    def render(self) -> list[RenderFrame] | None:
        if self.render_mode != "rgb_array":
            return None
        image = self._env.render()
        return None if image is None else [image]



def make_mjlab_env(
    task_id: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    render_mode: str | None = None
) -> MjlabVectorEnv:
    return MjlabVectorEnv(
        task_id=task_id,
        num_envs=num_envs,
        seed=seed,
        device=device,
        render_mode=render_mode
    )
