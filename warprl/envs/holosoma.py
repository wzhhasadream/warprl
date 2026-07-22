from __future__ import annotations

import dataclasses
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


HOLOSOMA_TASKS = [
    "g1-29dof",
    "g1-29dof-fast-sac",
    "t1-29dof",
    "t1-29dof-fast-sac",
    "g1-29dof-wbt",
    "g1-29dof-wbt-fast-sac",
    "g1-29dof-wbt-w-object",
    "g1-29dof-wbt-fast-sac-w-object",
]



def recursive_to_numpy(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {key: recursive_to_numpy(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(value) for value in data)
    return data


class HoloSomaVectorEnv(VectorEnv):
    """Expose a HoloSoma manager task through the Gymnasium vector API."""

    def __init__(
        self,
        task: Any,
        simulation_app: Any | None,
        close_simulation_app: Callable[[Any], None],
        to_numpy: bool = True,
        render_mode: str | None = None,
    ) -> None:
        self.task = task
        self._simulation_app = simulation_app
        self._close_simulation_app = close_simulation_app
        self._closed = False
        self.num_envs = int(task.num_envs)
        self.device = task.device
        self.to_numpy = to_numpy
        self.render_mode = render_mode

        initial_obs_dict = task.reset_all()
        initial_obs = self._pack_observations(initial_obs_dict)
        self._pending_reset_obs: np.ndarray | None = initial_obs

        self.actor_observation_size = initial_obs_dict["actor_obs"].shape[-1]
        self.asymmetric_obs = True
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(initial_obs.shape[-1],),
            dtype=np.float32,
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

        self.action_boundaries = self._action_boundaries()
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_boundaries.shape[0],),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | torch.Tensor, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        if self._pending_reset_obs is not None:
            obs = self._pending_reset_obs
            self._pending_reset_obs = None
        else:
            obs = self._pack_observations(self.task.reset_all())

        return obs, self._reset_info()

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        torch_actions = self._to_torch_actions(actions)
        obs_dict, rewards, reset_buf, extras = self.task.step(
            {"actions": torch_actions}
        )
        obs = self._pack_observations(obs_dict)
        truncated = extras["time_outs"].bool()
        dones = reset_buf.bool()
        terminated = dones & ~truncated

        final_obs_dict = extras.get("final_observations")
        final_obs = (
            self._pack_observations(final_obs_dict)
            if final_obs_dict is not None
            else obs
        )
        infos = self._reset_info()
        infos.update(
            {
                "time_outs": truncated,
                "final_obs": final_obs,
                "episode": extras.get("episode", {}),
            }
        )
        if self.to_numpy:
            obs, rewards, terminated, truncated, infos = recursive_to_numpy(
                (obs, rewards, terminated, truncated, infos)
            )
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self.task.simulator.close()
        if self._simulation_app is not None:
            self._close_simulation_app(self._simulation_app)

    def render(self) -> np.ndarray | torch.Tensor | None:
        if self.render_mode != "rgb_array":
            return None

        simulator = self.task.simulator
        simulator.render_sensors()
        frames = simulator.get_camera_data("overview", "rgb", device="cpu")
        return recursive_to_numpy(frames) if self.to_numpy else frames

    def _reset_info(self) -> dict[str, Any]:
        return {
            "actor_observation_size": self.actor_observation_size,
            "asymmetric_obs": self.asymmetric_obs,
        }

    def _pack_observations(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> np.ndarray | torch.Tensor:
        observations = torch.cat(
            (obs_dict["actor_obs"], obs_dict["critic_obs"]), dim=-1
        )
        return recursive_to_numpy(observations) if self.to_numpy else observations

    def _action_boundaries(self) -> np.ndarray:
        robot_config = self.task.robot_config
        lower = np.asarray(robot_config.dof_pos_lower_limit_list, dtype=np.float32)
        upper = np.asarray(robot_config.dof_pos_upper_limit_list, dtype=np.float32)
        default = np.asarray(
            [
                robot_config.init_state.default_joint_angles[name]
                for name in robot_config.dof_names
            ],
            dtype=np.float32,
        )
        action_scale = np.asarray(
            robot_config.control.action_scale, dtype=np.float32
        )
        if np.any(action_scale == 0.0):
            raise ValueError("HoloSoma action_scale must be non-zero")

        boundaries = np.maximum(np.abs(lower - default), np.abs(upper - default))
        return boundaries / action_scale

    def _to_torch_actions(self, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.reshape(self.num_envs, -1)
        if actions.shape[-1] != self.action_boundaries.shape[0]:
            raise ValueError(
                "Invalid action shape: expected "
                f"({self.num_envs}, {self.action_boundaries.shape[0]}), got {actions.shape}"
            )
        action_boundaries = torch.as_tensor(
            self.action_boundaries, dtype=torch.float32, device=self.device
        )
        return torch.clamp(actions, -1.0, 1.0) * action_boundaries

def make_holosoma_env(
    task_id: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
    to_numpy: bool = True,
    render_mode: str | None = None,
) -> HoloSomaVectorEnv:
    """Create a HoloSoma task from its registered experiment ID."""
    try:
        from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
        from holosoma.config_values.sensor import CAMERA_REGISTRY
        from holosoma.config_values.simulator import SIMULATOR_REGISTRY
        from holosoma.utils.sim_utils import (
            close_simulation_app,
            setup_simulation_environment,
        )
    except ImportError as exc:
        raise ImportError(
            "HoloSoma is not installed. Install the package from "
            "third_party/holosoma/src/holosoma in the active environment."
        ) from exc

    if task_id not in HOLOSOMA_TASKS:
        available_task_ids = ", ".join(HOLOSOMA_TASKS)
        raise ValueError(
            f"Unknown HoloSoma task_id {task_id!r}. "
            f"Available task IDs: {available_task_ids}"
        )
    registry_key = task_id.replace("-", "_")

    if render_mode not in (None, "rgb_array"):
        raise ValueError(
            f"Unsupported HoloSoma render_mode {render_mode!r}; expected 'rgb_array' or None"
        )
    config = EXPERIMENT_REGISTRY[registry_key]

    training = dataclasses.replace(
        config.training,
        num_envs=num_envs,
        seed=seed,
        headless=headless,
    )
    config = dataclasses.replace(
        config,
        training=training,
        simulator=SIMULATOR_REGISTRY["isaacsim"],
    )
    if render_mode == "rgb_array":
        sensors = dict(config.sensor)
        sensors["overview"] = CAMERA_REGISTRY["overview"]
        config = dataclasses.replace(config, sensor=sensors)
    task, _, simulation_app = setup_simulation_environment(config)

    return HoloSomaVectorEnv(
        task,
        simulation_app=simulation_app,
        close_simulation_app=close_simulation_app,
        to_numpy=to_numpy,
        render_mode=render_mode,
    )
