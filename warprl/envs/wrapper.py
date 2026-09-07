from collections import deque
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from .types import Tensor


class ForwardingVectorWrapper(gym.vector.VectorWrapper):
    """Expose custom attributes defined by the wrapped vector environment."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


class VectorActionRepeat(ForwardingVectorWrapper):
    """Repeat a batched action until any vector slot finishes."""

    def __init__(self, env: VectorEnv, action_repeat: int = 4) -> None:
        super().__init__(env)
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be positive, got {action_repeat}")
        self._action_repeat = action_repeat

    def step(self, action: Tensor):
        total_reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, step_terminated, step_truncated, info = self.env.step(action)
            total_reward += reward
            terminated |= step_terminated
            truncated |= step_truncated
            combined_info.update(info)
            if np.any(terminated | truncated):
                break

        return obs, total_reward, terminated, truncated, combined_info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class PixelObservation(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        size: int = 84,
        stack: int = 3,
        gray: bool = False,
        image_layout: Literal["CHW", "HWC"] = "HWC",
    ) -> None:
        super().__init__(env)
        if image_layout not in ("HWC", "CHW"):
            raise ValueError(f"image_layout must be 'HWC' or 'CHW', got {image_layout!r}")

        self.size = size
        self.gray = gray
        self.image_layout = image_layout
        self.frames = deque(maxlen=stack)
        channels = 1 if gray else 3
        shape = (
            (size, size, channels * stack)
            if image_layout == "HWC"
            else (channels * stack, size, size)
        )
        self.observation_space = spaces.Box(
            0, 255, shape=shape, dtype=np.uint8
        )

    def _frame(self) -> np.ndarray:
        import cv2

        frame = self.env.render()    # (H, W, C)
        if frame.shape[:2] != (self.size, self.size):
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
        frame = np.asarray(frame, dtype=np.uint8)
        if self.image_layout == "CHW":
            frame = frame.transpose(2, 0, 1)
        return frame

    def _observation(self) -> np.ndarray:
        axis = -1 if self.image_layout == "HWC" else 0
        return np.concatenate(tuple(self.frames), axis=axis)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self._frame()
        self.frames.clear()
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return self._observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._frame())
        return self._observation(), reward, terminated, truncated, info


class ActionClip(ForwardingVectorWrapper):
    """Clip batched actions to the vector environment's action bounds."""

    def step(self, action: Tensor):
        clipped_action = action.clip(
            self.single_action_space.low,
            self.single_action_space.high,
        )
        return self.env.step(clipped_action)


def wrap_vector_env(
    env: VectorEnv,
    *,
    action_repeat: int,
    clip_action: bool,
) -> VectorEnv:
    if clip_action:
        env = ActionClip(env)
    if action_repeat > 1:
        env = VectorActionRepeat(env, action_repeat)
    return env
