from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from gymnasium.vector import VectorEnv

from ..agent.base_agent import OffPolicyAgent
from ..buffers import Transition
from ..utils import evaluate_policy, record_video, replace_done_next_obs
from ..utils import logger
from .types import EnvironmentConfig, OffPolicyRunnerConfig

class OffPolicyRunner:
    """Run the environment-facing lifecycle shared by off-policy algorithms."""

    def __init__(
        self,
        train_envs: VectorEnv,
        eval_envs: VectorEnv,
        record_envs: VectorEnv,
        agent: OffPolicyAgent,
        env_cfg: EnvironmentConfig,
        run_cfg: OffPolicyRunnerConfig,
        *,
        results_dir: str | Path,
        project: str,
        run_name: str
    ) -> None:
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.record_envs = record_envs
        self.agent = agent
        self.env_cfg = env_cfg
        self.run_cfg = run_cfg
        self.results_dir = str(results_dir)
        self.project = project
        self.run_name = run_name
        self._grad_step_accumulator = 0.0
        if run_cfg.num_eval < 1 or run_cfg.num_log < 1:
            raise ValueError("num_eval and num_log must be positive")
        self.num_log = max(1, self.num_interaction_steps // run_cfg.num_log)
        self.num_eval = max(1, self.num_interaction_steps // run_cfg.num_eval)



        logger.init(
            project=self.project,
            name=self.run_name,
            config={**vars(agent.cfg), **self.agent.observation_debug_info},
            dir=self.results_dir,
        )

    @property
    def num_interaction_steps(self) -> int:
        return self.run_cfg.total_timesteps // self.train_envs.num_envs

    @property
    def log_dir(self) -> str | None:
        return logger.get_run_dir() if logger.has_active_run() else None



    def _log_evaluation(self, global_step: int) -> None:
        info = evaluate_policy(
            self.agent.get_action,
            self.eval_envs,
            self.env_cfg.eval_episode,
            self.env_cfg.env_type,
        )
        logger.log(info, global_step)
        if self.run_cfg.record_video:
            logger.video(
                record_video(
                    self.agent.get_action,
                    self.record_envs,
                    self.env_cfg.env_type,
                ),
                global_step,
            )
        if self.run_cfg.save_agent:
            logger.save_agent(self.agent, global_step)
        if self.run_cfg.save_onnx:
            logger.save_onnx(self.agent, global_step)

    def _real_next_observations(
        self,
        next_observations: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: dict[str, Any],
    ) -> np.ndarray:
        dones = np.logical_or(terminations, truncations)
        final_obs_mask = (
            truncations if self.env_cfg.env_type == "myosuite" else dones
        )
        return replace_done_next_obs(next_observations, final_obs_mask, infos)

    def _close_envs(self) -> None:
        self.train_envs.close()
        if self.eval_envs is not self.train_envs:
            self.eval_envs.close()
        if (
            self.record_envs is not self.train_envs
            and self.record_envs is not self.eval_envs
        ):
            self.record_envs.close()

    def run(self) -> None:
        np.random.seed(self.env_cfg.seed)
        config = self.agent.cfg

        try:
            self._log_evaluation(0)
            observations, _ = self.train_envs.reset(seed=self.env_cfg.seed)
            for interaction_step in tqdm.tqdm(
                range(1, self.num_interaction_steps + 1),
                smoothing=0.1,
                mininterval=0.5,
            ):
                if self.agent.can_update:
                    actions = self.agent.get_exploration_action(observations)
                else:
                    actions = self.train_envs.action_space.sample()

                next_observations, rewards, terminations, truncations, infos = (
                    self.train_envs.step(actions)
                )
                real_next_observations = self._real_next_observations(
                    next_observations,
                    terminations,
                    truncations,
                    infos,
                )
                self.agent.process_transition(
                    Transition(
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        truncations=truncations,
                        terminations=terminations,
                        next_observations=real_next_observations,
                    )
                )

                if self.agent.can_update:
                    update_info = {}
                    self._grad_step_accumulator += (
                        self.run_cfg.grad_step_per_interaction_step
                    )
                    num_updates = int(self._grad_step_accumulator)
                    self._grad_step_accumulator -= num_updates
                    for _ in range(num_updates):
                        update_info = self.agent.update()

                    global_step = interaction_step * self.train_envs.num_envs
                    if interaction_step % self.num_log == 0:
                        logger.log(update_info, global_step)
                    if interaction_step % self.num_eval == 0:
                        self._log_evaluation(global_step)

                observations = next_observations

            self._log_evaluation(
                self.num_interaction_steps * self.train_envs.num_envs
            )
        finally:
            logger.finish()
            self._close_envs()
