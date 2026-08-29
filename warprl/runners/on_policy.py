from __future__ import annotations

from pathlib import Path

import numpy as np
import tqdm
from gymnasium.vector import VectorEnv

from ..agent.base_agent import OnPolicyAgent
from ..buffers import RolloutTransition
from ..utils import bootstrap_timeout_rewards, evaluate_policy, record_video
from ..utils import logger
from .types import EnvironmentConfig, OnPolicyRunnerConfig


class OnPolicyRunner:
    """Run the environment-facing lifecycle shared by on-policy algorithms."""

    def __init__(
        self,
        train_envs: VectorEnv,
        eval_envs: VectorEnv,
        record_envs: VectorEnv,
        agent: OnPolicyAgent,
        env_cfg: EnvironmentConfig,
        run_cfg: OnPolicyRunnerConfig,
        *,
        results_dir: str | Path,
        project: str,
        run_name: str,
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
        if run_cfg.num_eval < 1 or run_cfg.num_log < 1:
            raise ValueError("num_eval and num_log must be positive")
        self.num_rollout = self.total_timesteps // (
            self.train_envs.num_envs * run_cfg.rollout_steps
        )
        self.num_log = max(1, self.num_rollout // run_cfg.num_log)
        self.num_eval = max(1, self.num_rollout // run_cfg.num_eval)

    @property
    def total_timesteps(self) -> int:
        return self.run_cfg.total_timesteps

    def _log_evaluation(self, rollout_idx: int) -> None:
        global_step = rollout_idx * self.train_envs.num_envs * self.run_cfg.rollout_steps
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
        logger.init(
            project=self.project,
            name=self.run_name,
            config={**vars(config), **self.agent.observation_debug_info},
            dir=self.results_dir,
        )

        try:
            self._log_evaluation(0)
            observations, _ = self.train_envs.reset(seed=self.env_cfg.seed)
            for rollout_idx in tqdm.tqdm(
                range(1, self.num_rollout + 1),
                smoothing=0.1,
                mininterval=0.5,
            ):
                for _ in range(self.run_cfg.rollout_steps):
                    actions, values, log_probs, actions_mean, actions_std = (
                        self.agent.sample_action_and_value(observations)
                    )
                    next_observations, rewards, terminations, truncations, infos = (
                        self.train_envs.step(actions)
                    )
                    timeouts = np.logical_and(truncations, ~terminations)
                    rewards = bootstrap_timeout_rewards(
                        rewards,
                        timeouts,
                        self.run_cfg.gamma,
                        next_observations,
                        self.agent.get_value,
                        infos,
                    )
                    self.agent.process_transition(
                        RolloutTransition(
                            observations=observations,
                            actions=actions,
                            rewards=rewards,
                            terminated=terminations,
                            truncated=truncations,
                            values=values,
                            log_probs=log_probs,
                            actions_mean=actions_mean,
                            actions_std=actions_std,
                        )
                    )
                    observations = next_observations

                update_info = self.agent.update(observations)
                global_step = (
                    rollout_idx
                    * self.train_envs.num_envs
                    * self.run_cfg.rollout_steps
                )
                if rollout_idx % self.num_log == 0:
                    logger.log(update_info, global_step)
                if rollout_idx % self.num_eval == 0:
                    self._log_evaluation(rollout_idx)

            self._log_evaluation(self.num_rollout)
        finally:
            logger.finish()
            self._close_envs()
