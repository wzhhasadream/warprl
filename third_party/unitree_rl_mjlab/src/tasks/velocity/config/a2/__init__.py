from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_a2_flat_env_cfg,
  unitree_a2_rough_env_cfg,
)
from .rl_cfg import unitree_a2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-A2-Rough",
  env_cfg=unitree_a2_rough_env_cfg(),
  play_env_cfg=unitree_a2_rough_env_cfg(play=True),
  rl_cfg=unitree_a2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-A2-Flat",
  env_cfg=unitree_a2_flat_env_cfg(),
  play_env_cfg=unitree_a2_flat_env_cfg(play=True),
  rl_cfg=unitree_a2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
