from functools import lru_cache
import warnings

import gymnasium as gym
import mujoco
from dm_control import suite
from dm_control.mujoco import index as dmc_index
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium


# 20 tasks
DMC_EASY_MEDIUM = [
    "acrobot-swingup",
    "ball_in_cup-catch",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-walk",
    "quadruped-run",
    "reacher-easy",
    "reacher-hard",
    "walker-stand",
    "walker-walk",
    "walker-run",
]

# 8 tasks
DMC_SPARSE = [
    "cartpole-balance_sparse",
    "cartpole-swingup_sparse",
    "ball_in_cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "reacher-easy",
    "reacher-hard",
]

# 7 tasks
DMC_HARD = [
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
    "dog-stand",
    "dog-walk",
    "dog-run",
    "dog-trot",
]


# mjlab requires MuJoCo 3.10.0, but the dm_control named-index schema still
# references fields removed from newer MuJoCo builds. Keep the mjlab-compatible
# MuJoCo version and filter only unavailable indexing fields before loading DMC.
@lru_cache(maxsize=1)
def _patch_dm_control_mujoco_schema() -> tuple[str, ...]:
    """Remove dm_control index fields unavailable in the loaded MuJoCo build."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="compat_camera"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    missing_fields: list[str] = []

    for struct_name, struct in (("mjmodel", model), ("mjdata", data)):
        schema = dmc_index.sizes.array_sizes[struct_name]
        for field_name in tuple(schema):
            if not hasattr(struct, field_name):
                schema.pop(field_name)
                missing_fields.append(f"{struct_name}.{field_name}")

    if missing_fields:
        warnings.warn(
            f"Patched dm_control named indexing for MuJoCo {mujoco.__version__}; "
            f"ignored {len(missing_fields)} unavailable fields.",
            RuntimeWarning,
            stacklevel=2,
        )

    return tuple(missing_fields)


def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    _patch_dm_control_mujoco_schema()
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode=render_mode)
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env
