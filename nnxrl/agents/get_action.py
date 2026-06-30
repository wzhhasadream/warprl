import jax
import jax.numpy as jnp
from flax import nnx

from nnxrl.model import FlashSACActor
from nnxrl.utils import RewardNormalizer, sample_truncated_zeta, select_actor_observations


@nnx.jit(static_argnames=("asymmetric_obs",))
def get_eval_action(
    actor: FlashSACActor,
    asymmetric_obs: bool,
    obs: jax.Array,
) -> jax.Array:
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)
    return actor.get_mean_action(obs)


@nnx.jit(static_argnames=("asymmetric_obs",))
def get_exploration_action(
    actor: FlashSACActor,
    asymmetric_obs: bool,
    obs: jax.Array,
    repeat_n: jax.Array,
    repeat_count: jax.Array,
    cached_key: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    zeta_key, action_key = jax.random.split(key, 2)
    obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)

    refresh = jnp.logical_or(repeat_count == 0, repeat_count >= repeat_n)
    true_action_key = jnp.where(refresh, action_key, cached_key)
    actions = actor.get_action(obs, key=true_action_key, training=False)[0]

    new_repeat_n = jnp.where(refresh, sample_truncated_zeta(zeta_key), repeat_n)
    new_repeat_count = jnp.where(refresh, 1, repeat_count + 1)
    return true_action_key, actions, new_repeat_n, new_repeat_count


@jax.jit
def update_reward_normalizer(
    reward_normalizer: RewardNormalizer | None,
    rewards: jax.Array,
    dones: jax.Array,
) -> RewardNormalizer | None:
    if reward_normalizer is None:
        return None
    return reward_normalizer.update(rewards, dones)
