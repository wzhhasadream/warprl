import jax
from flax import nnx
from .network import Actor, ActorCritic, Critic
from ....utils import select_actor_observations
from ....model.jax import Network


@nnx.jit(static_argnames=("asymmetric_obs",))
def get_eval_action(
    agent: Network[ActorCritic[Actor, Critic]],
    asymmetric_obs: bool,
    obs: jax.Array,
) -> jax.Array:
    obs = select_actor_observations(obs, asymmetric_obs, agent.model.actor.obs_dim)
    return agent.model.get_mean_action(obs)



@nnx.jit
def get_value(
    agent: Network[ActorCritic[Actor, Critic]],
    obs: jax.Array,
) -> jax.Array:
    return agent.model.critic(obs, update_rms=False)


@nnx.jit(static_argnames=("asymmetric_obs",))
def sample_and_value(
    agent: Network[ActorCritic[Actor, Critic]],
    asymmetric_obs: bool,
    obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    actor = agent.model.actor
    critic = agent.model.critic
    actor_obs = select_actor_observations(obs, asymmetric_obs, actor.obs_dim)
    actions, log_probs, _, actions_mean, actions_std = actor.get_action(actor_obs, key, update_rms=True)
    value = critic(obs, update_rms=True)

    return actions, log_probs, value, actions_mean, actions_std


