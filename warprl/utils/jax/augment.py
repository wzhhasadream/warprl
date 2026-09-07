import jax
import jax.numpy as jnp


def random_shift(
    key: jax.Array,
    image: jax.Array,
    padding: int = 4,
) -> jax.Array:
    """Apply an edge-padded random shift to one HWC image."""
    offset = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    offset = jnp.concatenate((offset, jnp.zeros(1, dtype=jnp.int32)))
    padded_image = jnp.pad(
        image,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )
    return jax.lax.dynamic_slice(padded_image, offset, image.shape)


def batched_random_shift(
    key: jax.Array,
    images: jax.Array,
    padding: int = 4,
) -> jax.Array:
    """Apply independent random shifts to a batch of NHWC images."""
    keys = jax.random.split(key, images.shape[0])
    return jax.vmap(random_shift, in_axes=(0, 0, None))(keys, images, padding)


def augment_observations(
    key: jax.Array,
    observations: jax.Array,
    next_observations: jax.Array,
    padding: int = 4,
) -> tuple[jax.Array, jax.Array]:
    """Augment current and next pixel observations with independent shifts."""
    observation_key, next_observation_key = jax.random.split(key)
    return (
        batched_random_shift(observation_key, observations, padding),
        batched_random_shift(next_observation_key, next_observations, padding),
    )
