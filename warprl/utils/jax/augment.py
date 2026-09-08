import jax
import jax.numpy as jnp


def augment_observations(
    rng: jax.Array,
    observations: jax.Array,
    padding: int = 4,
) -> jax.Array:
    """Random shift with bilinear interpolation, matching DrQv2.

    Observations are channels-last and frame-first: [B, F, H, W, C].
    """
    n, f, h, w, c = observations.shape
    assert h == w

    observations = jnp.pad(
        observations,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )

    eps = 1.0 / (h + 2 * padding)

    arange = jnp.linspace(-1.0 + eps, 1.0 - eps, num=h + 2 * padding)[:h]
    arange = jnp.expand_dims(arange, 0)
    arange = jnp.repeat(arange, h, axis=0)
    arange = jnp.expand_dims(arange, -1)

    base_grid = jnp.concatenate([jnp.swapaxes(arange, 1, 0), arange], axis=2)
    base_grid = jnp.expand_dims(base_grid, (0, 1))
    base_grid = jnp.repeat(base_grid, n, axis=0)  # [N,1,H,W,2]
    base_grid = jnp.repeat(base_grid, f * c, axis=1)  # [N,F*C,H,W,2]

    shift = jax.random.randint(rng, (n, 1, 1, 1, 2), 0, 2 * padding + 1)  # [N,1,1,1,2]
    shift = shift * 2.0 / (h + 2 * padding)

    grid = base_grid + shift  # [N,F*C,H,W,2]

    # convert coords (from torch's grid_sample to jax's map_coordinates)
    grid = (grid + 1) / 2 * (h + 2 * padding - 1)

    grid_coords = grid.reshape(n * f * c, h * w, 2)
    observations = jnp.transpose(observations, (0, 1, 4, 2, 3)).reshape(
        n * f * c, h + 2 * padding, w + 2 * padding
    )

    def grid_sample(ch, coords):
        return jax.scipy.ndimage.map_coordinates(
            ch, [coords[:, 0], coords[:, 1]], order=1
        )

    grid_sample_batch = jax.vmap(grid_sample, in_axes=(0, 0), out_axes=0)
    augmented_obs = grid_sample_batch(observations, grid_coords)
    augmented_obs = augmented_obs.reshape(n, f, c, h, w)
    augmented_obs = jnp.transpose(augmented_obs, (0, 1, 3, 4, 2))

    return augmented_obs
