from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from .layer import FlashSACEncoder
from .policy import squash_tanh_action, squash_log_std_tanh

def encode_low_to_high_batch(x: jax.Array, perm: jax.Array, dim: int) -> jax.Array:
    """Batch version: x shape (B, m), output z shape (B, dim)."""
    x = jnp.asarray(x)
    m = x.shape[-1]
    if dim < m:
        raise ValueError(f"dim={dim} must be >= x.shape[-1]={m}")
    if perm.shape[0] != dim:
        raise ValueError(f"perm length {perm.shape[0]} must equal dim={dim}")

    pad_width = ((0, 0), (0, dim - m))
    x_pad = jnp.pad(x, pad_width)
    z = x_pad[:, perm]
    return z


def decode_high_to_low_batch(z: jax.Array, inv_perm: jax.Array, m: int) -> jax.Array:
    """Batch version: z shape (B, dim), output x shape (B, m)."""
    z = jnp.asarray(z)
    dim = z.shape[-1]
    if m > dim:
        raise ValueError(f"m={m} must be <= z.shape[-1]={dim}")
    if inv_perm.shape[0] != dim:
        raise ValueError(
            f"inv_perm length {inv_perm.shape[0]} must equal z.shape[-1]={dim}")

    x_pad = z[:, inv_perm]
    x = x_pad[:, :m]
    return x


def make_perm_invperm(dim: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    perm = jax.random.permutation(key, dim)
    inv_perm = jnp.argsort(perm)
    return perm, inv_perm


class AlphaNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ):
        self.backbone = FlashSACEncoder(
                int(latent_dim) + int(obs_dim), num_blocks, hidden_dim, rngs)
        self.scale_head = nnx.Linear(
            hidden_dim, int(latent_dim), rngs=rngs.fork())
        self.bias_head = nnx.Linear(
            hidden_dim, int(latent_dim), rngs=rngs.fork())

    def __call__(
        self,
        x: jax.Array,
        obs: jax.Array,
        cond_mask: jax.Array,
        ode_mask: jax.Array,
        training: bool
    ) -> tuple[jax.Array, jax.Array]:
        cond_mask = jnp.asarray(cond_mask, dtype=x.dtype)
        ode_mask = jnp.asarray(ode_mask, dtype=x.dtype)
        if cond_mask.ndim == 1:
            cond_mask = cond_mask[None, :]
        if ode_mask.ndim == 1:
            ode_mask = ode_mask[None, :]

        num_masks = cond_mask.shape[0]
        batch_size = x.shape[0]
        cond_x = x[None, :, :] * cond_mask[:, None, :]
        obs = jnp.broadcast_to(obs[None, :, :], (num_masks,) + obs.shape)
        network_input = jnp.concatenate([cond_x, obs], axis=-1)
        network_input = network_input.reshape((num_masks * batch_size, -1))

        h = self.backbone(network_input, training=training)
        raw_scale = squash_log_std_tanh(
            self.scale_head(h), log_std_min=-10, log_std_max=2
        )
        scale = raw_scale.reshape((num_masks, batch_size, x.shape[-1]))
        bias = self.bias_head(h).reshape((num_masks, batch_size, x.shape[-1]))
        scale = (jnp.exp(scale) - 1.0) * ode_mask[:, None, :]
        bias = bias * ode_mask[:, None, :]
        return scale, bias


class VelocityNetwork(nnx.Module):
    def __init__(
        self,
        split_dim: Sequence[int],
        obs_dim: int,
        latent_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2
    ):
        self.split_dim = tuple(int(v) for v in split_dim)
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)

        if sum(self.split_dim) != self.latent_dim:
            raise ValueError(
                f"sum(split_dim)={sum(self.split_dim)} must equal latent_dim={self.latent_dim}"
            )

        self.starts = tuple(sum(self.split_dim[:i])
                            for i in range(len(self.split_dim)))
        self.ends = tuple(s + w for s, w in zip(self.starts, self.split_dim))

        # Build masks once: each mask shape is (action_dim,)
        ode_masks = []
        cond_masks = []
        for start, end in zip(self.starts, self.ends):
            ode = jnp.zeros((self.latent_dim,),
                            dtype=jnp.float32).at[start:end].set(1.0)
            ode_masks.append(ode)
            cond_masks.append(1.0 - ode)

        self.ode_masks = jnp.stack(ode_masks, axis=0)
        self.cond_masks = jnp.stack(cond_masks, axis=0)

        self.networks = AlphaNetwork(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            rngs=rngs.fork(),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        )

    def euler_step(
        self, x: jax.Array, obs: jax.Array, training: bool
    ) -> tuple[jax.Array, jax.Array]:

        scale, bias = self.networks(
            x, obs, self.cond_masks, self.ode_masks, training)

        multiplier = 1.0 + scale.sum(0)
        x = x * multiplier + bias.sum(0)
        delta_logdet = jnp.sum(jnp.log(multiplier), axis=-1)

        return x, delta_logdet


class CoupleFlowActor(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        action_low: jax.Array = -1,
        action_high: jax.Array = 1,
        num_step: int = 5,
        num_ode: int = 3
    ):
        self.latent_dim = action_dim
        if num_ode > action_dim:
            self.perm, self.inv_perm = make_perm_invperm(
                num_ode, jax.random.PRNGKey(0))
            self.latent_dim = num_ode

        base = self.latent_dim // num_ode
        remainder = self.latent_dim % num_ode
        self.split_dim = tuple(base + (1 if i < remainder else 0)
                               for i in range(num_ode))

        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.num_step = int(num_step)

        self.v = VelocityNetwork(
            split_dim=self.split_dim,
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            rngs=rngs.fork(),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        )

        self.action_low = jnp.asarray(action_low, dtype=jnp.float32)
        self.action_high = jnp.asarray(action_high, dtype=jnp.float32)

    def _base_log_prob(self, z: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(z**2 + jnp.log(2.0 * jnp.pi), axis=-1)

    def _integrate(self, x: jax.Array, obs: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        logdet = jnp.zeros((x.shape[0],), dtype=x.dtype)
        for step in range(self.num_step):
            x, step_logdet = self.v.euler_step(x, obs, training)
            logdet = logdet + step_logdet
        return x, logdet

    def sample_and_log_prob(self, obs: jax.Array, *, training: bool, key: jax.Array | None = None, noise: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        if noise is not None:
            x_0 = noise
        elif key is not None:
            x_0 = jax.random.normal(
                key, (obs.shape[0], self.action_dim), dtype=jnp.float32)
        else:
            raise ValueError("Either key or noise must be provided.")
        z_0 = x_0
        if hasattr(self, 'perm'):
            z_0 = encode_low_to_high_batch(x_0, self.perm, self.latent_dim)
        pre_action, flow_logdet = self._integrate(z_0, obs, training)
        pre_log_prob = self._base_log_prob(x_0) - flow_logdet
        if hasattr(self, 'inv_perm'):
            pre_action = decode_high_to_low_batch(
                pre_action, self.inv_perm, self.action_dim)
        action, log_prob = squash_tanh_action(pre_action, pre_log_prob,
                                              self.action_low, self.action_high)
        return action, log_prob

    def get_action(self, obs: jax.Array, key: jax.Array, training: bool) -> tuple[jax.Array, jax.Array]:
        action, log_prob = self.sample_and_log_prob(obs, key=key, training=training)
        return action, log_prob[:, None]


    def get_mean_action(self, obs: jax.Array):
        noise = jnp.zeros((obs.shape[0], self.action_dim), dtype=jnp.float32)
        return self.sample_and_log_prob(obs, noise=noise, training=False)[0]
