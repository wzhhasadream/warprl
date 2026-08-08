import jax
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def _as_float_array(x: jax.Array) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)



def action_scale_bias(action_low: jax.Array, action_high: jax.Array) -> tuple[jax.Array, jax.Array]:
    '''
    return scale, bias
    '''
    action_low = _as_float_array(action_low)
    action_high = _as_float_array(action_high)
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    return scale, bias


def squash_log_std_tanh(log_std: jax.Array, *, log_std_min: float, log_std_max: float) -> jax.Array:
    """Squashes log_std to [log_std_min, log_std_max] using tanh."""
    log_std = jnp.tanh(log_std)
    return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)


def _column_log_prob(log_prob: jax.Array) -> jax.Array:
    return jnp.reshape(log_prob, (-1, 1))


def squash_tanh_action(pre_action: jax.Array, pre_log_prob: jax.Array, action_low: jax.Array, action_high: jax.Array):
    """Apply tanh squashing with affine action scaling and corrected log-prob."""
    scale, bias = action_scale_bias(action_low, action_high)
    action = scale * jax.nn.tanh(pre_action) + bias
    logdet_tanh = jnp.sum(
        2.0 * (jnp.log(2.0) - pre_action - jax.nn.softplus(-2.0 * pre_action)),
        axis=-1,
        keepdims=True,
    )
    log_scale = jnp.log(jnp.abs(scale))
    logdet_affine = jnp.sum(log_scale)
    log_prob = _column_log_prob(pre_log_prob) - logdet_tanh - logdet_affine
    return action, log_prob


def mask_logits(
    logits: jax.Array,
    legal_action_mask: jax.Array | None,
    *,
    invalid_logit: float = -1e9,
) -> jax.Array:
    """Masks invalid actions by setting their logits to a large negative value.

    Notes:
      - `legal_action_mask` is expected to be boolean with the same shape as `logits`.
      - If a row has no legal actions (all-False), we fall back to uniform logits (all zeros)
        to avoid NaNs from normalizing all `-inf` logits.
    """
    if legal_action_mask is None:
        return logits
    mask = jnp.asarray(legal_action_mask, dtype=jnp.bool_)
    masked = jnp.where(mask, logits, jnp.asarray(
        invalid_logit, dtype=logits.dtype))
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    return jnp.where(any_valid, masked, jnp.zeros_like(masked))


class MaskedCategoricalPolicy(nnx.Module):
    """Categorical policy over discrete actions with an optional legality mask."""

    def __init__(self, invalid_logit: float = -1e9):
        self.invalid_logit = invalid_logit

    def dist(
        self,
        logits: jax.Array,
        legal_action_mask: jax.Array | None = None,
    ):
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return tfd.Categorical(logits=masked_logits)

    def sample_and_log_prob(
        self,
        logits: jax.Array,
        key: jax.Array,
        legal_action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(logits, legal_action_mask)
        action = d.sample(seed=key).astype(jnp.int32)
        log_prob = _column_log_prob(d.log_prob(action))
        return action, log_prob

    def greedy_action(self, logits: jax.Array, legal_action_mask: jax.Array | None = None) -> jax.Array:
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)


class DiscreteQGreedyPolicy(nnx.Module):
    """Greedily select the highest-valued legal discrete action."""

    @staticmethod
    def _mask_q_values(
        q_values: jax.Array,
        legal_action_mask: jax.Array | None,
    ) -> jax.Array:
        if legal_action_mask is None:
            return q_values
        mask = jnp.asarray(legal_action_mask, dtype=jnp.bool_)
        masked_q_values = jnp.where(mask, q_values, -jnp.inf)
        return jnp.where(
            jnp.any(mask, axis=-1, keepdims=True),
            masked_q_values,
            jnp.zeros_like(masked_q_values),
        )

    def action(
        self,
        q_values: jax.Array,
        legal_action_mask: jax.Array | None = None,
    ) -> jax.Array:
        return jnp.argmax(
            self._mask_q_values(q_values, legal_action_mask), axis=-1
        ).astype(jnp.int32)

    def greedy_q(
        self,
        q_values: jax.Array,
        legal_action_mask: jax.Array | None = None,
    ) -> jax.Array:
        return jnp.max(self._mask_q_values(q_values, legal_action_mask), axis=-1).astype(
            jnp.float32
        )

    def greedy_action(
        self,
        q_values: jax.Array,
        legal_action_mask: jax.Array | None = None,
    ) -> jax.Array:
        return self.action(q_values, legal_action_mask)


class GaussianPolicy(nnx.Module):
    """Diagonal Gaussian policy (no tanh squashing)."""

    def __init__(
        self,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = False,
    ):
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash_log_std = squash_log_std

    def dist(self, mean: jax.Array, log_std: jax.Array):
        if self.squash_log_std:
            log_std = self.transform_log_std(log_std)
        std = jnp.exp(log_std)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

    def sample_and_log_prob(
        self, mean: jax.Array, log_std: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(mean, log_std)
        action = d.sample(seed=key)
        log_prob = _column_log_prob(d.log_prob(action))
        return action, log_prob

    def transform_log_std(self, log_std: jax.Array) -> jax.Array:
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return log_std


class _BoundedActionPolicy(nnx.Module):
    def __init__(self, action_low: jax.Array, action_high: jax.Array):
        action_low = _as_float_array(action_low)
        action_high = _as_float_array(action_high)
        action_scale, action_bias = action_scale_bias(action_low, action_high)
        self.action_low = nnx.Variable(action_low)
        self.action_high = nnx.Variable(action_high)
        self.action_scale = nnx.Variable(action_scale)
        self.action_bias = nnx.Variable(action_bias)

    def mean_action(self, pre_tanh: jax.Array) -> jax.Array:
        return jnp.tanh(pre_tanh) * self.action_scale.value + self.action_bias.value


class SquashedTanhGaussianPolicy(_BoundedActionPolicy):
    """Tanh-squashed diagonal Gaussian policy (SAC-style)."""

    def __init__(
        self,
        action_low: jax.Array,
        action_high: jax.Array,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash_log_std: bool = True,
    ):
        super().__init__(action_low, action_high)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash_log_std = squash_log_std

    def dist(self, mean: jax.Array, log_std: jax.Array):
        if self.squash_log_std:
            log_std = self.transform_log_std(log_std)
        std = jnp.exp(log_std)
        base = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        scale = self.action_scale.value
        bias = self.action_bias.value
        bijector = tfb.Chain(
            [
                tfb.Shift(shift=bias),
                tfb.Scale(scale=scale),
                tfb.Tanh(),
            ]
        )
        return tfd.TransformedDistribution(base, bijector)

    def sample_and_log_prob(
        self, mean: jax.Array, log_std: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(mean, log_std)
        action = d.sample(seed=key)
        log_prob = _column_log_prob(d.log_prob(action))
        return action, log_prob

    def transform_log_std(self, log_std: jax.Array) -> jax.Array:
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return log_std


class TanhDeterministicPolicy(_BoundedActionPolicy):
    """Deterministic tanh policy (no stochasticity)."""

    def action(self, pre_tanh: jax.Array) -> jax.Array:
        return self.mean_action(pre_tanh)


class CoupledFlowPolicy(_BoundedActionPolicy):
    def __init__(self,
                 action_low: jax.Array,
                 action_high: jax.Array,
                 action_dim: int,
                 num_ode: int,
                 mask_key: jax.Array | None = None,
                 alpha_min: float = -10,
                 alpha_max: float = 2,
                 squash_alpha: bool = False
                 ):
        super().__init__(action_low, action_high)
        self.latent_dim = max(action_dim, num_ode)
        self.action_dim = action_dim
        self.num_ode = num_ode
        self.squash_alpha = squash_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self._perm = nnx.Variable(jax.random.permutation(
            jax.random.PRNGKey(0), self.latent_dim))

        self._inv_perm = nnx.Variable(jnp.argsort(self.perm))

        cond_mask, ode_mask = self.make_masks(mask_key)
        self._cond_mask, self._ode_mask = nnx.Variable(
            cond_mask), nnx.Variable(ode_mask)    # (num_ode, latent_dim)

    @property
    def cond_mask(self):
        return self._cond_mask.value

    @property
    def ode_mask(self):
        return self._ode_mask.value

    @property
    def perm(self):
        return self._perm.value

    @property
    def inv_perm(self):
        return self._inv_perm.value

    def make_masks(
        self,
        key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Create conditioner and ODE masks from latent dim and ODE count."""

        base = self.latent_dim // self.num_ode
        remainder = self.latent_dim % self.num_ode
        split_dim = tuple(base + (1 if i < remainder else 0)
                          for i in range(self.num_ode))
        indices = jnp.arange(self.latent_dim)
        if key is not None:
            indices = jax.random.permutation(key, indices)

        ode_masks = []
        cond_masks = []
        start = 0
        for width in split_dim:
            end = start + width
            ode = jnp.zeros((self.latent_dim,),
                            dtype=jnp.float32).at[indices[start:end]].set(1.0)
            ode_masks.append(ode)
            cond_masks.append(1.0 - ode)
            start = end

        return jnp.stack(cond_masks, axis=0), jnp.stack(ode_masks, axis=0)

    def encode_low_to_high_batch(self, x: jax.Array) -> jax.Array:
        """Batch version: x shape (B, m), output z shape (B, dim)."""
        x = jnp.asarray(x)
        pad_width = ((0, 0), (0, self.latent_dim - self.action_dim))
        x_pad = jnp.pad(x, pad_width)
        z = x_pad[:, self.perm]
        return z

    def decode_high_to_low_batch(self, z: jax.Array) -> jax.Array:
        """Batch version: z shape (B, dim), output x shape (B, m)."""
        z = jnp.asarray(z)

        x_pad = z[:, self.inv_perm]
        x = x_pad[:, : self.action_dim]
        return x

    def affine_params(
        self,
        raw_alpha: jax.Array,
        raw_beta: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Convert network heads into masked direct affine multipliers.

        Shapes:
          raw_alpha: (num_ode, batch_size, latent_dim)
          raw_beta: (num_ode, batch_size, latent_dim)

        Returns:
          alpha: (batch_size, latent_dim)
          beta: (batch_size, latent_dim)
        """
        alpha = raw_alpha * self.ode_mask[:, None, :]
        beta = raw_beta * self.ode_mask[:, None, :]
        if self.squash_alpha:
            alpha = squash_log_std_tanh(
                alpha.sum(axis=0), log_std_min=self.alpha_min, log_std_max=self.alpha_max)
        else:
            alpha = alpha.sum(axis=0)
        return alpha, beta.sum(axis=0)

    def flow_step(
        self,
        x: jax.Array,
        alpha: jax.Array,
        beta: jax.Array,
        step_size: float = 1
    ) -> tuple[jax.Array, jax.Array]:
        """Apply one direct affine CoupledFlow step and return log-det.

        Shapes:
          x: (batch_size, latent_dim)
          alpha: (batch_size, latent_dim)
          beta: (batch_size, latent_dim)

        Returns:
          next_x: (batch_size, latent_dim)
          delta_logprob: (batch_size, 1)
        """
        v = x * alpha + beta
        x = x + v * step_size
        delta_logprob = jnp.sum(alpha, axis=-1, keepdims=True) * step_size
        return x, delta_logprob

    def base_log_prob(self, z: jax.Array) -> jax.Array:
        """Compute standard normal base log-probability.

        Shapes:
          z: (batch_size, action_dim)

        Returns:
          log_prob: (batch_size, 1)
        """
        return -0.5 * jnp.sum(z**2 + jnp.log(2.0 * jnp.pi), axis=-1, keepdims=True)

    def squash_action(
        self,
        pre_action: jax.Array,
        pre_log_prob: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Squash pre-actions to action bounds with corrected log-prob.

        Shapes:
          pre_action: (batch_size, action_dim)
          pre_log_prob: (batch_size, 1)

        Returns:
          action: (batch_size, action_dim)
          log_prob: (batch_size, 1)
        """
        return squash_tanh_action(
            pre_action,
            pre_log_prob,
            self.action_low.value,
            self.action_high.value,
        )
