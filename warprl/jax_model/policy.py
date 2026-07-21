from typing import Any, Optional
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


def make_action_affine_bijector(
    action_low: jax.Array,
    action_high: jax.Array,
) -> Any:
    """Creates an affine bijector that maps [-1, 1] -> [low, high]."""
    scale, bias = action_scale_bias(action_low, action_high)
    return tfb.Chain([tfb.Shift(shift=bias), tfb.Scale(scale=scale)])


def unbounded_to_action(
    pre_tanh: jax.Array,
    *,
    action_low: jax.Array,
    action_high: jax.Array,
) -> jax.Array:
    """Maps an unbounded latent action to a bounded action in [low, high].

    This is the standard tanh-squash used in SAC-style policies, generalized
    to arbitrary per-dimension bounds.
    """
    scale, bias = action_scale_bias(action_low, action_high)
    return jnp.tanh(pre_tanh) * scale + bias


def action_to_unbounded(
    action: jax.Array,
    *,
    action_low: jax.Array,
    action_high: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Maps a bounded action in [low, high] to an unbounded latent space.

    This is the inverse transform of `unbounded_to_action`:
      pre_tanh = atanh((action - bias) / scale)

    Notes:
    - We clip the normalized action to (-1 + eps, 1 - eps) to avoid infs.
    - This transform is useful when you want to model a bounded action with
      an unbounded density (e.g., a Gaussian in pre-tanh space).
    """
    scale, bias = action_scale_bias(action_low, action_high)
    normalized = (action - bias) / scale
    normalized = jnp.clip(normalized, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(normalized)


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
    )
    log_scale = jnp.log(jnp.abs(scale))
    if jnp.ndim(log_scale) == 0:
        logdet_affine = pre_action.shape[-1] * log_scale
    else:
        logdet_affine = jnp.sum(log_scale, axis=-1)
    log_prob = pre_log_prob - logdet_tanh - logdet_affine
    return action, _column_log_prob(log_prob)


def diagonal_gaussian_kl(mu_c: jax.Array, std_c: jax.Array, mu_o: jax.Array, std_o: jax.Array) -> jax.Array:
    kl = (
        jnp.log(std_c / std_o)
        + (std_o ** 2 + (mu_o - mu_c) ** 2) / (2.0 * std_c ** 2)
        - 0.5
    )
    return kl.sum(axis=-1)


def mask_logits(
    logits: jax.Array,
    legal_action_mask: Optional[jax.Array],
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
        legal_action_mask: Optional[jax.Array] = None,
    ) -> Any:
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return tfd.Categorical(logits=masked_logits)

    def sample_and_log_prob(
        self,
        logits: jax.Array,
        key: jax.Array,
        legal_action_mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(logits, legal_action_mask)
        action = d.sample(seed=key).astype(jnp.int32)
        log_prob = _column_log_prob(d.log_prob(action))
        return action, log_prob

    def greedy_action(self, logits: jax.Array, legal_action_mask: Optional[jax.Array] = None) -> jax.Array:
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)


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

    def dist(self, mean: jax.Array, log_std: jax.Array) -> Any:
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

    def transform_log_std(self, log_std: jax.Array):
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

    def dist(self, mean: jax.Array, log_std: jax.Array) -> Any:
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

    def transform_log_std(self, log_std: jax.Array):
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return log_std


class TanhDeterministicPolicy(_BoundedActionPolicy):
    """Deterministic tanh policy (no stochasticity)."""

    def action(self, pre_tanh: jax.Array) -> jax.Array:
        return self.mean_action(pre_tanh)


class MultivariateBetaPolicy(_BoundedActionPolicy):
    """Independent Beta distribution per action dimension, mapped to [low, high]."""

    def __init__(
        self,
        action_low: jax.Array,
        action_high: jax.Array,
        epsilon: float = 1e-4,
    ):
        super().__init__(action_low, action_high)
        self.epsilon = epsilon

    def dist(self, alpha: jax.Array, beta: jax.Array) -> Any:
        alpha = jax.nn.softplus(alpha) + self.epsilon
        beta = jax.nn.softplus(beta) + self.epsilon

        base = tfd.Independent(
            tfd.Beta(concentration1=alpha, concentration0=beta),
            reinterpreted_batch_ndims=1,
        )
        scale = self.action_scale.value
        bias = self.action_bias.value
        bijector = tfb.Chain(
            [tfb.Shift(shift=bias - scale), tfb.Scale(scale=2.0 * scale)])
        return tfd.TransformedDistribution(base, bijector)

    def sample_and_log_prob(
        self, alpha: jax.Array, beta: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(alpha, beta)
        action = d.sample(seed=key)
        log_prob = _column_log_prob(d.log_prob(action))
        return action, log_prob
