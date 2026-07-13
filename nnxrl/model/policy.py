from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp


def _as_float_array(x: Any) -> jax.Array:
    if hasattr(x, "get_value"):
        try:
            x = x[...]
        except TypeError:
            x = x.get_value()
    return jnp.asarray(x, dtype=jnp.float32)


def flattened_dim(spec: Any) -> int:
    """Converts an observation size spec into a flat integer dimension.

    The MuJoCo Playground/Brax API may expose observation sizes as:
      - int (already a dimension)
      - tuple[int, ...] / list[int] (shape)
    """
    if isinstance(spec, int):
        return int(spec)
    if isinstance(spec, (tuple, list)):
        dim = 1
        for v in spec:
            dim *= int(v)
        return int(dim)
    raise TypeError(f"Unsupported observation size spec type: {type(spec)}")




def action_scale_bias(action_low: jax.Array, action_high: jax.Array) -> tuple[jax.Array, jax.Array]:
    '''
    return scale, bias
    '''
    action_low = _as_float_array(action_low)
    action_high = _as_float_array(action_high)
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    return scale, bias


@dataclass(frozen=True)
class ActionAffineTransform:
    """Affine transform that maps [-1, 1] to [action_low, action_high]."""

    action_low: jax.Array
    action_high: jax.Array

    def forward(self, x: jax.Array) -> jax.Array:
        scale, bias = action_scale_bias(self.action_low, self.action_high)
        return x * scale + bias

    def inverse(self, y: jax.Array) -> jax.Array:
        scale, bias = action_scale_bias(self.action_low, self.action_high)
        return (y - bias) / scale

    def forward_log_det_jacobian(self, event_ndims: int = 1) -> jax.Array:
        del event_ndims
        scale, _ = action_scale_bias(self.action_low, self.action_high)
        log_scale = jnp.log(jnp.abs(scale))
        if jnp.ndim(log_scale) == 0:
            return log_scale
        return jnp.sum(log_scale, axis=-1)


def make_action_affine_bijector(action_low: jax.Array, action_high: jax.Array) -> ActionAffineTransform:
    """Creates an affine transform that maps [-1, 1] to [low, high]."""
    return ActionAffineTransform(action_low, action_high)


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
    if getattr(pre_log_prob, "ndim", None) == 2:
        log_prob = log_prob[:, None]
    return action, log_prob


def diagonal_gaussian_kl(mu_c: jax.Array, std_c: jax.Array, mu_o: jax.Array, std_o: jax.Array) -> jax.Array:
    kl = (
        jnp.log(std_c / std_o)
        + (std_o ** 2 + (mu_o - mu_c) ** 2) / (2.0 * std_c ** 2)
        - 0.5
    )
    return kl.sum(axis=-1)


def _diagonal_gaussian_log_prob(
    value: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=value.dtype))
    normalized = (value - mean) * jnp.exp(-log_std)
    per_dim = -0.5 * jnp.square(normalized) - log_std - 0.5 * log_two_pi
    return jnp.sum(per_dim, axis=-1)


@dataclass(frozen=True)
class CategoricalDistribution:
    """Small JAX categorical distribution compatible with the policy API."""

    logits: jax.Array

    def sample(self, seed: jax.Array) -> jax.Array:
        return jax.random.categorical(seed, self.logits, axis=-1).astype(jnp.int32)

    def log_prob(self, value: jax.Array) -> jax.Array:
        value = jnp.asarray(value, dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1)[..., 0]

    def entropy(self) -> jax.Array:
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs, axis=-1)


@dataclass(frozen=True)
class DiagonalGaussianDistribution:
    """Small JAX diagonal Gaussian distribution compatible with the policy API."""

    mean: jax.Array
    log_std: jax.Array

    def sample(self, seed: jax.Array) -> jax.Array:
        noise = jax.random.normal(seed, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + jnp.exp(self.log_std) * noise

    def log_prob(self, value: jax.Array) -> jax.Array:
        return _diagonal_gaussian_log_prob(value, self.mean, self.log_std)

    def entropy(self) -> jax.Array:
        return jnp.sum(
            self.log_std + 0.5 * jnp.log(jnp.asarray(2.0 * jnp.pi * jnp.e, dtype=self.log_std.dtype)),
            axis=-1,
        )


@dataclass(frozen=True)
class SquashedTanhGaussianDistribution:
    """Tanh-squashed diagonal Gaussian distribution with affine action scaling."""

    mean: jax.Array
    log_std: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    eps: float = 1e-6

    def sample(self, seed: jax.Array) -> jax.Array:
        noise = jax.random.normal(seed, self.mean.shape, dtype=self.mean.dtype)
        pre_tanh = self.mean + jnp.exp(self.log_std) * noise
        return unbounded_to_action(
            pre_tanh,
            action_low=self.action_low,
            action_high=self.action_high,
        )

    def log_prob(self, value: jax.Array) -> jax.Array:
        pre_tanh = action_to_unbounded(
            value,
            action_low=self.action_low,
            action_high=self.action_high,
            eps=self.eps,
        )
        pre_log_prob = _diagonal_gaussian_log_prob(pre_tanh, self.mean, self.log_std)
        _, log_prob = squash_tanh_action(
            pre_tanh,
            pre_log_prob,
            self.action_low,
            self.action_high,
        )
        return log_prob


@dataclass(frozen=True)
class AffineBetaDistribution:
    """Independent beta distribution per action dimension mapped to action bounds."""

    alpha: jax.Array
    beta: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    eps: float = 1e-6

    def sample(self, seed: jax.Array) -> jax.Array:
        unit_action = jax.random.beta(seed, self.alpha, self.beta)
        scale, bias = action_scale_bias(self.action_low, self.action_high)
        return unit_action * (2.0 * scale) + (bias - scale)

    def log_prob(self, value: jax.Array) -> jax.Array:
        scale, bias = action_scale_bias(self.action_low, self.action_high)
        unit_action = (value - (bias - scale)) / (2.0 * scale)
        unit_action = jnp.clip(unit_action, self.eps, 1.0 - self.eps)
        log_beta = (
            jax.scipy.special.gammaln(self.alpha)
            + jax.scipy.special.gammaln(self.beta)
            - jax.scipy.special.gammaln(self.alpha + self.beta)
        )
        per_dim = (
            (self.alpha - 1.0) * jnp.log(unit_action)
            + (self.beta - 1.0) * jnp.log1p(-unit_action)
            - log_beta
        )
        log_det = jnp.log(jnp.abs(2.0 * scale))
        if jnp.ndim(log_det) == 0:
            log_det = value.shape[-1] * log_det
        else:
            log_det = jnp.sum(log_det, axis=-1)
        return jnp.sum(per_dim, axis=-1) - log_det




    
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
    masked = jnp.where(mask, logits, jnp.asarray(invalid_logit, dtype=logits.dtype))
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    return jnp.where(any_valid, masked, jnp.zeros_like(masked))


@dataclass(frozen=True)
class MaskedCategoricalPolicy:
    """Categorical policy over discrete actions with an optional legality mask."""

    invalid_logit: float = -1e9

    def dist(self, logits: jax.Array, legal_action_mask: Optional[jax.Array] = None) -> CategoricalDistribution:
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return CategoricalDistribution(masked_logits)

    def sample_and_log_prob(
        self,
        logits: jax.Array,
        key: jax.Array,
        legal_action_mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(logits, legal_action_mask)
        action = d.sample(seed=key).astype(jnp.int32)
        log_prob = d.log_prob(action)
        return action, log_prob

    def greedy_action(self, logits: jax.Array, legal_action_mask: Optional[jax.Array] = None) -> jax.Array:
        masked_logits = mask_logits(
            logits, legal_action_mask, invalid_logit=self.invalid_logit
        )
        return jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)


@dataclass(frozen=True)
class GaussianPolicy:
    """Diagonal Gaussian policy (no tanh squashing)."""

    log_std_min: float = -10.0
    log_std_max: float = 2.0
    squash_log_std: bool = False

    def dist(self, mean: jax.Array, log_std: jax.Array) -> DiagonalGaussianDistribution:
        if self.squash_log_std:
            log_std = self.transform_log_std(log_std)
        return DiagonalGaussianDistribution(mean, log_std)

    def sample_and_log_prob(
        self, mean: jax.Array, log_std: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(mean, log_std)
        action = d.sample(seed=key)
        log_prob = d.log_prob(action)
        return action, log_prob

    def transform_log_std(self, log_std: jax.Array):
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return log_std

@dataclass(frozen=True)
class SquashedTanhGaussianPolicy:
    """Tanh-squashed diagonal Gaussian policy (SAC-style)."""

    action_low: jax.Array
    action_high: jax.Array
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    squash_log_std: bool = True

    def dist(self, mean: jax.Array, log_std: jax.Array) -> SquashedTanhGaussianDistribution:
        if self.squash_log_std:
            log_std = self.transform_log_std(log_std)
        return SquashedTanhGaussianDistribution(
            mean=mean,
            log_std=log_std,
            action_low=self.action_low,
            action_high=self.action_high,
        )

    def sample_and_log_prob(
        self, mean: jax.Array, log_std: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if self.squash_log_std:
            log_std = self.transform_log_std(log_std)
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        pre_tanh = mean + jnp.exp(log_std) * noise
        pre_log_prob = _diagonal_gaussian_log_prob(pre_tanh, mean, log_std)
        return squash_tanh_action(
            pre_tanh,
            pre_log_prob,
            self.action_low,
            self.action_high,
        )

    def transform_log_std(self, log_std: jax.Array):
        if self.squash_log_std:
            return squash_log_std_tanh(
                log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return log_std


@dataclass(frozen=True)
class TanhDeterministicPolicy:
    """Deterministic tanh policy (no stochasticity)."""

    action_low: jax.Array
    action_high: jax.Array

    def action(self, pre_tanh: jax.Array) -> jax.Array:
        scale, bias = action_scale_bias(self.action_low, self.action_high)
        return jnp.tanh(pre_tanh) * scale + bias


@dataclass(frozen=True)
class MultivariateBetaPolicy:
    """Independent Beta distribution per action dimension, mapped to [low, high]."""

    action_low: jax.Array
    action_high: jax.Array
    epsilon: float = 1e-4

    def dist(self, alpha: jax.Array, beta: jax.Array) -> AffineBetaDistribution:
        # Ensure positivity of concentration parameters.
        alpha = jax.nn.softplus(alpha) + self.epsilon
        beta = jax.nn.softplus(beta) + self.epsilon

        return AffineBetaDistribution(
            alpha=alpha,
            beta=beta,
            action_low=self.action_low,
            action_high=self.action_high,
            eps=self.epsilon,
        )

    def sample_and_log_prob(
        self, alpha: jax.Array, beta: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        d = self.dist(alpha, beta)
        action = d.sample(seed=key)
        log_prob = d.log_prob(action)
        return action, log_prob
