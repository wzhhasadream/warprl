import copy

import jax
import jax.numpy as jnp
from flax import nnx
from typing import TypeVar, Generic, Sequence
from .normalize_params import project_param
import orbax.checkpoint as ocp
from pathlib import Path
from .dist_head import CategoricalPolicy, QuantilePolicy
from .normalization import OnPolicyRMS, RewardNormalizer, RMS
from .layer import MLP


ModelT = TypeVar("ModelT", bound=nnx.Module)


class Network(nnx.Module, Generic[ModelT]):
    def __init__(
        self,
        model: ModelT,
        opt: nnx.Optimizer | None = None,
        source_model: nnx.Module | None = None,
        tau: float | None = None,
        forward_name: str | None = None,
    ) -> None:
        if (source_model is None) != (tau is None):
            raise ValueError("target_model and tau must be provided together")

        self.model: ModelT = model
        self.opt = opt
        self.tau = tau
        self.source_model = source_model
        self.forward_name = forward_name

    def grad_step(self, grads: nnx.State, max_grad_norm: float | None = None):
        if self.opt is not None:
            if max_grad_norm is not None:
                grads = clip_grads(grads, max_grad_norm)
            self.opt.update(grads)

    def soft_update(self) -> None:
        if self.source_model is not None:
            soft_update(self.source_model, self.model, self.tau)

    def project_param(self) -> None:
        project_param(self.model)

    def __call__(self, *args, **kwargs):
        if self.forward_name is not None:
            return getattr(self.model, self.forward_name)(*args, **kwargs)
        return self.model(*args, **kwargs)

    def save(self, checkpoint_dir: str | Path) -> None:
        """Save parameters and optimizer state to an Orbax checkpoint."""
        checkpoint_dir = Path(checkpoint_dir).absolute()
        # Orbax checkpoints are directories, even when named with a .ckpt suffix.
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        objects = (self.model, self.opt)
        state = nnx.state(objects)

        with ocp.StandardCheckpointer() as ckpt:
            ckpt.save(checkpoint_dir, state)
            ckpt.wait_until_finished()

    def load(self, checkpoint_dir: str | Path) -> None:
        """Load parameters and optimizer state from an Orbax checkpoint."""
        checkpoint_dir = Path(checkpoint_dir).absolute()
        objects = (self.model, self.opt)
        template = nnx.state(objects)
        with ocp.StandardCheckpointer() as ckpt:
            restored = ckpt.restore(checkpoint_dir, template)

        nnx.update(objects, restored)

    def save_onnx(
        self,
        file: str | Path,
        input_shapes: list[tuple[int | str, ...]],
        input_names: Sequence[str] = ("obs",),
        output_names: Sequence[str] = ("actions",),
    ) -> None:
        """Export this network with JAX-style input shape specifications.

        Each tuple describes one positional input. Integer dimensions are
        fixed; string dimensions are symbolic and dynamic. Reusing a symbol
        requires those dimensions to match across inputs.
        """
        import onnx
        from jax2onnx import to_onnx

        out_file = Path(file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        export_model = copy.deepcopy(self.model)
        export_model.eval()
        forward_name = self.forward_name

        def export_forward(*args, **kwargs):
            if forward_name is not None:
                return getattr(export_model, forward_name)(*args, **kwargs)
            return export_model(*args, **kwargs)

        model = to_onnx(export_forward, input_shapes)
        input_renames = {
            value.name: name for value, name in zip(model.graph.input, input_names)
        }
        output_renames = {
            value.name: name for value, name in zip(model.graph.output, output_names)
        }
        for node in model.graph.node:
            node.input[:] = [input_renames.get(x, x) for x in node.input]
            node.output[:] = [output_renames.get(x, x) for x in node.output]
        for value, name in zip(model.graph.input, input_names):
            value.name = name
        for value, name in zip(model.graph.output, output_names):
            value.name = name
        onnx.save(model, out_file)


def soft_update(
    online: nnx.Module,
    target_net: nnx.Module,
    tau: float = 0.005,
) -> None:
    """Soft-update network parameters."""
    online_params = nnx.state(online, nnx.Param)
    target_params = nnx.state(target_net, nnx.Param)
    new_params = jax.tree.map(
        lambda online_p, target_p: tau *
        online_p + (1 - tau) * target_p,
        online_params,
        target_params,
    )
    nnx.update(target_net, new_params)


def clip_grads(grads: nnx.State, max_grad_norm: float) -> nnx.State:
    grad_norm = jnp.sqrt(
        sum(jnp.sum(grad * grad) for grad in jax.tree.leaves(grads))
    )
    scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
    return jax.tree.map(lambda grad: grad * scale, grads)


class Alpha(nnx.Module):
    def __init__(self, initial_value: float = 0.01) -> None:
        if initial_value <= 0.0:
            raise ValueError("initial_value must be positive")

        self.log_alpha = nnx.Param(
            jnp.log(jnp.asarray(initial_value, dtype=jnp.float32))
        )

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha)
