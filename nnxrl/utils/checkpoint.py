from pathlib import Path
import jax
from flax import nnx
import orbax.checkpoint as ocp


def _to_state_tree(tree):
    return jax.tree.map(
        lambda x: nnx.state(x) if isinstance(x, nnx.Object) else x,
        tree,
        is_leaf=lambda x: isinstance(x, nnx.Object),
    )


def _merge_from_template(template_tree, state_tree):
    return jax.tree.map(
        lambda obj, st: nnx.merge(nnx.graphdef(obj), st)
        if isinstance(obj, nnx.Object) else st,
        template_tree,
        state_tree,
        is_leaf=lambda x: isinstance(x, nnx.Object),
    )


def save_states(path: str, state_dict: dict[str, object]) -> None:
    path = Path(path).resolve()
    state_tree = _to_state_tree(state_dict)

    with ocp.StandardCheckpointer() as ckpt:
        ckpt.save(path, state_tree)
        ckpt.wait_until_finished()


def load_states(
    path: str,
    model_dict: dict[str, object],
) -> dict[str, object]:
    path = Path(path).resolve()
    abstract_state_tree = _to_state_tree(model_dict)

    with ocp.StandardCheckpointer() as ckpt:
        restored_state_tree = ckpt.restore(path, abstract_state_tree)

    return _merge_from_template(model_dict, restored_state_tree)
