from pathlib import Path
import jax
from flax import nnx
import orbax.checkpoint as ocp

# Check if the object is a nnx graph node
def _is_nnx_graph_node(x: object) -> bool:
    return nnx.graph.is_graph_node(x)


def _to_state_tree(tree):
    return jax.tree.map(
        lambda x: nnx.to_pure_dict(nnx.state(x)) if _is_nnx_graph_node(x) else x,
        tree,
        is_leaf=_is_nnx_graph_node,
    )


def _merge_from_template(template_tree, state_tree):
    def restore_node(obj, restored_state):
        if not _is_nnx_graph_node(obj):
            return restored_state

        state = nnx.state(obj)
        nnx.replace_by_pure_dict(state, restored_state)
        return nnx.merge(nnx.graphdef(obj), state)

    return jax.tree.map(
        restore_node,
        template_tree,
        state_tree,
        is_leaf=_is_nnx_graph_node,
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
