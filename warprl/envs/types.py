from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from typing import Any, Union

if TYPE_CHECKING:
    import jax
    import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, "jax.Array", "torch.Tensor"]