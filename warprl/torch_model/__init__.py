from .dist_head import CategoricalPolicy, QuantilePolicy
from .normalization import RMS, RewardNormalizer
from .normalize_params import project_param
import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math
from typing import Generic, TypeVar, Any, Sequence
from pathlib import Path
from .flashsac import FlashSACActor, FlashSACDoubleCritic
from torch.amp.grad_scaler import GradScaler
ModelT = TypeVar("ModelT", bound=nn.Module)

compile_mode = "max-autotune"

__all__ = [
    "Alpha",
    "CategoricalPolicy",
    "FlashSACActor",
    "FlashSACDoubleCritic",
    "Network",
    "QuantilePolicy",
    "RMS",
    "RewardNormalizer",
    "project_param",
]


class Network(nn.Module, Generic[ModelT]):
    def __init__(
        self,
        model: ModelT,
        opt: Optimizer | None = None,
        source_model: nn.Module | None = None,
        tau: float | None = None,
        scheduler: LRScheduler | None = None,
        grad_scaler: GradScaler | None = None,
        forward_name: str | None = None,
    ) -> None:
        super().__init__()
        if (source_model is None) != (tau is None):
            raise ValueError("source_model and tau must be provided together")

        self.model: ModelT = model
        self.opt = opt
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.forward_name = forward_name

        if opt is not None:
            self.opt_step = torch.compile(opt.step, mode=compile_mode)


        self.target_params: tuple[torch.Tensor, ...] | None = None
        self.source_params: tuple[torch.Tensor, ...] | None = None
        self.tau: float | None = None

        if source_model is not None and tau is not None:
            self.target_params = tuple(model.parameters())
            self.source_params = tuple(source_model.parameters())
            self.tau = tau

    def grad_step(self, loss: torch.Tensor) -> None:
        if self.opt is None:
            raise RuntimeError("grad_step requires an optimizer")

        self.opt.zero_grad(set_to_none=True)    # the only one that can't be compile

        if self.grad_scaler is None:
            loss.backward()
            self.opt_step()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()


    @torch.compile(mode=compile_mode)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.forward_name is None:
            return self.model(*args, **kwargs)
        return getattr(self.model, self.forward_name)(*args, **kwargs)



    @torch.no_grad()
    @torch.compile(mode=compile_mode)
    def project_param(self) -> None:
        project_param(self.model)


    @torch.no_grad()
    @torch.compile(mode=compile_mode)
    def soft_update(self) -> None:
        assert self.target_params is not None
        assert self.source_params is not None
        assert self.tau is not None

        torch._foreach_lerp_(self.target_params, self.source_params, self.tau)

    def save(self, file: str | Path) -> None:
        out_file = Path(file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict() if self.opt is not None else None,
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "grad_scaler": (
                self.grad_scaler.state_dict()
                if self.grad_scaler is not None
                else None
            )
        }
        torch.save(state, out_file)

    def load(self, file: str | Path) -> None:
        device = next(self.model.parameters()).device
        state = torch.load(file, map_location=device)
        self.model.load_state_dict(state["model"])
        
        if self.opt is not None and state["opt"] is not None:
            self.opt.load_state_dict(state["opt"])

        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])

        if self.grad_scaler is not None and state["grad_scaler"] is not None:
            self.grad_scaler.load_state_dict(state["grad_scaler"])


    def save_onnx(
        self,
        file: str | Path,
        input_shapes: list[tuple[int | str, ...]],
        external_data: bool = False,
        input_names: Sequence[str] = ("obs",),
        output_names: Sequence[str] = ("actions",),
    ) -> None:
        """Export this network with JAX-style input shape specifications.

        Each tuple describes one positional input. Integer dimensions are
        fixed; string dimensions are symbolic and dynamic. Reusing a symbol
        requires those dimensions to match across inputs, e.g. ``[("B", 480)]``
        exports an actor that accepts ``[B, 480]`` observations.
        """
        out_file = Path(file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        parameter = next(self.model.parameters())
        symbols: dict[str, torch.export.Dim] = {}
        example_inputs = []
        dynamic_shapes = []
        for shape in input_shapes:
            example_inputs.append(
                torch.zeros(
                    tuple(1 if isinstance(dim, str) else dim for dim in shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
            dynamic_shapes.append(
                {
                    axis: symbols.setdefault(dim, torch.export.Dim(dim))
                    for axis, dim in enumerate(shape)
                    if isinstance(dim, str)
                }
            )
        was_training = self.training
        self.eval()
        try:
            torch.onnx.export(
                self,
                tuple(example_inputs),
                out_file,
                dynamic_shapes=(tuple(dynamic_shapes),),
                external_data=external_data,
                input_names=input_names,
                output_names=output_names
            )
        finally:
            self.train(was_training)

        




class Alpha(nn.Module):
    def __init__(self, init_value: float = 0.01) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.as_tensor(math.log(init_value), dtype=torch.float32))

    def forward(self):
        return torch.exp(self.log_alpha)
