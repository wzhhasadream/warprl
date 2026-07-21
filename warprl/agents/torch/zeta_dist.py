import torch


def build_truncated_zeta_cdf(
    mu: float | torch.Tensor = 2.0,
    max_n: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    values = torch.arange(1, max_n + 1, dtype=dtype, device=device)
    probs = values.pow(-torch.as_tensor(mu, dtype=dtype, device=device))
    return probs.cumsum(0) / probs.sum()


def sample_integer_from_cdf(
    cdf: torch.Tensor,
    shape: tuple[int, ...] = (),
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.rand(shape, dtype=cdf.dtype, device=cdf.device)
    return torch.searchsorted(cdf, noise, right=True).long() + 1


def sample_truncated_zeta(
    mu: float | torch.Tensor = 2.0,
    max_n: int = 16,
    shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    cdf = build_truncated_zeta_cdf(mu, max_n, dtype, device)
    return sample_integer_from_cdf(cdf, shape, noise=noise)
