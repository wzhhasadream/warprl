import torch
import torch.nn.functional as F


def random_shift(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply independent edge-padded random shifts to NCHW images."""
    batch_size, _, height, width = images.shape
    images = F.pad(images, (padding,) * 4, mode="replicate")

    eps = 1.0 / (height + 2 * padding)
    coordinates = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        height + 2 * padding,
        device=images.device,
        dtype=images.dtype,
    )[:height]
    coordinates = coordinates.unsqueeze(0).repeat(height, 1).unsqueeze(2)
    grid = torch.cat((coordinates, coordinates.transpose(0, 1)), dim=2)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1, 1, 2),
        device=images.device,
        dtype=images.dtype,
    )
    shift *= 2.0 / (height + 2 * padding)
    return F.grid_sample(
        images,
        grid + shift,
        padding_mode="zeros",
        align_corners=False,
    )


def augment_observations(
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    padding: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment current and next pixel observations with independent shifts."""
    return (
        random_shift(observations, padding),
        random_shift(next_observations, padding),
    )
