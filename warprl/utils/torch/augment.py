import torch
import torch.nn.functional as F


def augment_observations(
    observations: torch.Tensor,
    padding: int = 4,
) -> torch.Tensor:
    """Random shift with bilinear interpolation, matching DrQv2.

    Observations are channels-last and frame-first: [B, F, H, W, C].
    """
    if observations.ndim != 5:
        raise ValueError("observations must be 5-D [B, F, H, W, C]")
    batch_size, frame_stack, height, width, channels = observations.shape
    assert height == width

    input_dtype = observations.dtype
    images = observations.permute(0, 1, 4, 2, 3).reshape(
        batch_size, frame_stack * channels, height, width
    )
    images = images.float()
    padded = F.pad(
        images,
        (padding,) * 4,
        mode="replicate",
    )

    eps = 1.0 / (height + 2 * padding)
    coordinates = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        height + 2 * padding,
        device=observations.device,
        dtype=torch.float32,
    )[:height]
    coordinates = coordinates.unsqueeze(0).repeat(height, 1).unsqueeze(2)
    grid = torch.cat((coordinates, coordinates.transpose(0, 1)), dim=2)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1, 1, 2),
        device=observations.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (height + 2 * padding)
    augmented = F.grid_sample(
        padded,
        grid + shift,
        padding_mode="zeros",
        align_corners=False,
    )
    augmented = augmented.reshape(
        batch_size, frame_stack, channels, height, width
    ).permute(0, 1, 3, 4, 2)
    if input_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        augmented = torch.round(augmented).to(dtype=input_dtype)
    return augmented
