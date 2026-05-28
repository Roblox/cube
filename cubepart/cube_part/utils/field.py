"""Implicit-field coarse-to-fine evaluator and supporting volume filters.

The pipeline evaluates the autoencoder's implicit field on a coarse grid,
identifies the voxels near the surface, then re-evaluates only those
voxels at full resolution. :class:`ImplicitFieldCoarseToFineEvaluator`
orchestrates that loop; the helpers above it (``separable_max_filter``,
``upsample``, ``upsample_binary_mask``, ``compute_implicit_grid_and_mask``)
implement the volume operations it needs.
"""

import logging
from typing import Callable, Tuple

import torch
from torch.nn import functional as F

from cube_part.utils.grid import create_3d_grid, generate_voxel_samples


def separable_max_filter(volume: torch.Tensor, filter_size: int = 7) -> torch.Tensor:
    """Apply a 3D separable max filter (1D max-pool along each axis).

    Args:
        volume: ``(B, D, H, W)`` batched 3D volume.
        filter_size: kernel size, must be odd. Defaults to 7.

    Returns:
        Filtered volume of the same shape and dtype as the input.
    """
    assert filter_size % 2 == 1, "filter_size must be an odd number."
    pad_size = filter_size // 2

    input_dtype = volume.dtype
    if not volume.is_floating_point():
        volume = volume.to(torch.float32)

    volume = volume.unsqueeze(1)  # (B, 1, D, H, W)

    temp_x = F.max_pool3d(
        volume,
        kernel_size=(filter_size, 1, 1),
        stride=1,
        padding=(pad_size, 0, 0),
    )
    temp_y = F.max_pool3d(
        temp_x,
        kernel_size=(1, filter_size, 1),
        stride=1,
        padding=(0, pad_size, 0),
    )
    temp_z = F.max_pool3d(
        temp_y,
        kernel_size=(1, 1, filter_size),
        stride=1,
        padding=(0, 0, pad_size),
    )

    return temp_z.squeeze(1).to(input_dtype)


def upsample(grid: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
    """Nearest-neighbor upsample a 3D (or batched 4D) float grid."""
    if grid.dim() not in (3, 4):
        raise ValueError("Input grid must have shape (D, W, H) or (B, D, W, H)")

    added_batch = grid.dim() == 3
    if added_batch:
        grid = grid.unsqueeze(0)
    grid = grid.unsqueeze(1)  # (B, 1, D, W, H)

    d_out, w_out, h_out = target_size
    grid_upsampled = F.interpolate(
        grid, size=(d_out, w_out, h_out), mode="nearest-exact"
    )
    grid_upsampled = grid_upsampled.squeeze(1)
    if added_batch:
        grid_upsampled = grid_upsampled.squeeze(0)
    return grid_upsampled


def upsample_binary_mask(
    mask: torch.Tensor, target_size: Tuple[int, int, int]
) -> torch.Tensor:
    """Nearest-neighbor upsample a 3D (or batched 4D) boolean mask."""
    if mask.dim() not in (3, 4):
        raise ValueError("Input mask must have shape (D, W, H) or (B, D, W, H)")

    added_batch = mask.dim() == 3
    if added_batch:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1)  # (B, 1, D, W, H)

    d_out, w_out, h_out = target_size
    mask_upsampled = F.interpolate(
        mask.float(), size=(d_out, w_out, h_out), mode="nearest-exact"
    )
    mask_upsampled = mask_upsampled.squeeze(1)
    if added_batch:
        mask_upsampled = mask_upsampled.squeeze(0)
    return (mask_upsampled > 0.5).to(torch.bool)


def compute_implicit_grid_and_mask(
    coarse_samples: torch.Tensor, tau: float, dilate_radius: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a coarse implicit grid and a refinement mask from voxel samples.

    Args:
        coarse_samples: ``(B, D, W, H, K)`` float tensor of K samples per voxel
            (typically SDF or occupancy values).
        tau: 'close to surface' threshold.
        dilate_radius: dilation radius applied to the candidate masks.

    Returns:
        implicit_grid: ``(B, D, W, H)`` field from the first sample per voxel.
        mask: ``(B, D, W, H)`` bool mask of voxels selected for refinement.
    """
    close_to_zero = (coarse_samples.abs() < tau).any(dim=-1)
    close_to_zero = separable_max_filter(close_to_zero, filter_size=dilate_radius)

    inside = (coarse_samples >= 0).any(dim=-1)
    outside = (coarse_samples <= 0).any(dim=-1)
    implicit_grid = coarse_samples[..., 0]

    inside = separable_max_filter(inside, filter_size=dilate_radius)
    outside = separable_max_filter(outside, filter_size=dilate_radius)
    inside_and_outside = inside & outside

    mask = close_to_zero | inside_and_outside
    return implicit_grid, mask


class ImplicitFieldCoarseToFineEvaluator:
    """Evaluate an implicit occupancy/SDF field with adaptive refinement.

    First evaluates a coarse grid, then upsamples and re-evaluates a fine
    grid only inside the mask of voxels marked as near-surface.
    """

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        fine_grid_resolution: Tuple[int, int, int],
        embed_positions_func,
        coarse_grid_resolution: Tuple[int, int, int] = (64, 64, 64),
        K: int = 5,
    ):
        """Set up the evaluator and precompute sample positions / embeddings.

        Args:
            bbox_min: ``(3,)`` float min corner of the bounding box.
            bbox_max: ``(3,)`` float max corner of the bounding box.
            fine_grid_resolution: ``(Dx, Dy, Dz)`` final-grid resolution.
            embed_positions_func: callable mapping ``(N, 3)`` positions to
                ``(N, E)`` embeddings (pass identity if no embedding is needed).
            coarse_grid_resolution: coarse grid resolution. Defaults to
                ``(64, 64, 64)``.
            K: samples per coarse voxel. Defaults to 5.
        """
        coarse_positions = generate_voxel_samples(
            bbox_min, bbox_max, coarse_grid_resolution, K
        )
        fine_positions, _ = create_3d_grid(bbox_min, bbox_max, fine_grid_resolution)

        self.coarse_positions_embedded = embed_positions_func(
            coarse_positions.reshape(-1, 3)
        )
        self.fine_positions_embedded = embed_positions_func(fine_positions).reshape(
            *fine_grid_resolution, -1
        )

        self.coarse_grid_resolution = coarse_grid_resolution
        self.K = K
        self.fine_grid_resolution = fine_grid_resolution
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def evaluate(
        self,
        eval_func_coarse: Callable[[torch.Tensor], torch.Tensor],
        eval_func_fine: Callable[[torch.Tensor, int], torch.Tensor],
        tau: float = 5.0,
        fine_dilate_radius: int = 1,
        coarse_dilate_radius: int = 3,
    ) -> torch.Tensor:
        """Run the coarse-to-fine implicit field evaluation.

        Args:
            eval_func_coarse: evaluates the implicit fields for **all** batch
                elements at the same input positions. Takes the precomputed
                coarse position embeddings; returns ``(B, N)``.
            eval_func_fine: evaluates the implicit field for one batch element
                at a time at a per-batch subset of positions. Takes the
                masked fine position embeddings and a batch index; returns
                ``(N,)``.
            tau: 'close to surface' threshold for the coarse mask.
            fine_dilate_radius: dilation radius for the fine mask.
            coarse_dilate_radius: dilation radius for the coarse mask.

        Returns:
            Tensor of shape ``(B, Dx, Dy, Dz)``.
        """
        coarse_samples = eval_func_coarse(self.coarse_positions_embedded).reshape(
            -1,
            *self.coarse_grid_resolution,
            self.K,
        )  # (B, D, W, H, K)
        implicit_grids, masks = compute_implicit_grid_and_mask(
            coarse_samples, tau, coarse_dilate_radius
        )  # (B, D, W, H), (B, D, W, H)

        implicit_grids = upsample(implicit_grids, self.fine_grid_resolution)
        masks = upsample_binary_mask(masks, self.fine_grid_resolution)

        if fine_dilate_radius > 1:
            masks = separable_max_filter(masks, filter_size=fine_dilate_radius)
        eval_masks = separable_max_filter(masks, filter_size=3)

        for batch_idx, (implicit_grid, eval_mask) in enumerate(
            zip(implicit_grids.unbind(0), eval_masks.unbind(0))
        ):
            fine_positions_embedded_masked = self.fine_positions_embedded[
                eval_mask
            ].reshape(-1, self.fine_positions_embedded.shape[-1])

            fine_samples = eval_func_fine(fine_positions_embedded_masked, batch_idx)
            implicit_grid[eval_mask] = fine_samples

            title = "----- evaluate_implicit_field_coarse_to_fine -----"
            text = "\n\n" + title + "\n"
            total_samples = coarse_samples[batch_idx].numel() + fine_samples.numel()
            total_samples_wo_culling = implicit_grid.numel()
            text += (
                f"occupancy: {eval_mask.sum().item() / eval_mask.numel() * 100:.2f}%\n"
            )
            text += f"num coarse samples: {coarse_samples[batch_idx].numel()}\n"
            text += f"num fine samples: {fine_samples.numel()}\n"
            text += f"total samples (coarse + fine): {total_samples}\n"
            text += f"total samples w/o culling: {total_samples_wo_culling}\n"
            text += f"theoretical speed-up: {total_samples_wo_culling / total_samples:.1f}\n"
            text += "-" * len(title) + "\n"
            logging.debug(text)

        implicit_grids[:, 0, :, :] = -1
        implicit_grids[:, -1, :, :] = -1
        implicit_grids[:, :, 0, :] = -1
        implicit_grids[:, :, -1, :] = -1
        implicit_grids[:, :, :, 0] = -1
        implicit_grids[:, :, :, -1] = -1

        return implicit_grids
