"""Dense / voxel grid sampling helpers shared by the mesh extractors.

* :func:`generate_dense_grid_points` produces a ``numpy`` grid of
  coordinates used by the naive marching-cubes extractor.
* :func:`create_3d_grid` and :func:`generate_voxel_samples` produce
  ``torch`` grids and jittered per-voxel samples used by the coarse-to-fine
  evaluator in :mod:`cube_part.utils.field`.
"""

from typing import Tuple

import numpy as np
import torch


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution_base: float,
    indexing: str = "ij",
):
    """Create a flattened ``(N, 3)`` grid of dense sample points (numpy)."""
    length = bbox_max - bbox_min
    num_cells = np.exp2(resolution_base)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]
    return xyz, grid_size, length


def create_3d_grid(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    grid_resolution: Tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a 3D grid of voxel-centered sample coordinates inside an AABB.

    Args:
        bbox_min: ``(3,)`` minimum corner of the bounding box.
        bbox_max: ``(3,)`` maximum corner of the bounding box.
        grid_resolution: ``(Dx, Dy, Dz)`` number of samples per dimension.
        dtype: dtype of the produced grid.

    Returns:
        grid: ``(Dx*Dy*Dz, 3)`` voxel-center coordinates.
        voxel_size: ``(3,)`` voxel side length.
    """
    device = bbox_min.device
    _grid_resolution = torch.tensor(grid_resolution, dtype=dtype, device=device)
    bbox_min = bbox_min.to(dtype)
    bbox_max = bbox_max.to(dtype)

    voxel_size = (bbox_max - bbox_min) / _grid_resolution
    half_voxel = voxel_size / 2

    x = torch.linspace(
        bbox_min[0] + half_voxel[0],
        bbox_max[0] - half_voxel[0],
        grid_resolution[0],
        device=device,
        dtype=dtype,
    )
    y = torch.linspace(
        bbox_min[1] + half_voxel[1],
        bbox_max[1] - half_voxel[1],
        grid_resolution[1],
        device=device,
        dtype=dtype,
    )
    z = torch.linspace(
        bbox_min[2] + half_voxel[2],
        bbox_max[2] - half_voxel[2],
        grid_resolution[2],
        device=device,
        dtype=dtype,
    )
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    return grid, voxel_size


def generate_voxel_samples(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    grid_resolution: Tuple[int, int, int],
    K: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate ``K`` deterministically-jittered random samples per voxel.

    Args:
        bbox_min: ``(3,)`` minimum corner of the bounding box.
        bbox_max: ``(3,)`` maximum corner of the bounding box.
        grid_resolution: ``(Dx, Dy, Dz)`` number of voxels per dimension.
        K: number of random samples per voxel.

    Returns:
        Tensor of shape ``(Dx, Dy, Dz, K, 3)``.
    """
    device = bbox_min.device

    voxel_centers, voxel_size = create_3d_grid(
        bbox_min, bbox_max, grid_resolution, dtype
    )
    num_voxels = voxel_centers.shape[0]

    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    random_offsets = (
        torch.rand(
            (num_voxels, K, 3),
            generator=rng,
            device=device,
            dtype=dtype,
        )
        - 0.5
    ) * voxel_size
    sampled_positions = voxel_centers[:, None, :] + random_offsets
    return sampled_positions.reshape(*grid_resolution, K, 3)
