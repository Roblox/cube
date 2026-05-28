from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import trimesh


def rescale(vertices: np.ndarray, mesh_scale: int = 0.96):
    """
    Rescale the vertices to the range [-0.5, 0.5] * mesh_scale
    """
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, 1.0 / scale


def load_mesh(mesh_path: Union[Path, str], mesh_scale: int = 0.96):
    """
    Returns a mesh with vertices normalized in the range (-0.5, 0.5)
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices, center, scale = rescale(mesh.vertices, mesh_scale=mesh_scale)
    mesh.vertices = vertices

    return mesh, center, 1.0 / scale


def sample_coarse_points(mesh: trimesh.Trimesh, num_points: int):
    """
    Sample points uniformly from the mesh.
    Parameters:
        mesh: The trimesh mesh.
        num_points: The number of points to sample.
    Returns:
        samples: The sampled points.
        normals: The normals of the sampled points.
    """
    samples, sample_faces = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[sample_faces]

    # handle nan and inf
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    normals = np.nan_to_num(normals, nan=0.0, posinf=1.0, neginf=-1.0)

    return samples, normals


def sample_sharp_points(
    mesh: trimesh.Trimesh, num_points: int, threshold: float = 0.985
):
    """
    Sample points near sharp edges from the mesh.
    Parameters:
        mesh: The trimesh mesh.
        num_points: The number of points to sample.
        threshold: The threshold for the sharp edges.
    Returns:
        samples: The sampled points.
        normals: The normals of the sampled points.
    """
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals  # [Nv, 3]
    face_normals = mesh.face_normals  # [Nf, 3]
    faces = mesh.faces  # [Nf, 3]

    # get the 3 vertex normals for every faces
    face_vertex_normals = vertex_normals[faces.reshape(-1), :].reshape(
        faces.shape[0], 3, 3
    )  # [Nf, 3, 3]

    # calculate the minimal dot product of the vertex normals and corresponding face normals
    dot = np.sum(face_vertex_normals * face_normals[:, None, :], axis=-1)
    vertex_normals_dot_face_normals = np.ones(vertex_normals.shape[0])
    for i in range(3):
        vertex_normals_dot_face_normals[faces[:, i]] = np.minimum(
            vertex_normals_dot_face_normals[faces[:, i]], dot[:, i]
        )

    sharp_vertex_mask = vertex_normals_dot_face_normals < threshold

    # collect edge
    edge_a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    edge_b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    sharp_edge = sharp_vertex_mask[edge_a] * sharp_vertex_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    # if no sharp edges, fallback to coarse
    if edge_a.shape[0] == 0 or edge_b.shape[0] == 0:
        return sample_coarse_points(mesh, num_points=num_points)

    sharp_verts_a = vertices[edge_a]
    sharp_verts_b = vertices[edge_b]
    sharp_verts_an = vertex_normals[edge_a]
    sharp_verts_bn = vertex_normals[edge_b]

    # calc weights based on length of edges
    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    # randomly pick edges and interpolate between the two endpoints
    random_number = np.random.rand(num_points)
    w = np.random.rand(num_points, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def sample_surface(
    mesh: trimesh.Trimesh,
    num_samples: int = 128_000,
    fps_multiplier: int = 5,
    kd_tree_height: int = 6,
):
    """
    Sample coarse-only surface points (+normals) with farthest-point thinning.

    Parameters:
        mesh: The trimesh mesh.
        num_samples: The final number of points to return.
        fps_multiplier: Oversampling factor before FPS (dev default: 5).
        kd_tree_height: ``h`` parameter for the FPS kd-line sampler
            (dev default: 6).

    Returns:
        coarse_surface: ``(num_samples, 6)`` array of ``[xyz, nxnynz]``.
    """
    import fpsample

    points, face_index = mesh.sample(
        num_samples * fps_multiplier, return_index=True
    )
    points = np.asarray(points)
    normals = np.asarray(mesh.face_normals)[face_index]

    indices = fpsample.bucket_fps_kdline_sampling(
        points, num_samples, h=kd_tree_height
    )
    points = points[indices]
    normals = normals[indices]

    coarse_surface = np.concatenate([points, normals], axis=1)
    return coarse_surface


def marching_cubes_with_warp(
    grid_logits: Union[np.ndarray, torch.Tensor],
    level: Optional[float] = None,
    device: Union[str, torch.device] = "cuda",
    max_verts: int = 3_000_000,
    max_tris: int = 3_000_000,
):
    """
    Extract a mesh from the grid of logits using the marching cubes algorithm
    Parameters
    ----------
    grid_logits : Union[np.ndarray, torch.Tensor]
        3D grid of logits
    level : float
        Threshold level for the marching cubes algorithm, default is the mean of the grid logits if None
    device : Union[str, torch.device]
        Device to run the marching cubes algorithm on. Can be "cuda", "cuda:x" or torch.device
    max_verts : int
        Maximum number of vertices in the mesh, default is 3_000_000
    max_tris : int
        Maximum number of triangles in the mesh, default is 3_000_000

    Returns
    -------
    vertices : np.ndarray
        Vertices of the mesh, shape (V, 3)
    faces : np.ndarray
        Faces of the mesh, shape (F, 3)
    """
    import warp as wp

    if isinstance(device, torch.device):
        device = str(device)

    assert grid_logits.ndim == 3
    if "cuda" in device:
        assert wp.is_cuda_available()
    else:
        raise ValueError(
            f"Device {device} is not supported for marching_cubes_with_warp"
        )
    if level is None:
        level = (grid_logits.max() + grid_logits.min()) / 2

    dim = grid_logits.shape[0]
    if isinstance(grid_logits, np.ndarray):
        field = wp.from_torch(torch.tensor(grid_logits))
    else:
        field = wp.from_torch(grid_logits)

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        iso = wp.MarchingCubes(
            nx=dim,
            ny=dim,
            nz=dim,
            max_verts=int(max_verts),
            max_tris=int(max_tris),
            device=device,
        )
        iso.surface(field=field, threshold=level)
    vertices = iso.verts.numpy()
    faces = iso.indices.numpy().reshape(-1, 3)
    return vertices, faces
