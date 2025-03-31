import argparse
import logging

import numpy as np
import torch
import trimesh

from cube3d.inference.utils import load_config, load_model_weights, parse_structured, select_device
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder

MESH_SCALE = 0.96


def rescale(vertices: np.ndarray, mesh_scale: float = MESH_SCALE) -> np.ndarray:
    """Rescale the vertices to a cube, e.g., [-1, -1, -1] to [1, 1, 1] when mesh_scale=1.0"""
    vertices = vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices


def load_scaled_mesh(file_path: str) -> trimesh.Trimesh:
    """
    Load a mesh and scale it to a unit cube, and clean the mesh.
    Parameters:
        file_obj: str | IO
        file_type: str
    Returns:
        mesh: trimesh.Trimesh
    """
    mesh: trimesh.Trimesh = trimesh.load(file_path, force="mesh")
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces after cleaning")
    mesh.vertices = rescale(mesh.vertices)
    return mesh


def load_and_process_mesh(file_path: str, n_samples: int = 8192):
    """
    Loads a 3D mesh from the specified file path, samples points from its surface,
    and processes the sampled points into a point cloud with normals.
    Args:
        file_path (str): The file path to the 3D mesh file.
        n_samples (int, optional): The number of points to sample from the mesh surface. Defaults to 8192.
    Returns:
        torch.Tensor: A tensor of shape (1, n_samples, 6) containing the processed point cloud.
                        Each point consists of its 3D position (x, y, z) and its normal vector (nx, ny, nz).
    """

    mesh = load_scaled_mesh(file_path)
    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    point_cloud = np.concatenate(
        [positions, normals], axis=1
    )  # Shape: (num_samples, 6)
    point_cloud = torch.from_numpy(point_cloud.reshape(1, -1, 6)).float()
    return point_cloud


@torch.inference_mode()
def run_shape_decode(
    shape_model: OneDAutoEncoder,
    output_ids: torch.Tensor,
    resolution_base: float = 8.0,
    chunk_size: int = 100_000,
):
    """
    Decodes the shape from the given output IDs and extracts the geometry.
    Args:
        shape_model (OneDAutoEncoder): The shape model.
        output_ids (torch.Tensor): The tensor containing the output IDs.
        resolution_base (float, optional): The base resolution for geometry extraction. Defaults to 8.43.
        chunk_size (int, optional): The chunk size for processing. Defaults to 100,000.
    Returns:
        tuple: A tuple containing the vertices and faces of the mesh.
    """
    shape_ids = (
        output_ids[:, : shape_model.cfg.num_encoder_latents, ...]
        .clamp_(0, shape_model.cfg.num_codes - 1)
        .view(-1, shape_model.cfg.num_encoder_latents)
    )
    latents = shape_model.decode_indices(shape_ids)
    mesh_v_f, _ = shape_model.extract_geometry(
        latents,
        resolution_base=resolution_base,
        chunk_size=chunk_size,
        use_warp=True,
    )
    return mesh_v_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cube shape encode and decode example script"
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        required=True,
        help="Path to the input mesh file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--recovered-mesh-path",
        type=str,
        default="recovered_mesh.obj",
        help="Path to save the recovered mesh file.",
    )
    args = parser.parse_args()
    device = select_device()
    logging.info(f"Using device: {device}")

    cfg = load_config(args.config_path)

    shape_model = OneDAutoEncoder(
        parse_structured(OneDAutoEncoder.Config, cfg.shape_model)
    )
    load_model_weights(
        shape_model,
        args.shape_ckpt_path,
    )
    shape_model = shape_model.eval().to(device)
    point_cloud = load_and_process_mesh(args.mesh_path)
    output = shape_model.encode(point_cloud.to(device))
    indices = output[3]["indices"]
    print("Got the following shape indices:")
    print(indices)
    print("Indices shape: ", indices.shape)
    mesh_v_f = run_shape_decode(shape_model, indices)
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(args.recovered_mesh_path)
