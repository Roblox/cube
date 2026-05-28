import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from skimage import measure
from torch.nn import functional as F
from tqdm import tqdm

from cube_part.utils.base import BaseModule
from cube_part.utils.field import ImplicitFieldCoarseToFineEvaluator
from cube_part.utils.grid import generate_dense_grid_points
from cube_part.utils.mesh import marching_cubes_with_warp

logger = logging.getLogger(__name__)
DEFAULT_EXTRACT_GEOMETRY_BOUNDS = 1.05


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.in_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, **kwargs):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, **kwargs)

    elif embed_type == "siren":
        embedder_obj = Siren(
            in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim
        )
    else:
        raise ValueError(f"Embedding type: {embed_type} is not supported.")
    return embedder_obj


###################### AutoEncoder
class AutoEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_latents: int = 256
        embed_dim: int = 64
        width: int = 768

    cfg: Config

    def encode(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def query(
        self,
        queries: torch.FloatTensor,
        latents: torch.FloatTensor,
        skip_query_transform: bool = False,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @torch.no_grad()
    def extract_geometry(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = (
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
        ),
        resolution_base: float = 9.0,
        chunk_size: int = 2_000_000,
        use_warp: bool = False,
        fn_name: str = "extract_geometry_naive",
        **kwargs,
    ):
        if not hasattr(self, fn_name):
            raise ValueError(f"{self.__class__.__name__}.{fn_name} does not exist.")
        extract_geometry_fn = getattr(self, fn_name)
        return extract_geometry_fn(
            latents, bounds, resolution_base, chunk_size, use_warp, **kwargs
        )

    @torch.no_grad()
    def extract_geometry_naive(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = (
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
        ),
        resolution_base: float = 9.0,
        chunk_size: int = 2_000_000,
        use_warp: bool = False,
        **kwargs,
    ):
        """
        Extracts the geometry from the latents

        Parameters
        ----------
        latents : torch.FloatTensor
            Latent codes, which are input to the last decoder layer
        bounds : Union[Tuple[float], List[float], float], optional
            Bounding box, by default (-DEFAULT_EXTRACT_GEOMETRY_BOUNDS, -DEFAULT_EXTRACT_GEOMETRY_BOUNDS, -DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS)
        resolution_base: float, optional
            Octree depth, by default 9.0
        chunk_size : int, optional
            Size of the samples used for surface decoding, by default 1_000_000
        use_warp : bool, optional
            Use warp-lang, which has CUDA acceleration, by default False
        octree_depth : float, optional
            Octree depth, if set this will override resolution_base, going to be deprecated

        Returns
        -------
        mesh_v_f: list of (vertices, faces), len(mesh_v_f) == latents.shape[0]
        has_surface: np.array, shape as (latents.shape[0], )
        """
        if "octree_depth" in kwargs:
            resolution_base = kwargs["octree_depth"]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution_base=resolution_base,
            indexing="ij",
        )
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        batch_logits = []

        progress_bar = tqdm(
            range(0, xyz_samples.shape[0], chunk_size),
            desc=f"{type(self)} extracting geometry",
            unit="chunk",
        )

        for start in progress_bar:
            queries = xyz_samples[start : start + chunk_size, :]

            # pad last chunk of queries to the same length to avoid torch.compile re-compiling this block
            num_queries = queries.shape[0]
            if start > 0 and num_queries < chunk_size:
                queries = F.pad(queries, [0, 0, 0, chunk_size - num_queries])
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size).to(latents)

            logits = self.query(batch_queries, latents)[:, :num_queries]
            batch_logits.append(logits)

        grid_logits = (
            torch.cat(batch_logits, dim=1)
            .detach()
            .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
            .float()
        )

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                warp_success = False
                if use_warp and getattr(self, "training", False) == False:
                    # make sure we disable warp when training
                    # since warp can cause memory illegal access and also crash the kernel
                    # causing the following cuda operations to fail
                    # it's ok to run this at inference time, since the following operations are not cuda
                    try:
                        vertices, faces = marching_cubes_with_warp(
                            grid_logits[i],
                            level=0.0,
                            device=grid_logits.device,
                        )
                        warp_success = True
                    except Exception as e:
                        logging.warning(
                            f"Warning: error in marching cubes with warp: {e}"
                        )
                        warp_success = False  # Fall back to CPU version

                if not warp_success:
                    vertices, faces, _, _ = measure.marching_cubes(
                        grid_logits[i].cpu().numpy(), 0, method="lewiner"
                    )

                vertices = vertices / grid_size * bbox_size + bbox_min
                faces = faces[:, [2, 1, 0]]
                mesh_v_f.append(
                    (vertices.astype(np.float32), np.ascontiguousarray(faces))
                )
                has_surface[i] = True
            except Exception as e:
                logging.error(f"Error: error in extract_geometry: {e}")
                mesh_v_f.append((None, None))
                has_surface[i] = False

        return mesh_v_f, has_surface

    @torch.no_grad()
    def extract_geometry_coarse_to_fine(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = (
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            -DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
            DEFAULT_EXTRACT_GEOMETRY_BOUNDS,
        ),
        resolution_base: float = 9.0,
        chunk_size: int = 2_000_000,
        use_warp: bool = False,
        precompute_embeddings=False,
        **kwargs,
    ):
        """
        Extracts the geometry from the latents

        Parameters
        ----------
        latents : torch.FloatTensor
            Latent codes, which are input to the last decoder layer
        bounds : Union[Tuple[float], List[float], float], optional
            Bounding box, by default (-DEFAULT_EXTRACT_GEOMETRY_BOUNDS, -DEFAULT_EXTRACT_GEOMETRY_BOUNDS, -DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS, DEFAULT_EXTRACT_GEOMETRY_BOUNDS)
        resolution_base: float, optional
            Octree depth, by default 9.0
        chunk_size : int, optional
            Size of the samples used for surface decoding, by default 1_000_000
        use_warp : bool, optional
            Use warp-lang, which has CUDA acceleration, by default False
        precompute_embeddings : bool, optional
            Precompute query embeddings, resolution_base is limited to 8.2 because of high memory consumption, by default False
        octree_depth : float, optional
            Octree depth, if set this will override resolution_base, going to be deprecated

        Returns
        -------
        mesh_v_f: list of (vertices, faces), len(mesh_v_f) == latents.shape[0]
        has_surface: np.array, shape as (latents.shape[0], )
        """

        if "octree_depth" in kwargs:
            resolution_base = kwargs["octree_depth"]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        R = int(2**resolution_base) + 1
        fine_grid_resolution = (R, R, R)

        device = latents.device
        bbox_min = torch.tensor(bounds[0:3], device=device, dtype=torch.float32)
        bbox_max = torch.tensor(bounds[3:6], device=device, dtype=torch.float32)
        batch_size = latents.shape[0]

        def embed_positions(positions):
            if not precompute_embeddings:
                return positions
            embedded_positions = None
            for start in range(0, positions.shape[0], chunk_size):
                positions_chunk = positions[start : start + chunk_size]
                actual_chunk_size = positions_chunk.shape[0]
                if actual_chunk_size < chunk_size:
                    positions_chunk = F.pad(
                        positions_chunk, [0, 0, 0, chunk_size - actual_chunk_size]
                    )
                positions_chunk = positions_chunk.unsqueeze(0)
                embedded_positions_chunk = self.occupancy_decoder.query(
                    positions_chunk
                )[0, :actual_chunk_size]
                if embedded_positions is None:
                    total_size = positions.shape[0]
                    embedded_positions = torch.empty(
                        (total_size, *embedded_positions_chunk.shape[1:]),
                        dtype=embedded_positions_chunk.dtype,
                        device=embedded_positions_chunk.device,
                    )
                embedded_positions[start : start + chunk_size] = (
                    embedded_positions_chunk
                )
            return embedded_positions

        # Do some cacheable setup work. If we do not precompute embeddings then
        # only the query positions are precomputed here.
        has_valid_cache = hasattr(self, "implicit_field_coarse_to_fine_evaluator")
        if has_valid_cache:
            # Check if parameters have changed. If yes, then we have to re-run the setup code.
            has_valid_cache = (
                fine_grid_resolution
                == self.implicit_field_coarse_to_fine_evaluator.fine_grid_resolution
                and (
                    bbox_min == self.implicit_field_coarse_to_fine_evaluator.bbox_min
                ).all()
                and (
                    bbox_max == self.implicit_field_coarse_to_fine_evaluator.bbox_max
                ).all()
                and self.coarse_to_fine_precompute_embeddings == precompute_embeddings
                and self.coarse_to_fine_chunk_size == chunk_size
            )
        if not has_valid_cache:
            self.implicit_field_coarse_to_fine_evaluator = (
                ImplicitFieldCoarseToFineEvaluator(
                    bbox_min, bbox_max, fine_grid_resolution, embed_positions
                )
            )
            self.coarse_to_fine_precompute_embeddings = precompute_embeddings
            self.coarse_to_fine_chunk_size = chunk_size

        # Evaluates the coarse grid. For batch size > 1, we exploit that all
        # functions are evaluated at the same positions.
        def eval_func_coarse(positions):
            occupancy_logits = torch.empty(
                (batch_size, positions.shape[0]),
                device=positions.device,
                dtype=torch.float32,
            )
            for start in range(0, positions.shape[0], chunk_size):
                # Slice out chunk
                positions_chunk = positions[start : start + chunk_size, :]

                # pad last chunk of queries to the same length to avoid torch.compile re-compiling this block
                actual_chunk_size = positions_chunk.shape[0]
                if start > 0 and actual_chunk_size < chunk_size:
                    positions_chunk = F.pad(
                        positions_chunk, [0, 0, 0, chunk_size - actual_chunk_size]
                    )

                # Repeat positions along batch dimension, each field is queried at the same positions.
                positions_chunk = positions_chunk.unsqueeze(0)
                positions_chunk = positions_chunk.expand(
                    batch_size, *positions_chunk.shape[1:]
                )

                # Query network
                occupancy_logits_chunk = self.query(
                    positions_chunk, latents, skip_query_transform=precompute_embeddings
                )
                occupancy_logits[:, start : start + chunk_size] = (
                    occupancy_logits_chunk[:, :actual_chunk_size].float()
                )
            return occupancy_logits

        # Evaluates the fine grid. Unlike in the function above where all fields are evaluated
        # at the same position, here each field is evaluated at different positions. Consequently,
        # for batch size > 1, this function will be called in a for loop as each occupancy field
        # at a variable amount of positions. This does not come at a performance loss since we are
        # anyhow looping over the chunk dimension.
        def eval_func_fine(positions, batch_idx):
            occupancy_logits = torch.empty(
                positions.shape[0], device=positions.device, dtype=torch.float32
            )
            for start in range(0, positions.shape[0], chunk_size):
                positions_chunk = positions[start : start + chunk_size, :]

                # pad last chunk of queries to the same length to avoid torch.compile re-compiling this block
                actual_chunk_size = positions_chunk.shape[0]
                if start > 0 and actual_chunk_size < chunk_size:
                    positions_chunk = F.pad(
                        positions_chunk, [0, 0, 0, chunk_size - actual_chunk_size]
                    )
                positions_chunk = positions_chunk.unsqueeze(0)
                latent = latents[batch_idx].unsqueeze(0)
                occupancy_logits_chunk = self.query(
                    positions_chunk, latent, skip_query_transform=precompute_embeddings
                )[0, :actual_chunk_size]
                occupancy_logits[start : start + chunk_size] = (
                    occupancy_logits_chunk.detach().float()
                )
            return occupancy_logits

        tau = 0.25
        if hasattr(self.cfg, "logits_scale"):
            tau = tau / self.cfg.logits_scale

        implicit_grids = self.implicit_field_coarse_to_fine_evaluator.evaluate(
            eval_func_coarse, eval_func_fine, tau=tau
        )

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for batch_idx in range(batch_size):
            try:
                warp_success = False
                if use_warp and getattr(self, "training", False) == False:
                    # make sure we disable warp when training
                    # since warp can cause memory illegal access and also crash the kernel
                    # causing the following cuda operations to fail
                    # it's ok to run this at inference time, since the following operations are not cuda
                    try:
                        vertices, faces = marching_cubes_with_warp(
                            implicit_grids[batch_idx],
                            level=0.0,
                            device=implicit_grids.device,
                        )
                        warp_success = True
                    except Exception as e:
                        logging.warning(
                            f"Warning: error in marching cubes with warp: {e}"
                        )
                        warp_success = False  # Fall back to CPU version

                if not warp_success:
                    vertices, faces, _, _ = measure.marching_cubes(
                        implicit_grids[batch_idx].cpu().numpy(), 0, method="lewiner"
                    )

                bbox_min_np = np.array(bounds[0:3])
                bbox_max_np = np.array(bounds[3:6])
                bbox_size_np = bbox_max_np - bbox_min_np
                vertices = vertices / fine_grid_resolution * bbox_size_np + bbox_min_np
                faces = faces[:, [2, 1, 0]]
                mesh_v_f.append(
                    (vertices.astype(np.float32), np.ascontiguousarray(faces))
                )
                has_surface[batch_idx] = True
            except Exception as e:
                logging.error(f"Error: error in extract_geometry: {e}")
                mesh_v_f.append((None, None))
                has_surface[batch_idx] = False

        return mesh_v_f, has_surface


class DiagonalGaussianDistribution:
    """Diagonal-covariance Gaussian over latent activations.

    Only the inference-time ``mode()`` (i.e. the posterior mean) is used by
    the released pipeline.
    """

    def __init__(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        deterministic=False,
        feat_dim=1,
    ):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
