import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from torch.nn import functional as F
from tqdm import tqdm

from cube3d.model.autoencoder.embedder import PhaseModulatedFourierEmbedder
from cube3d.model.autoencoder.grid import (
    generate_dense_grid_points,
    marching_cubes_with_warp,
)
from cube3d.model.autoencoder.spherical_vq import SphericalVectorQuantizer
from cube3d.model.transformers.attention import (
    EncoderCrossAttentionLayer,
    EncoderLayer,
    init_linear,
    init_tfixup,
)
from cube3d.model.transformers.norm import LayerNorm


def init_sort(x):
    """
    Sorts the input tensor `x` based on its pairwise distances to the first element.
    This function computes the pairwise distances between all elements in `x` and the
    first element of `x`. It then sorts the elements of `x` in ascending order of
    their distances to the first element.
    Args:
        x (torch.Tensor): A 2D tensor where each row represents a data point.
    Returns:
        torch.Tensor: A tensor containing the rows of `x` sorted by their distances
        to the first row of `x`.
    """

    distances = torch.cdist(x, x[:1])
    _, indices = torch.sort(distances.squeeze(), dim=0)
    x = x[indices]
    return x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, embed_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.apply(partial(init_linear, embed_dim=embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class OneDEncoder(nn.Module):
    def __init__(
        self,
        embedder,
        num_latents: int,
        point_feats: int,
        embed_point_feats: bool,
        width: int,
        num_heads: int,
        num_layers: int,
        with_cls_token: bool = False,
        cross_attention_levels: Optional[List[int]] = None,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the OneDEncoder model.
        Args:
            embedder: An embedding module that provides the input embedding functionality.
            num_latents (int): The number of latent variables.
            point_feats (int): The number of point features.
            embed_point_feats (bool): Whether to embed point features or not.
            width (int): The width of the embedding dimension.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of encoder layers.
            with_cls_token (bool, optional): Whether to include a classification token like in Vision Transformers (ViT). Defaults to False.
            cross_attention_levels (Optional[List[int]], optional): The indices of layers where cross-attention is applied. Defaults to None.
            eps (float, optional): A small value added for numerical stability in normalization layers. Defaults to 1e-6.
        Returns:
            None
        """
        super().__init__()

        self.embedder = embedder

        # add cls token like ViT
        self.with_cls_token = with_cls_token
        if self.with_cls_token:
            query = torch.empty((1 + num_latents, width))
        else:
            query = torch.empty((num_latents, width))

        # initialize then sort query to potentially get better ordering
        query.uniform_(-1.0, 1.0)
        query = init_sort(query)

        # set parameter
        self.query = nn.Parameter(query)

        self.embed_point_feats = embed_point_feats
        in_dim = (
            self.embedder.out_dim * 2
            if self.embed_point_feats
            else self.embedder.out_dim + point_feats
        )
        self.feat_in = MLPEmbedder(in_dim, embed_dim=width)

        if cross_attention_levels is None:
            cross_attention_levels = [0]

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i in cross_attention_levels:
                self.blocks.append(
                    EncoderCrossAttentionLayer(
                        embed_dim=width,
                        num_heads=num_heads,
                        eps=eps,
                    )
                )
            else:
                self.blocks.append(
                    EncoderLayer(embed_dim=width, num_heads=num_heads, eps=eps)
                )
        self.ln_f = LayerNorm(width, eps=eps)

        init_tfixup(self, num_layers)

    def _forward(self, h, data, attn_mask=None):
        """
        Forward pass for the autoencoder model.

        Args:
            h (torch.Tensor): The input tensor to be processed, typically representing
                the hidden state or intermediate representation.
            data (torch.Tensor): The input data tensor to be transformed by the feature
                extraction layer and used in cross-attention layers.
            attn_mask (torch.Tensor, optional): An optional attention mask tensor to be
                used in attention layers for masking specific positions. Defaults to None.
        Returns:
            torch.Tensor: The output tensor after processing through the layers and
            applying final normalization.
        """

        data = self.feat_in(data)

        for block in self.blocks:
            if isinstance(block, EncoderCrossAttentionLayer):
                h = block(h, data)
            else:
                h = block(h, attn_mask=attn_mask)

        h = self.ln_f(h)
        return h

    def forward(
        self, pts: torch.Tensor, feats: torch.Tensor
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the 1D autoencoder model.
        Args:
            pts (torch.Tensor): Input tensor representing points with shape (batch_size, num_points, point_dim).
            feats (torch.Tensor): Input tensor representing features with shape (batch_size, num_points, feature_dim).
                                  Can be None if no features are provided.
        Returns:
            Tuple[torch.Tensor, list[torch.Tensor]]:
                - The output tensor after processing the input data.
                - A list of intermediate tensors (if applicable) generated during the forward pass.
        """

        b = pts.shape[0]
        data = self.embedder(pts)

        if feats is not None:
            if self.embed_point_feats:
                feats = self.embedder(feats)
            data = torch.cat([data, feats], dim=-1)

        # prepare query and data
        h = self.query.unsqueeze(0).expand(b, -1, -1)
        return self._forward(h, data, attn_mask=None)


class OneDBottleNeck(nn.Module):
    def __init__(
        self,
        block,
    ) -> None:
        """
        Initializes the OneDBottleNeck class.
        Args:
            block: The building block or module used within the autoencoder.
        """
        super().__init__()

        self.block = block

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of the OneDBottleNeck function.
        Args:
            h (torch.Tensor): Input tensor to the model.
        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing:
                - The transformed tensor `z` after passing through the block (if applicable).
                - A dictionary `ret_dict` containing additional information:
                    - "indices": Indices from the block output (if present).
                    - "z_q": Quantized tensor from the block output (if present).

        """

        z = h
        ret_dict = {}
        if self.block is not None:
            z, d = self.block(z)

            key_mappings = {
                "q": "indices",
                "z_q": "z_q",
            }
            for in_key, out_key in key_mappings.items():
                if in_key in d:
                    ret_dict[out_key] = d[in_key]

        return z, ret_dict


class OneDDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        width: int,
        num_heads: int,
        num_layers: int,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the OneDDecoder class.
        Args:
            num_latents (int): The number of latent variables.
            width (int): The width of the embedding dimension.
            num_heads (int): The number of attention heads in each encoder layer.
            num_layers (int): The number of encoder layers.
            eps (float, optional): A small value added for numerical stability. Defaults to 1e-6.
        """
        super().__init__()

        self.register_buffer("query", torch.empty([0, width]), persistent=False)
        self.positional_encodings = nn.Parameter(
            init_sort(F.normalize(torch.empty(num_latents, width).normal_()))
        )
        self.blocks = nn.ModuleList(
            [
                EncoderLayer(embed_dim=width, num_heads=num_heads, eps=eps)
                for _ in range(num_layers)
            ]
        )

        init_tfixup(self, num_layers)

    def _forward(self, h):
        """
        Applies a sequence of operations to the input tensor `h` using the blocks
        defined in the model.
        Args:
            h (torch.Tensor): The input tensor to be processed by the blocks.
        Returns:
            torch.Tensor: The output tensor after applying all blocks sequentially.
        """

        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, z):
        """
        This method processes the input tensor `z` by padding it to a fixed length,
        adding positional encodings, and then passing it through the `_forward` method.

        Args:
            z (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after processing through the autoencoder.
        Notes:
            - If the `query` attribute has a non-zero shape, the input tensor `z` is padded
              to match the required length using slices of `query`.
            - Positional encodings are added to the padded input tensor before passing it
              to the `_forward` method.
        """

        # pad input to fixed length
        if self.query.shape[0] > 0:
            pad_len = self.query.shape[0] + 1 - z.shape[1]
            paddings = self.query[:pad_len, ...].unsqueeze(0).expand(z.shape[0], -1, -1)
            z = torch.cat([paddings, z], dim=1)
        h = z + self.positional_encodings[: z.shape[1], :].unsqueeze(0).expand(
            z.shape[0], -1, -1
        )

        return self._forward(h)


class OneDOccupancyDecoder(nn.Module):
    def __init__(
        self, embedder, out_features: int, width: int, num_heads: int, eps=1e-6
    ) -> None:
        """
        Initializes the OneDOccupancyDecoder module.
        Args:
            embedder: An embedding module that provides input embeddings.
            out_features (int): The number of output features for the final linear layer.
            width (int): The width of the intermediate layers.
            num_heads (int): The number of attention heads for the cross-attention layer.
            eps (float, optional): A small value added for numerical stability in layer normalization. Defaults to 1e-6.
        """
        super().__init__()

        self.embedder = embedder
        self.query_in = MLPEmbedder(self.embedder.out_dim, width)

        self.attn_out = EncoderCrossAttentionLayer(embed_dim=width, num_heads=num_heads)
        self.ln_f = LayerNorm(width, eps=eps)
        self.c_head = nn.Linear(width, out_features)

    def query(self, queries: torch.Tensor):
        """
        Processes the input tensor through the embedder and query_in layers.
        Args:
            queries (torch.Tensor): A tensor containing the input data to be processed.
        Returns:
            torch.Tensor: The output tensor after being processed by the embedder and query_in layers.
        """

        return self.query_in(self.embedder(queries))

    def forward(self, queries: torch.Tensor, latents: torch.Tensor):
        """
        Defines the forward pass of the model.
        Args:
            queries (torch.Tensor): Input tensor representing the queries.
            latents (torch.Tensor): Input tensor representing the latent representations.
        Returns:
            torch.Tensor: Output tensor after applying the query transformation,
                          attention mechanism, and final processing layers.
        """
        queries = self.query(queries)
        x = self.attn_out(queries, latents)
        x = self.c_head(self.ln_f(x))
        return x


class OneDAutoEncoder(nn.Module):
    @dataclass
    class Config:
        checkpoint_path: str = ""

        # network params
        num_encoder_latents: int = 256
        num_decoder_latents: int = 256
        embed_dim: int = 12
        width: int = 768
        num_heads: int = 12
        out_dim: int = 1
        eps: float = 1e-6

        # grid features embedding
        num_freqs: int = 128
        point_feats: int = 0
        embed_point_feats: bool = False

        num_encoder_layers: int = 1
        encoder_cross_attention_levels: list[int] = field(default_factory=list)
        num_decoder_layers: int = 23

        encoder_with_cls_token: bool = True
        num_codes: int = 16384

    def __init__(self, cfg: Config) -> None:
        """
        Initializes the OneDAutoencoder model.
        Args:
            cfg (Config): Configuration object containing the parameters for the model.
        Attributes:
            cfg (Config): Stores the configuration object.
            embedder (PhaseModulatedFourierEmbedder): Embeds input data using phase-modulated Fourier features.
            encoder (OneDEncoder): Encodes the input data into latent representations.
            bottleneck (OneDBottleNeck): Bottleneck layer containing a spherical vector quantizer for dimensionality reduction.
            decoder (OneDDecoder): Decodes latent representations back into the original data space.
            occupancy_decoder (OneDOccupancyDecoder): Decodes occupancy information from latent representations.
        """

        super().__init__()

        self.cfg = cfg

        self.embedder = PhaseModulatedFourierEmbedder(
            num_freqs=self.cfg.num_freqs, input_dim=3
        )

        self.encoder = OneDEncoder(
            embedder=self.embedder,
            num_latents=self.cfg.num_encoder_latents,
            with_cls_token=self.cfg.encoder_with_cls_token,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_encoder_layers,
            cross_attention_levels=self.cfg.encoder_cross_attention_levels,
            eps=self.cfg.eps,
        )

        block = SphericalVectorQuantizer(
            self.cfg.embed_dim,
            self.cfg.num_codes,
            self.cfg.width,
            codebook_regularization="kl",
        )
        self.bottleneck = OneDBottleNeck(block=block)

        self.decoder = OneDDecoder(
            num_latents=self.cfg.num_encoder_latents,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_decoder_layers,
            eps=self.cfg.eps,
        )

        self.occupancy_decoder = OneDOccupancyDecoder(
            embedder=self.embedder,
            out_features=self.cfg.out_dim,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            eps=self.cfg.eps,
        )

    @torch.no_grad()
    def decode_indices(self, shape_ids: torch.Tensor):
        """
        Decodes the given shape indices into latent representations.
        Args:
            shape_ids (torch.Tensor): A tensor containing the shape indices to be decoded.
        Returns:
            torch.Tensor: The decoded latent representations corresponding to the input shape indices.
        """

        z_q = self.bottleneck.block.lookup_codebook(shape_ids)
        latents = self.decode(z_q)
        return latents

    @torch.no_grad()
    def query_embeds(self, shape_ids: torch.Tensor):
        """
        Retrieves the latent embeddings corresponding to the given shape IDs.
        Args:
            shape_ids (torch.Tensor): A tensor containing the IDs of the shapes
                for which the latent embeddings are to be queried.
        Returns:
            torch.Tensor: A tensor containing the latent embeddings retrieved
                from the codebook for the provided shape IDs.
        """

        z_q = self.bottleneck.block.lookup_codebook_latents(shape_ids)
        return z_q

    @torch.no_grad()
    def query_indices(self, shape_embs: torch.Tensor):
        """
        Queries the indices of the quantized embeddings from the bottleneck layer.
        Args:
            shape_embs (torch.Tensor): The input tensor containing shape embeddings
                to be quantized.
        Returns:
            torch.Tensor: A tensor containing the quantized indices.
        """

        _, ret_dict = self.bottleneck.block.quantize(shape_embs)
        return ret_dict["q"]

    def encode(self, x: torch.Tensor, **kwargs):
        """
        Encodes the input tensor using the encoder and bottleneck layers.
        Args:
            x (torch.Tensor): Input tensor with shape (..., N), where the first 3
                dimensions represent points (pts) and the remaining dimensions
                represent features (feats).
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, None, dict]: A tuple containing:
                - z_e (torch.Tensor): Encoded tensor before bottleneck processing.
                - z (torch.Tensor): Encoded tensor after bottleneck processing.
                - None: Placeholder for compatibility with other methods.
                - d (dict): Dictionary containing additional information, including:
                    - "z_cls" (torch.Tensor, optional): Class token if
                      `self.cfg.encoder_with_cls_token` is True.
        """

        pts, feats = x[..., :3], x[..., 3:]
        z_e = self.encoder(pts, feats)

        # split class token
        if self.cfg.encoder_with_cls_token:
            z_cls = z_e[:, 0, ...]
            z_e = z_e[:, 1:, ...]

        # quantize or kl
        z, d = self.bottleneck(z_e)

        if self.cfg.encoder_with_cls_token:
            d["z_cls"] = z_cls
        return z_e, z, None, d

    def decode(self, z: torch.Tensor):
        """
        Decodes the latent representation `z` using the decoder network.
        Args:
            z (torch.Tensor): The latent representation tensor to be decoded.
        Returns:
            torch.Tensor: The decoded output tensor.
        """

        h = self.decoder(z)
        return h

    def query(self, queries: torch.Tensor, latents: torch.Tensor):
        """
        Computes the logits by decoding the given queries and latent representations.
        Args:
            queries (torch.Tensor): A tensor containing the query points to be decoded.
            latents (torch.Tensor): A tensor containing the latent representations corresponding to the queries.
        Returns:
            torch.Tensor: A tensor containing the decoded logits for the given queries and latents.
        """

        logits = self.occupancy_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(self, surface, queries, **kwargs):
        """
        Perform a forward pass through the autoencoder model.
        Args:
            surface (torch.Tensor): The input surface tensor to be encoded.
            queries (torch.Tensor): The query tensor used for generating logits.
            **kwargs: Additional keyword arguments.
        Returns:
            tuple: A tuple containing:
                - z (torch.Tensor): The latent representation of the input surface.
                - latents (torch.Tensor): The decoded output from the latent representation.
                - None: Placeholder for a potential future return value.
                - logits (torch.Tensor): The logits generated from the queries and latents.
                - d (torch.Tensor): Additional output from the encoding process.
        """

        _, z, _, d = self.encode(surface)

        latents = self.decode(z)
        logits = self.query(queries, latents)

        return z, latents, None, logits, d

    @torch.no_grad()
    def extract_geometry(
        self,
        latents: torch.FloatTensor,
        bounds: list[float] = [
            -1.05,
            -1.05,
            -1.05,
            1.05,
            1.05,
            1.05,
        ],
        resolution_base: float = 9.0,
        chunk_size: int = 2_000_000,
        use_warp: bool = False,
    ):
        """
        Extracts 3D geometry from latent representations using a dense grid sampling
        and marching cubes algorithm.
        Args:
            latents (torch.FloatTensor): A tensor of latent representations with shape
                (batch_size, latent_dim).
            bounds (list[float], optional): A list of six floats defining the bounding box
                for the 3D grid in the format [xmin, ymin, zmin, xmax, ymax, zmax].
                Defaults to [-1.05, -1.05, -1.05, 1.05, 1.05, 1.05].
            resolution_base (float, optional): The base resolution for the grid. Higher
                values result in finer grids. Defaults to 9.0.
            chunk_size (int, optional): The number of grid points to process in a single
                chunk. Defaults to 2,000,000.
            use_warp (bool, optional): Whether to use a GPU-accelerated marching cubes
                implementation. If False, falls back to a CPU implementation. Defaults to False.
        Returns:
            tuple:
                - mesh_v_f (list[tuple]): A list of tuples containing vertices and faces
                  for each batch element. Each tuple is of the form
                  (vertices, faces), where:
                    - vertices (np.ndarray): Array of vertex coordinates with shape
                      (num_vertices, 3).
                    - faces (np.ndarray): Array of face indices with shape
                      (num_faces, 3).
                  If geometry extraction fails for a batch element, the tuple will be
                  (None, None).
                - has_surface (np.ndarray): A boolean array indicating whether a surface
                  was successfully extracted for each batch element.
        Raises:
            Exception: Logs warnings or errors if geometry extraction fails for any
            batch element or if the marching cubes algorithm encounters issues.
        """
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
            desc=f"extracting geometry",
            unit="chunk",
        )
        for start in progress_bar:
            queries = xyz_samples[start : start + chunk_size, :]

            num_queries = queries.shape[0]
            if start > 0 and num_queries < chunk_size:
                queries = F.pad(queries, [0, 0, 0, chunk_size - num_queries])
            batch_queries = queries.unsqueeze(0).expand(batch_size, -1, -1).to(latents)

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
                if use_warp:
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
                    logging.warning(
                        "Warning: falling back to CPU version of marching cubes using skimage measure"
                    )
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
