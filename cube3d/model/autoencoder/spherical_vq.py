import sys
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube3d.model.transformers.norm import RMSNorm


class SphericalVectorQuantizer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_codes: int,
        width: Optional[int] = None,
        codebook_regularization: Literal["batch_norm", "kl"] = "batch_norm",
    ):
        """
        Initializes the SphericalVQ module.
        Args:
            embed_dim (int): The dimensionality of the embeddings.
            num_codes (int): The number of codes in the codebook.
            width (Optional[int], optional): The width of the input. Defaults to None.
        Raises:
            ValueError: If beta is not in the range [0, 1].
        """
        super().__init__()

        self.num_codes = num_codes

        self.codebook = nn.Embedding(num_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        width = width or embed_dim
        if width != embed_dim:
            self.c_in = nn.Linear(width, embed_dim)
            self.c_x = nn.Linear(width, embed_dim)  # shortcut
            self.c_out = nn.Linear(embed_dim, width)
        else:
            self.c_in = self.c_out = self.c_x = nn.Identity()

        self.norm = RMSNorm(embed_dim, elementwise_affine=False)
        self.cb_reg = codebook_regularization
        if self.cb_reg == "batch_norm":
            self.cb_norm = nn.BatchNorm1d(embed_dim, track_running_stats=False)
        else:
            self.cb_weight = nn.Parameter(torch.ones([embed_dim]))
            self.cb_bias = nn.Parameter(torch.zeros([embed_dim]))
            self.cb_norm = lambda x: x.mul(self.cb_weight).add_(self.cb_bias)

    def get_codebook(self):
        """
        Retrieves the normalized codebook weights.
        This method applies a series of normalization operations to the
        codebook weights, ensuring they are properly scaled and normalized
        before being returned.
        Returns:
            torch.Tensor: The normalized weights of the codebook.
        """

        return self.norm(self.cb_norm(self.codebook.weight))

    @torch.no_grad()

    def lookup_codebook(self, q: torch.Tensor):
        """
        Perform a lookup in the codebook and process the result.
        This method takes an input tensor of indices, retrieves the corresponding
        embeddings from the codebook, and applies a transformation to the retrieved
        embeddings.
        Args:
            q (torch.Tensor): A tensor containing indices to look up in the codebook.
        Returns:
            torch.Tensor: The transformed embeddings retrieved from the codebook.
        """

        # normalize codebook
        z_q = F.embedding(q, self.get_codebook())
        z_q = self.c_out(z_q)
        return z_q

    @torch.no_grad()
    def lookup_codebook_latents(self, q: torch.Tensor):
        """
        Retrieves the latent representations from the codebook corresponding to the given indices.
        Args:
            q (torch.Tensor): A tensor containing the indices of the codebook entries to retrieve.
                              The indices should be integers and correspond to the rows in the codebook.
        Returns:
            torch.Tensor: A tensor containing the latent representations retrieved from the codebook.
                          The shape of the returned tensor depends on the shape of the input indices
                          and the dimensionality of the codebook entries.
        """

        # normalize codebook
        z_q = F.embedding(q, self.get_codebook())
        return z_q

    def quantize(self, z: torch.Tensor):
        """
        Quantizes the latent codes z with the codebook

        Args:
                z (Tensor): B x ... x F
        """

        # normalize codebook
        codebook = self.get_codebook()
        # the process of finding quantized codes is non differentiable
        with torch.no_grad():
            # flatten z
            z_flat = z.view(-1, z.shape[-1])

            # calculate distance and find the closest code
            d = torch.cdist(z_flat, codebook)
            q = torch.argmin(d, dim=1)  # num_ele

        z_q = codebook[q, :].reshape(*z.shape[:-1], -1)
        q = q.view(*z.shape[:-1])

        return z_q, {"z": z.detach(), "q": q}

    def straight_through_approximation(self, z, z_q):
        """passed gradient from z_q to z"""
        z_q = z + (z_q - z).detach()
        return z_q

    def forward(self, z: torch.Tensor):
        """
        Forward pass of the spherical vector quantization autoencoder.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, ..., feature_dim).
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - z_q (torch.Tensor): The quantized output tensor after applying the
                  straight-through approximation and output projection.
                - ret_dict (Dict[str, Any]): A dictionary containing additional
                  information:
                    - "z_q" (torch.Tensor): Detached quantized tensor.
                    - "q" (torch.Tensor): Indices of the quantized vectors.
                    - "perplexity" (torch.Tensor): The perplexity of the quantization,
                      calculated as the exponential of the negative sum of the
                      probabilities' log values.
        """

        with torch.autocast(device_type=z.device.type, enabled=False):
            # work in full precision
            z = z.float()

            # project and normalize
            z_e = self.norm(self.c_in(z))
            z_q, ret_dict = self.quantize(z_e)

            ret_dict["z_q"] = z_q.detach()
            z_q = self.straight_through_approximation(z_e, z_q)
            z_q = self.c_out(z_q)

        return z_q, ret_dict
