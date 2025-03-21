import torch
import torch.nn.functional as F


def top_p_filtering(logits, top_p: float = 1.0):
    """
    Filter a distribution of logits using top-p filtering.
    The input logits tensor is modified in-place.

    Args:
        logits (torch.Tensor): A tensor of logits to be filtered. Expected shape is [..., vocab_size].
        top_p (float, optional): The cumulative probability threshold for top-p sampling.
               If < 1.0, only keep the smallest set of tokens whose
               cumulative probability does not exceed this threshold.

    Returns:
        torch.Tensor: logits where values outside the top-p threshold are set to -∞.
    """
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        sorted_idx_to_remove[..., 0] = False

        idx_to_remove = sorted_idx_to_remove.scatter(
            -1, sorted_idx, sorted_idx_to_remove
        )
        logits.masked_fill_(idx_to_remove, -torch.inf)

    return logits


def process_logits(
        logits,
        top_p: float = None,
    ):
    """
    Process logits by optionally applying nucleus (top-p) filtering and token selection.

    If `top_p` is None, the token with the highest probability (argmax) is selected.
    If `top_p` is provided, smallest set of tokens with cumulative probability ≥ top_p are kept, then softmax is applied to obtain
    probabilities. A token is sampled from this filtered distribution using `torch.multinomial`.

    Args:
        logits (torch.Tensor): A tensor of logits to process.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling.
            If None, argmax selection is performed (deterministic generation). Otherwise, smallest set of tokens with cumulative probability ≥ top_p are kept (stochastic generation).

    Returns:
        torch.Tensor: selected token index.
    """
    if top_p is None:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = top_p_filtering(logits, top_p=0.9)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, replacement=True)
    return next_id
