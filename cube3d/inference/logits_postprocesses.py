import torch
import torch.nn.functional as F


def top_k_filtering(logits, top_k: int = 1):
    """
    Filter a distribution of logits using top-k and/or top-p (nucleus) filtering.
    The input logits tensor is modified in-place.

    Args:
        logits: A tensor of logits to be filtered. Expected shape is [..., vocab_size].
        top_k: If > 0, only keep the top k tokens with highest probability.

    Returns:
        A tensor of logits where values outside the top-k/top-p threshold are set to -∞.
    """
    if top_k > 0:
        idx_to_remove = logits < logits.topk(top_k, largest=True, sorted=False, dim=-1)[
            0
        ].amin(dim=-1, keepdim=True)
        logits.masked_fill_(idx_to_remove, -torch.inf)

    return logits


def process_logits(
        logits,
        top_k: int = 1,
    ):
    """
    Process logits by optionally applying top-k filtering.
    The final probabilities are returned after applying softmax on the filtered logits.

    Args:
        logits: A tensor of logits to process. Expected shape is [..., vocab_size].
        top_k: If > 0, only keep the top k tokens with highest probability.

    Returns:
        A tensor of probabilities after filtering, with the same shape as the input logits.
    """
    logits = top_k_filtering(logits, top_k=top_k)
    probs = F.softmax(logits, dim=-1)
    return probs
