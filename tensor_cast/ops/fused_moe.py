from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("moe_gating_topk")
def _(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gating + top-k routing for MoE layers.

    Corresponds to NPU's npu_moe_gating_top_k kernel (aclnnMoeGatingTopK).
    Meta-only: output shapes/dtypes valid, values undefined.

    Args:
        logits: (num_tokens, num_experts), raw router logits
        expert_bias: (num_experts,), per-expert bias
        top_k: number of experts per token

    Returns:
        (topk_weights, topk_indices):
            topk_weights: (num_tokens, top_k), float
            topk_indices: (num_tokens, top_k), int32
    """
    num_tokens = logits.shape[0]
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=logits.dtype, device=logits.device
    )
    topk_indices = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=logits.device
    )
    return topk_weights, topk_indices


@register_tensor_cast_op("permute_tokens")
def _(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Repeat the input tokens top-k times, and rearrange them according to the order of the experts
    selected by the tokens.

    Args:
        x: (bsz, seq_len, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        permuted_x: (bsz * seq_len * top_k, hidden_size)
    """
    num_tokens = topk_indices.numel()
    return torch.empty((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("unpermute_tokens")
def _(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange the input tokens (initially sorted by their selected experts) by token indices.

    Args:
        x: (bsz * seq_len * top_k, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        unpermuted_x: (bsz, seq_len, top_k, hidden_size)
    """
    return torch.empty_like(x).view(*topk_indices.shape, x.shape[-1])


@register_tensor_cast_op("moe_gating_top_k_softmax")
def _(x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused operation for Mixture of Experts (MoE) gating, combining softmax and top-k selection.

    This function is designed to handle both the softmax operation over the gating logits
    and the top-k selection of experts in one step. It returns the shape of the expected output
    tensors (experts_indices, and experts_weights) without performing any computation.

    Args:
        x (torch.Tensor): A tensor of containing the raw unnormalized logits for each experts.
                          These logits will be used to compute the softmax probabilities and
                          select the top-k experts.
        top_k (int): The number of top experts to select based on their softmax probabilities.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - topk_weights (torch.Tensor): Corresponding normalized weights (e.g., after softmax),
              with the same shape, dtype and device as input `x`.
            - topk_indices (torch.Tensor): Indices of the selected experts,
              with the same shape and device as input `x`, dtype int64.
    """
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.int64)
