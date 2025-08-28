import torch
import pytest
import math


def patch_token_balancing_fused_moe(curr_topk_ids, E):
    # Patch topk ids equally to experts
    if curr_topk_ids.numel() > 0:
        num_tokens, k = curr_topk_ids.shape
        expert_ids_uniform = torch.arange(
            E, device=curr_topk_ids.device
        ).repeat((num_tokens * k + E - 1) // E)[: num_tokens * k]
        expert_ids_uniform = expert_ids_uniform.view(num_tokens, k)
        curr_topk_ids = expert_ids_uniform
    return curr_topk_ids


@pytest.mark.parametrize(
    "num_tokens,k,E",
    [
        (10, 1, 3),
        (12, 2, 4),
        (7, 3, 2),
        (8, 2, 5),
    ],
)
def test_patch_token_balancing_even(num_tokens, k, E):
    curr_topk_ids = torch.zeros((num_tokens, k), dtype=torch.long)
    balanced_ids = patch_token_balancing_fused_moe(curr_topk_ids, E)
    flat = balanced_ids.view(-1)
    counts = torch.bincount(flat, minlength=E)

    assert max(counts) - min(counts) <= 1
    assert torch.all(flat < E)
    assert torch.all(flat >= 0)


def balance_expert_distribution(topk_idx, topk_weights, num_experts, top_k):
    num_tokens, k = topk_idx.shape
    E = num_experts
    expert_ids_uniform = torch.arange(E, device=topk_idx.device).repeat(
        (num_tokens * k + E - 1) // E
    )[: num_tokens * k]
    expert_ids_uniform = expert_ids_uniform.view(num_tokens, k)
    topk_idx = expert_ids_uniform
    topk_weights = torch.full_like(topk_weights, 1.0 / k)  # Uniform weights

    return topk_idx, topk_weights


@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k",
    [
        (10, 3, 1),
        (7, 2, 2),
        (12, 4, 1),
        (8, 5, 2),
    ],
)
def test_balance_expert_distribution(num_tokens, num_experts, top_k):
    topk_idx = torch.zeros((num_tokens, top_k), dtype=torch.long)
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32)
    balanced_idx, balanced_weights = balance_expert_distribution(
        topk_idx, topk_weights, num_experts, top_k
    )
    flat = balanced_idx[balanced_idx != -1].view(-1)
    counts = torch.bincount(flat, minlength=num_experts)

    assert counts.max() - counts.min() <= 1
    assert torch.allclose(
        balanced_weights[balanced_idx != -1],
        torch.ones_like(balanced_weights[balanced_idx != -1]) / top_k,
    )
    if (balanced_idx == -1).any():
        assert torch.all(balanced_weights[balanced_idx == -1] == 0.0)
