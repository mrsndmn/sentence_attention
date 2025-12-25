import torch

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow as optimized_closest_idx,
)


def slow_reference_impl_flexible_runs(special_token_mask: torch.Tensor) -> torch.Tensor:
    """
    Flexible gist-run mode reference:
    - Special (gist) tokens form contiguous runs.
    - A "boundary" is the last token of each run.
    - Output is the index of the last completed boundary; for the boundary token itself,
      the output remains the previous boundary (matches the model's shift-by-1 semantics).
    """
    special_token_mask_bool = special_token_mask.bool().cpu()
    clothest_token_idx = torch.zeros_like(special_token_mask, dtype=torch.long, device="cpu")

    for batch_i in range(special_token_mask_bool.shape[0]):
        current_boundary = 0
        for pos in range(special_token_mask_bool.shape[1]):
            is_special = bool(special_token_mask_bool[batch_i, pos].item())
            next_is_special = False
            if pos + 1 < special_token_mask_bool.shape[1]:
                next_is_special = bool(special_token_mask_bool[batch_i, pos + 1].item())

            if is_special and (not next_is_special):
                # End of a special run: emit previous boundary, then advance boundary for future tokens.
                clothest_token_idx[batch_i, pos] = current_boundary
                current_boundary = pos
            else:
                clothest_token_idx[batch_i, pos] = current_boundary

    return clothest_token_idx.to(special_token_mask.device)


def test_flexible_runs_mixed_lengths():
    # Mixed run lengths within a single sequence (runs are separated by non-special tokens).
    # Runs: len=2 at [1,2], len=1 at [5], len=3 at [8,9,10]
    special_mask = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]], dtype=torch.long)

    ref = slow_reference_impl_flexible_runs(special_mask)
    got = optimized_closest_idx(special_mask)

    assert ref.shape == got.shape
    assert ref.dtype == got.dtype == torch.long
    assert torch.equal(ref, got)
