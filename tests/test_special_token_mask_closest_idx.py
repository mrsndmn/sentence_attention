import pytest
import torch

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow as optimized_closest_idx,
)


def slow_reference_impl(special_token_mask: torch.Tensor, num_special_tokens: int) -> torch.Tensor:
    assert num_special_tokens is not None, "num_special_tokens must be provided"
    special_token_mask_bool = special_token_mask.bool()
    special_token_mask_bool = special_token_mask_bool.cpu()

    clothest_token_idx = torch.zeros_like(special_token_mask, dtype=torch.long, device="cpu")

    for batch_i in range(special_token_mask_bool.shape[0]):
        current_clothest_token_idx = 0
        current_sequrntial_num_special_tokens = 0
        for seq_len_i in range(special_token_mask_bool.shape[1]):
            if special_token_mask_bool[batch_i, seq_len_i].item():
                current_sequrntial_num_special_tokens += 1

            if (
                special_token_mask_bool[batch_i, seq_len_i].item()
                and current_sequrntial_num_special_tokens == num_special_tokens
            ):
                clothest_token_idx[batch_i, seq_len_i] = current_clothest_token_idx
                current_clothest_token_idx = seq_len_i
                current_sequrntial_num_special_tokens = 0
            else:
                clothest_token_idx[batch_i, seq_len_i] = current_clothest_token_idx

    clothest_token_idx = clothest_token_idx.to(special_token_mask.device)
    return clothest_token_idx


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 2, 5, 17, 64])
@pytest.mark.parametrize("num_special_tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("pattern", ["none", "all", "random", "blocks"])
def test_optimized_matches_slow_randomized(bs, seq_len, num_special_tokens, pattern):
    torch.manual_seed(0)
    device = torch.device("cpu")

    if pattern == "none":
        special_mask = torch.zeros((bs, seq_len), dtype=torch.long, device=device)
    elif pattern == "all":
        special_mask = torch.ones((bs, seq_len), dtype=torch.long, device=device)
    elif pattern == "random":
        # Bias towards sparse specials to mimic realistic data
        probs = 0.15
        special_mask = (torch.rand((bs, seq_len), device=device) < probs).to(torch.long)
    elif pattern == "blocks":
        special_mask = torch.zeros((bs, seq_len), dtype=torch.long, device=device)
        # Insert several blocks of specials separated by normals
        for b in range(bs):
            pos = 0
            step = max(1, seq_len // 5)
            while pos < seq_len:
                # make a short run of specials
                run = min(num_special_tokens, seq_len - pos)
                special_mask[b, pos : pos + run] = 1
                pos += run + step
    else:
        raise ValueError("unknown pattern")

    ref = slow_reference_impl(special_mask, num_special_tokens=num_special_tokens)
    got = optimized_closest_idx(special_mask, num_special_tokens=num_special_tokens)

    assert ref.shape == got.shape
    assert ref.dtype == got.dtype == torch.long
    assert torch.equal(ref, got), f"Mismatch for pattern={pattern}, bs={bs}, L={seq_len}, N={num_special_tokens}"


def test_small_fixed_example():
    # Construct a clear, small example with two sentences and N=2 specials
    # 0 0 1 1  0 0 1 1
    special_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.long)
    num_special_tokens = 2

    ref = slow_reference_impl(special_mask, num_special_tokens=num_special_tokens)
    got = optimized_closest_idx(special_mask, num_special_tokens=num_special_tokens)

    assert torch.equal(ref, got)
