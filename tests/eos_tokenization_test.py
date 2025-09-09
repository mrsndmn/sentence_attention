import torch

from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow


def test_special_token_mask_to_clothest_token_idx():

    # Simple
    eos_tokens_mask = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]).bool()
    clothest_eos_token_idx = torch.tensor([[0, 0, 1, 1, 1, 4, 4, 4, 7, 7]])
    assert torch.allclose(
        clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask, num_special_tokens=1)
    )

    # With multiple eos tokens
    eos_tokens_mask = torch.tensor([[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]]).bool()
    clothest_eos_token_idx = torch.tensor([[0, 0, 0, 2, 2, 4, 4, 4, 7, 7]])
    assert torch.allclose(
        clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask, num_special_tokens=1)
    )

    # With first token being eos
    eos_tokens_mask = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]]).bool()
    clothest_eos_token_idx = torch.tensor([[0, 0, 0, 0, 0, 4, 4, 4, 4, 8]])
    assert torch.allclose(
        clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask, num_special_tokens=2)
    )
