import torch
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow


def test_special_token_mask_to_clothest_token_idx():

    # Simple
    eos_tokens_mask = torch.tensor([[ 0, 1, 0, 0, 1, 0, 0, 1, 0, 1 ]]).bool()
    clothest_eos_token_idx = torch.tensor([[ 0, 0, 1, 1, 1, 4, 4, 4, 7, 7 ]])
    assert torch.allclose(clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask))

    # With multiple eos tokens
    eos_tokens_mask = torch.tensor([[ 0, 1, 1, 0, 1, 0, 0, 1, 0, 1 ]]).bool()
    clothest_eos_token_idx = torch.tensor([[ 0, 0, 1, 2, 2, 4, 4, 4, 7, 7 ]])
    assert torch.allclose(clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask))

    # With first token being eos
    eos_tokens_mask = torch.tensor([[ 1, 1, 0, 0, 1, 0, 0, 1, 0, 1 ]]).bool()
    clothest_eos_token_idx = torch.tensor([[ 0, 0, 1, 1, 1, 4, 4, 4, 7, 7 ]])
    assert torch.allclose(clothest_eos_token_idx, special_token_mask_to_clothest_token_idx_slow(eos_tokens_mask))
