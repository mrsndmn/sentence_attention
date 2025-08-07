import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow

def test_flex_attention_full():
    # Create a random tensor of shape (batch_size, seq_len, hidden_size)
    batch_size = 1
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    tensor = torch.randn(batch_size, num_heads, seq_len, hidden_size)

    # Create a FlexAttention layer
    def noop(score, b, h, q_idx, kv_idx):
        return score

    query, key, value = tensor, tensor, tensor

    # Forward pass
    output = flex_attention(query, key, value, score_mod=noop)

    output_2 = F.scaled_dot_product_attention(query, key, value, is_causal=False)

    assert torch.allclose(output, output_2)

def test_flex_attention_causal():
    # Create a random tensor of shape (batch_size, seq_len, hidden_size)
    batch_size = 1
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    tensor = torch.randn(batch_size, num_heads, seq_len, hidden_size)

    # Create a FlexAttention layer
    def causal_mask(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, -float("inf"))

    query, key, value = tensor, tensor, tensor

    # Forward pass
    output = flex_attention(query, key, value, score_mod=causal_mask)

    def block_causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them)
    block_mask = create_block_mask(block_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
    block_mask = block_mask.to('cpu')

    output_2 = flex_attention(query, key, value, block_mask=block_mask)

    output_3 = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    assert torch.allclose(output, output_2)
    assert torch.allclose(output, output_3)

def test_flex_attention_custom():
    # Create a random tensor of shape (batch_size, seq_len, hidden_size)
    batch_size = 1
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    tensor = torch.randn(batch_size, num_heads, seq_len, hidden_size)

    # Create a FlexAttention layer

    special_embeddings_mask = torch.tensor([[ 0, 1, 0, 0, 1, 0, 0, 1, 0, 1 ]]).repeat(batch_size, 1).bool()
    clothest_eos_token_idx = torch.tensor([[ 0, 0, 1, 1, 1, 4, 4, 4, 7, 7 ]]).repeat(batch_size, 1)
    attention_mask_bool = torch.tensor([[ 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 ]]).repeat(batch_size, 1).bool()

    expected_mask = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ])
    expected_mask = torch.where(expected_mask.bool(), 1, -float("inf"))

    def custom_mask(score, b, h, q_idx, kv_idx):
        eos_token_idx = clothest_eos_token_idx[b, q_idx]

        causal_mask = (kv_idx <= q_idx) & attention_mask_bool[b, q_idx] & attention_mask_bool[b, kv_idx]
        eos_sync_tokens = causal_mask & special_embeddings_mask[b, kv_idx]
        causal_triu_mask = causal_mask & (kv_idx >= eos_token_idx)

        return torch.where((causal_triu_mask | eos_sync_tokens), score, -float("inf"))


    generated_mask = torch.zeros(seq_len, seq_len)
    for q_idx in range(seq_len):
        for kv_idx in range(seq_len):
            generated_mask[q_idx, kv_idx] = custom_mask(1, 0, None, q_idx, kv_idx)

    assert (expected_mask == generated_mask).all()

    query, key, value = tensor, tensor, tensor
    output = flex_attention(query, key, value, score_mod=custom_mask)


    breakpoint()




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
