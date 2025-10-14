import torch
import torch.nn.functional as F
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaModel,
    sentence_attention_forward,
    sentence_attention_forward_flex,
    special_token_mask_to_clothest_token_idx_slow,
)
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention


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
    output_score_mod = flex_attention(query, key, value, score_mod=causal_mask)

    def block_causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them)
    block_mask = create_block_mask(block_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
    block_mask = block_mask.to("cpu")

    output_block_mask = flex_attention(query, key, value, block_mask=block_mask)

    output_sdpa = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    assert torch.allclose(output_score_mod, output_block_mask)
    assert torch.allclose(output_score_mod, output_sdpa)


def test_flex_attention_sentence_mask():
    # Create a random tensor of shape (batch_size, seq_len, hidden_size)
    batch_size = 1
    seq_len = 10

    class Dummy:
        # Ensure no KV repetition
        num_key_value_groups = 1

    # Create a FlexAttention layer

    num_special_tokens = 2
    special_embeddings_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0, 1, 1]]).repeat(batch_size, 1).bool()

    clothest_eos_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask,
        num_special_tokens=num_special_tokens,
    )

    attention_mask_bool = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).repeat(batch_size, 1).bool()

    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    expected_mask = torch.where(expected_mask.bool(), 1, -float("inf"))

    def custom_mask(b, h, q_idx, kv_idx):
        eos_token_idx = clothest_eos_token_idx[b, q_idx]

        causal_mask = (kv_idx <= q_idx) & attention_mask_bool[b, q_idx] & attention_mask_bool[b, kv_idx]
        eos_sync_tokens = causal_mask & special_embeddings_mask[b, kv_idx]
        causal_triu_mask = causal_mask & (kv_idx >= eos_token_idx)

        return causal_triu_mask | eos_sync_tokens

    generated_mask_bool = create_mask(custom_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device="cpu")[0, 0]
    generated_mask = torch.where(generated_mask_bool, 1, -float("inf"))

    assert (expected_mask == generated_mask).all()


def test_sentence_attention_forward():

    torch.set_default_device("cuda")

    torch.manual_seed(0)

    batch_size = 1
    num_heads = 4
    seq_len = 128
    head_dim = 64

    # Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # 2D attention mask (no padding)
    attention_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mark a few EOS tokens in the sequence
    special_embeddings_mask = torch.zeros_like(attention_mask_2d, dtype=torch.bool)
    special_positions = [10, 11, 12, 13, 60, 61, 62, 63]
    for pos in special_positions:
        special_embeddings_mask[:, pos] = 1

    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask,
        num_special_tokens=4,
    )

    cache_position = torch.arange(seq_len)

    causal_mask_4d_flex = SentenceLlamaModel._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention_flex(
        attention_mask=attention_mask_2d,
        sequence_length=seq_len,
        target_length=seq_len,
        dtype=query.dtype,
        device=query.device,
        cache_position=cache_position,
        batch_size=batch_size,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    # Build the 4D mask used by the SDPA path
    causal_mask_4d = SentenceLlamaModel._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
        attention_mask=attention_mask_2d,
        sequence_length=seq_len,
        target_length=seq_len,
        dtype=query.dtype,
        device=query.device,
        cache_position=cache_position,
        batch_size=batch_size,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    class Dummy:
        # Ensure no KV repetition
        num_key_value_groups = 1

    # scaling = 1.0 / (head_dim**0.5)
    scaling = None

    out_sdpa, _ = sentence_attention_forward(
        Dummy(), query, key, value, causal_mask_4d, dropout=0.0, scaling=scaling, is_causal=None
    )

    out_flex, _ = sentence_attention_forward_flex(
        Dummy(), query, key, value, attention_mask=causal_mask_4d_flex, scaling=scaling
    )

    # diff = (out_sdpa - out_flex).norm(2, dim=-1)
    # print("diff", diff.max())
    # print("out_sdpa", out_sdpa.shape)

    assert torch.allclose(out_sdpa, out_flex), "out_sdpa and out_flex are not close"


def test_sentence_attention_impl_equivalence():

    torch.set_default_device("cuda")

    torch.manual_seed(0)

    batch_size = 1
    num_heads = 4
    seq_len = 128
    head_dim = 64

    # Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # 2D attention mask (no padding)
    attention_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mark a few EOS tokens in the sequence
    special_embeddings_mask = torch.zeros_like(attention_mask_2d, dtype=torch.bool)
    special_positions = [3, 4, 5, 6, 100, 101, 102, 103]
    for pos in special_positions:
        special_embeddings_mask[:, pos] = 1

    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask,
        num_special_tokens=4,
    )

    cache_position = torch.arange(seq_len)

    causal_mask_4d_flex = SentenceLlamaModel._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention_flex(
        attention_mask=attention_mask_2d,
        sequence_length=seq_len,
        target_length=seq_len,
        dtype=query.dtype,
        device=query.device,
        cache_position=cache_position,
        batch_size=batch_size,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    # Build the 4D mask used by the SDPA path
    causal_mask_4d = SentenceLlamaModel._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
        attention_mask=attention_mask_2d,
        sequence_length=seq_len,
        target_length=seq_len,
        dtype=query.dtype,
        device=query.device,
        cache_position=cache_position,
        batch_size=batch_size,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    class Dummy:
        # Ensure no KV repetition
        num_key_value_groups = 1

    # scaling = 1.0 / (head_dim**0.5)
    scaling = None

    out_sdpa, _ = sentence_attention_forward(
        Dummy(), query, key, value, causal_mask_4d, dropout=0.0, scaling=scaling, is_causal=None
    )

    out_flex, _ = sentence_attention_forward_flex(
        Dummy(), query, key, value, attention_mask=causal_mask_4d_flex, scaling=scaling
    )

    diff = (out_sdpa - out_flex).norm(2, dim=-1)
    print("diff", diff.max())
    print("out_sdpa", out_sdpa.shape)

    # assert torch.allclose(out_sdpa, out_flex)

    # Backward: check gradient equivalence for Q, K, V
    query_sdpa = query.detach().clone().requires_grad_(True)
    key_sdpa = key.detach().clone().requires_grad_(True)
    value_sdpa = value.detach().clone().requires_grad_(True)

    out_sdpa, _ = sentence_attention_forward(
        Dummy(), query_sdpa, key_sdpa, value_sdpa, causal_mask_4d, dropout=0.0, scaling=scaling, is_causal=None
    )
    loss_sdpa = out_sdpa.sum()
    loss_sdpa.backward()

    query_flex = query.detach().clone().requires_grad_(True)
    key_flex = key.detach().clone().requires_grad_(True)
    value_flex = value.detach().clone().requires_grad_(True)

    out_flex, _ = sentence_attention_forward_flex(
        Dummy(),
        query_flex,
        key_flex,
        value_flex,
        attention_mask=causal_mask_4d_flex,
        scaling=scaling,
    )

    loss_flex = out_flex.sum()
    loss_flex.backward()

    assert torch.allclose(query_sdpa.grad, query_flex.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(key_sdpa.grad, key_flex.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(value_sdpa.grad, value_flex.grad, rtol=1e-4, atol=1e-6)
