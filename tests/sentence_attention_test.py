import time

import numpy as np
import torch
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaModel,
    sentence_attention_forward,
    special_token_mask_to_clothest_token_idx_slow,
)


def test_sentence_attention_impl_equivalence():

    torch.set_default_device("cuda")

    torch.manual_seed(0)

    batch_size = 1
    num_heads = 4
    seq_len = 64000
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
        training = False

    scaling = 1.0 / (head_dim**0.5)
    # scaling = None

    time_taken = []
    for _ in range(100):
        time_start = time.time()
        out_sdpa, _ = sentence_attention_forward(
            Dummy(), query, key, value, causal_mask_4d, dropout=0.0, scaling=scaling, is_causal=None
        )
        torch.cuda.synchronize()
        time_end = time.time()
        time_taken.append(time_end - time_start)
    print("time taken to prefill sentence_attention", np.mean(time_taken))

    time_taken2 = []
    for _ in range(100):
        time_start = time.time()
        out_flex, _ = sentence_attention_forward(Dummy(), query, key, value, attention_mask=None, scaling=scaling)
        torch.cuda.synchronize()
        time_end = time.time()
        time_taken2.append(time_end - time_start)
    print("time taken to prefill no mask", np.mean(time_taken2))
