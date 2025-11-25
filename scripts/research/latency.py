#!/usr/bin/env python3
import argparse
import os
import time
from typing import List, Optional, Tuple

import torch

from sentence_attention.models.checkpoint import load_model_from_checkpoint
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import (
    full_prefill_small_kv_cache,
)


DEFAULT_CHECKPOINT = (
    "artifacts/experiments/eos_8/" "sentence_Llama-3.2-3B_ft_16k_colddown_full_num_eos_tokens_8_2DJTIOM9/" "checkpoint-21000"
)


def find_non_special_token_id(tokenizer, forbidden_ids: List[int]) -> int:
    # Try a few common pieces until we get a non-special, non-forbidden id
    candidates = [" the", " a", " to", " in", " of", " and", " is", " it", " I", " you"]
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        for tid in ids:
            if tid not in forbidden_ids and tid >= 0:
                return tid
    # Fallback to id 1 if safe
    for tid in range(3, tokenizer.vocab_size):
        if tid not in forbidden_ids:
            return tid
    raise RuntimeError("Could not find a non-special token id")


def build_prompt_ids(
    tokenizer,
    total_len: int,
    eos_ids: List[int],
    eos_interval: Optional[int],
) -> torch.LongTensor:
    """
    Build a synthetic prompt of length total_len.
    - If eos_interval is None: fill with normal tokens only (no EOS).
    - Else: insert a block of all eos_ids after every eos_interval normal tokens.
    """
    normal_id = find_non_special_token_id(tokenizer, forbidden_ids=eos_ids)

    ids: List[int] = []
    if eos_interval is None:
        ids = [normal_id] * total_len
        return torch.tensor([ids], dtype=torch.long)

    block: List[int] = [normal_id] * eos_interval + eos_ids
    while len(ids) + len(block) <= total_len:
        ids.extend(block)
    # Tail
    remaining = total_len - len(ids)
    if remaining > 0:
        # Prefer to fill remaining with normal tokens, but ensure we don't exceed total_len
        # If remaining is smaller than eos block, add only normals.
        ids.extend([normal_id] * remaining)

    return torch.tensor([ids[:total_len]], dtype=torch.long)


@torch.inference_mode()
def measure_prefill_latency(model, input_ids: torch.LongTensor, device: torch.device, runs: int = 3) -> Tuple[float, float]:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    input_ids = input_ids.to(device)

    # Warmup
    _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = sum(times) / len(times)
    if len(times) > 1:
        var = sum((x - mean_ms) ** 2 for x in times) / (len(times) - 1)
        std_ms = var**0.5
    else:
        std_ms = 0.0
    return mean_ms, std_ms


def build_special_mask_and_closest_idx(input_ids: torch.LongTensor, eos_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    special_mask = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
    for eid in eos_ids:
        special_mask[input_ids == eid] = 1
    closest_idx = special_token_mask_to_clothest_token_idx_slow(special_mask, num_special_tokens=len(eos_ids))
    return special_mask, closest_idx


@torch.inference_mode()
def measure_decode_latency_with_compression(
    model,
    tokenizer,
    total_len: int,
    compression_rate: int,
    device: torch.device,
    new_tokens: int = 32,
    runs: int = 3,
) -> Tuple[float, float, int]:
    """
    Measure pure decode latency (excluding prefill) by:
    1) Building a prompt with uniform EOS insertion controlling effective compression.
    2) Running scrooge full prefill to precompress KV cache.
    3) Calling model.generate starting from the returned single-token input with populated KV.
    Returns (avg_ms_per_token, std_ms_per_token, kv_len_after_prefill).
    """
    eos_ids: List[int] = getattr(model.config, "end_of_sentence_token_ids", None)
    if not eos_ids or len(eos_ids) == 0:
        raise ValueError("Model config has no end_of_sentence_token_ids; sentence attention is required.")

    # Heuristic: interval ≈ compression_rate * num_eos_tokens to approximate r× compression
    eos_interval = compression_rate * len(eos_ids)

    # Build prompt ids and attention mask
    input_ids = build_prompt_ids(tokenizer, total_len=total_len, eos_ids=eos_ids, eos_interval=eos_interval).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # Prepare sentence-attention metadata
    special_mask, closest_idx = build_special_mask_and_closest_idx(input_ids, eos_ids)

    # Prefill once to get reference KV length and warm up generate
    sc_ref = full_prefill_small_kv_cache(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_embeddings_mask=special_mask,
        clothest_end_of_sentence_token_idx=closest_idx,
        output_hidden_states=False,
    )
    kv_len = sc_ref["past_key_values"].get_seq_length()

    # Warmup generate once (adjust cache_position to start right after current KV length)
    warmup_kwargs = dict(
        input_ids=sc_ref["input_ids"],
        attention_mask=sc_ref["attention_mask"],
        special_embeddings_mask=sc_ref["special_embeddings_mask"],
        clothest_end_of_sentence_token_idx=sc_ref["clothest_end_of_sentence_token_idx"],
        past_key_values=sc_ref["past_key_values"],
        cache_position=torch.arange(kv_len, kv_len + 1, device=device),
        max_new_tokens=new_tokens,
        return_dict_in_generate=False,
    )
    _ = model.generate(**warmup_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times: List[float] = []
    for _ in range(runs):
        # Recompute prefill to restore the exact same starting KV state
        sc = full_prefill_small_kv_cache(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_mask,
            clothest_end_of_sentence_token_idx=closest_idx,
            output_hidden_states=False,
        )
        # Ensure cache_position is contiguous with current compressed KV length
        kv_len_i = sc["past_key_values"].get_seq_length()
        cache_pos = torch.arange(kv_len_i, kv_len_i + 1, device=device)
        gen_kwargs = dict(
            input_ids=sc["input_ids"],
            attention_mask=sc["attention_mask"],
            special_embeddings_mask=sc["special_embeddings_mask"],
            clothest_end_of_sentence_token_idx=sc["clothest_end_of_sentence_token_idx"],
            past_key_values=sc["past_key_values"],
            cache_position=cache_pos,
            max_new_tokens=new_tokens,
            return_dict_in_generate=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(**gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        times.append(total_ms / float(new_tokens))

    mean_ms = sum(times) / len(times)
    if len(times) > 1:
        var = sum((x - mean_ms) ** 2 for x in times) / (len(times) - 1)
        std_ms = var**0.5
    else:
        std_ms = 0.0
    return mean_ms, std_ms, kv_len


@torch.inference_mode()
def measure_decode_latency_vanilla(
    model,
    tokenizer,
    total_len: int,
    device: torch.device,
    new_tokens: int = 32,
    runs: int = 3,
) -> Tuple[float, float, int]:
    """
    Baseline decode latency for vanilla models (no sentence attention).
    1) Prefill the whole prompt to get past_key_values.
    2) Start generate with a single next token and the populated KV cache.
    Returns (avg_ms_per_token, std_ms_per_token, kv_len_after_prefill).
    """
    # Build prompt without EOS insertions
    input_ids = build_prompt_ids(tokenizer, total_len=total_len, eos_ids=[], eos_interval=None).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    def kv_length(pkv) -> int:
        if hasattr(pkv, "get_seq_length"):
            return int(pkv.get_seq_length())
        # Legacy tuple format: ((k0, v0), (k1, v1), ...)
        k0 = pkv[0][0]
        return int(k0.shape[-2])

    # Prefill once and warmup generate
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past = outputs.past_key_values
    kv_len = kv_length(past)
    next_input_ids = outputs.logits[:, -1:].argmax(dim=-1)
    warmup_kwargs = dict(
        input_ids=next_input_ids,
        attention_mask=torch.ones((1, kv_len + 1), device=device, dtype=torch.long),
        past_key_values=past,
        cache_position=torch.arange(kv_len, kv_len + 1, device=device),
        max_new_tokens=new_tokens,
        return_dict_in_generate=False,
    )
    _ = model.generate(**warmup_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs (recompute prefill before each run to keep consistent starting state)
    times: List[float] = []
    for _ in range(runs):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_i = outputs.past_key_values
        kv_len_i = kv_length(past_i)
        next_ids_i = outputs.logits[:, -1:].argmax(dim=-1)
        gen_kwargs = dict(
            input_ids=next_ids_i,
            attention_mask=torch.ones((1, kv_len_i + 1), device=device, dtype=torch.long),
            past_key_values=past_i,
            cache_position=torch.arange(kv_len_i, kv_len_i + 1, device=device),
            max_new_tokens=new_tokens,
            return_dict_in_generate=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(**gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        times.append(total_ms / float(new_tokens))

    mean_ms = sum(times) / len(times)
    if len(times) > 1:
        var = sum((x - mean_ms) ** 2 for x in times) / (len(times) - 1)
        std_ms = var**0.5
    else:
        std_ms = 0.0
    return mean_ms, std_ms, kv_len


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefill and decode latency for Sentence LLaMA.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Prompt sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--compression-rates",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Compression rates (generate-only).",
    )
    parser.add_argument(
        "--decode-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate for decode benchmark.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs to average.")
    parser.add_argument("--attention-implementation", type=str, default=None, help="Override attention impl.")
    args = parser.parse_args()

    model, tokenizer = load_model_from_checkpoint(args.checkpoint, attention_implementation=args.attention_implementation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    eos_ids = getattr(model.config, "end_of_sentence_token_ids", None)
    if eos_ids:
        print(f"num_eos_tokens: {len(eos_ids)}")
    else:
        print("No end_of_sentence_token_ids detected: running vanilla baseline for decode.")

    # print("\n=== Prefill latency (no compression parameter) ===")
    # for L in args.seq_lens:
    #     prompt_ids = build_prompt_ids(tokenizer, total_len=L, eos_ids=eos_ids or [], eos_interval=None)
    #     mean_ms, std_ms = measure_prefill_latency(model, prompt_ids, device=device, runs=args.runs)
    #     print(f"- seq_len={L:5d}: prefill {mean_ms:.2f} ± {std_ms:.2f} ms")

    if eos_ids and len(eos_ids) > 0:
        print("\n=== Decode latency with compression (generate) ===")
        print(f"(Per-token latency over {args.decode_new_tokens} new tokens)")
        for L in args.seq_lens:
            print(f"\nseq_len={L}")
            for r in args.compression_rates:
                try:
                    per_token_ms, per_token_std, kv_len = measure_decode_latency_with_compression(
                        model=model,
                        tokenizer=tokenizer,
                        total_len=L,
                        compression_rate=r,
                        device=device,
                        new_tokens=args.decode_new_tokens,
                        runs=args.runs,
                    )
                    print(
                        f"- compression={r}x: kv_len_after_prefill={kv_len:5d}, "
                        f"decode {per_token_ms:.3f} ± {per_token_std:.3f} ms/token"
                    )
                except Exception as e:
                    print(f"- compression={r}x: FAILED ({e})")
    else:
        print("\n=== Decode latency (vanilla generate) ===")
        print(f"(Per-token latency over {args.decode_new_tokens} new tokens)")
        for L in args.seq_lens:
            try:
                per_token_ms, per_token_std, kv_len = measure_decode_latency_vanilla(
                    model=model,
                    tokenizer=tokenizer,
                    total_len=L,
                    device=device,
                    new_tokens=args.decode_new_tokens,
                    runs=args.runs,
                )
                print(
                    f"- seq_len={L:5d}: kv_len_after_prefill={kv_len:5d}, "
                    f"decode {per_token_ms:.3f} ± {per_token_std:.3f} ms/token"
                )
            except Exception as e:
                print(f"- seq_len={L:5d}: FAILED ({e})")


if __name__ == "__main__":
    # Allow running from project root or script directory
    if not os.path.isdir(DEFAULT_CHECKPOINT) and os.path.isdir(os.path.join(os.getcwd(), DEFAULT_CHECKPOINT)):
        DEFAULT_CHECKPOINT = os.path.join(os.getcwd(), DEFAULT_CHECKPOINT)
    main()
