import json
import math
import os
from typing import Dict, List, Literal

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill
from tqdm import tqdm
from transformers import AutoTokenizer


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def evaluate_pg19_ppl(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    model_type: Literal["sentence", "vanilla"] = "sentence",
    max_samples: int = 10,
    max_length: int = 1000,
) -> Dict:
    """
    Compute PPL on PG19 for either SentenceLlama or vanilla Llama and return a JSON-serializable dict.
    The returned dict contains overall ppl, per-sample ppls, aggregated prefix metrics, and diagnostics.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    dataset = datasets.Dataset.load_from_disk(dataset_path)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    print(
        f"Evaluating PG19 on dataset with max_samples={max_samples} (PG19 length={len(dataset)}) and max_length={max_length} on {device}"
    )

    tokens_log_probas: List[float] = []
    samples_ppls: List[float] = []
    samples_ppls_no_eos: List[float] = []
    all_input_ids: List[int] = []
    all_kv_lengths: List[int] = []

    prefix_lengths = [length for length in range(1024, max_length + 1, 1024)]
    tokens_log_probas_by_prefix: Dict[int, List[float]] = {length: [] for length in prefix_lengths}
    tokens_log_probas_by_prefix_no_eos: Dict[int, List[float]] = {length: [] for length in prefix_lengths}
    samples_ppls_by_prefix: Dict[int, List[float]] = {length: [] for length in prefix_lengths}
    samples_ppls_by_prefix_no_eos: Dict[int, List[float]] = {length: [] for length in prefix_lengths}

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating PG19"):
            current_tokens_log_probas: List[float] = []
            current_tokens_log_probas_no_eos: List[float] = []

            input_ids = tokenizer.encode(item["text"], return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            if model_type == "sentence":
                special_token_ids = tokenizer.end_of_sentence_token_ids
                special_embeddings_mask = torch.zeros_like(input_ids)
                for special_token_id in special_token_ids:
                    special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

                clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                    special_embeddings_mask,
                    num_special_tokens=len(special_token_ids),
                )

                def outputs_hook(inner_input_ids, outputs, prev_sentence_i, sentence_i):
                    outputs_logits_normed = F.log_softmax(outputs.logits.float(), dim=-1)
                    # if sentence_i == inner_input_ids.shape[1]:
                    #     labels = inner_input_ids[:, prev_sentence_i + 1 : sentence_i]
                    #     labels = labels.unsqueeze(-1)
                    #     log_probas = torch.gather(outputs_logits_normed[:, :-1, :], dim=-1, index=labels)
                    # else:
                    input_ids = inner_input_ids[:, prev_sentence_i:sentence_i]
                    labels = inner_input_ids[:, prev_sentence_i + 1 : sentence_i + 1]
                    labels = labels.unsqueeze(-1)
                    log_probas = torch.gather(outputs_logits_normed, dim=-1, index=labels)

                    log_probas_list = log_probas[0, :, 0].cpu().numpy().tolist()
                    current_tokens_log_probas.extend(log_probas_list)  # noqa: B023

                    log_probas_list_no_eos = []
                    for token_id, log_proba in zip(input_ids[0, :].cpu().numpy().tolist(), log_probas_list):
                        # Ignore all except the last EOS token
                        if token_id in special_token_ids[:-1]:  # noqa: B023
                            continue
                        log_probas_list_no_eos.append(log_proba)

                    current_tokens_log_probas_no_eos.extend(log_probas_list_no_eos)  # noqa: B023

                outputs = scrooge_prefill(
                    model,
                    input_ids,
                    attention_mask=attention_mask,
                    special_embeddings_mask=special_embeddings_mask,
                    clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                    outputs_hook=outputs_hook,
                )
                kv_seq_len = outputs["past_key_values"].get_seq_length()
                del outputs
            else:
                # dynamic_cache = DynamicCache()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits.float()
                log_probs = F.log_softmax(logits, dim=-1)
                labels = input_ids[:, 1:]
                gathered = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1))
                vanill_ppls = gathered[0, :, 0].cpu().numpy().tolist()
                current_tokens_log_probas.extend(vanill_ppls)
                current_tokens_log_probas_no_eos.extend(vanill_ppls)
                kv_seq_len = input_ids.shape[1]
                del outputs

            ppl_sample = (
                math.exp(-float(np.mean(current_tokens_log_probas))) if len(current_tokens_log_probas) > 0 else float("nan")
            )

            ppl_sample_no_eos = (
                math.exp(-float(np.mean(current_tokens_log_probas_no_eos)))
                if len(current_tokens_log_probas_no_eos) > 0
                else float("nan")
            )

            samples_ppls.append(ppl_sample)
            samples_ppls_no_eos.append(ppl_sample_no_eos)

            tokens_log_probas.extend(current_tokens_log_probas)
            all_input_ids.append(int(input_ids.shape[1]))
            all_kv_lengths.append(int(kv_seq_len))

            for pref_len in prefix_lengths:
                n_pred = min(len(current_tokens_log_probas), max(0, pref_len - 1))
                if n_pred > 0:
                    pref_ppl = math.exp(-float(np.mean(current_tokens_log_probas[:n_pred])))
                    samples_ppls_by_prefix[pref_len].append(pref_ppl)
                    tokens_log_probas_by_prefix[pref_len].extend(current_tokens_log_probas[:n_pred])

                    pref_ppl_no_eos = math.exp(-float(np.mean(current_tokens_log_probas_no_eos[:n_pred])))
                    samples_ppls_by_prefix_no_eos[pref_len].append(pref_ppl_no_eos)
                    tokens_log_probas_by_prefix_no_eos[pref_len].extend(current_tokens_log_probas_no_eos[:n_pred])
                else:
                    samples_ppls_by_prefix[pref_len].append(float("nan"))
                    samples_ppls_by_prefix_no_eos[pref_len].append(float("nan"))

    overall_ppl = math.exp(-float(np.mean(tokens_log_probas))) if len(tokens_log_probas) > 0 else float("nan")

    aggregated_by_prefix = {}
    aggregated_by_prefix_no_eos = {}
    for pref_len in prefix_lengths:
        token_logs = tokens_log_probas_by_prefix[pref_len]
        if len(token_logs) == 0:
            continue
        ppl_pref = math.exp(-float(np.mean(token_logs)))
        aggregated_by_prefix[str(pref_len)] = {
            "ppl": float(ppl_pref),
            "num_tokens": int(len(token_logs)),
        }

        token_logs_no_eos = tokens_log_probas_by_prefix_no_eos[pref_len]
        if len(token_logs_no_eos) == 0:
            continue
        ppl_pref_no_eos = math.exp(-float(np.mean(token_logs_no_eos)))
        aggregated_by_prefix_no_eos[str(pref_len)] = {
            "ppl": float(ppl_pref_no_eos),
            "num_tokens": int(len(token_logs_no_eos)),
        }

    result = {
        "model_type": model_type,
        "max_samples": int(max_samples),
        "max_length": int(max_length),
        "overall_ppl": float(overall_ppl) if not math.isnan(overall_ppl) else None,
        "samples_ppls_mean": float(np.nanmean(samples_ppls)) if len(samples_ppls) > 0 else None,
        "samples_ppls_std": float(np.nanstd(samples_ppls)) if len(samples_ppls) > 0 else None,
        "per_sample_ppls": [float(x) if x == x else None for x in samples_ppls],
        "aggregated_ppl_by_prefix": aggregated_by_prefix,
        "aggregated_ppl_by_prefix_no_eos": aggregated_by_prefix_no_eos,
        "diagnostics": {
            "input_ids_lengths": all_input_ids,
            "kv_cache_lengths": all_kv_lengths,
            "compression_ratios": [float(i) / float(k) if k > 0 else None for i, k in zip(all_input_ids, all_kv_lengths)],
            "max_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        },
    }

    return result


def save_pg19_results_json(output_dir: str, results: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pg19.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path
