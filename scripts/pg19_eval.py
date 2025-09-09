import argparse
import os
import pickle

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


def str_to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL on PG19 with sentence attention or vanilla Llama.")

    # checkpoint_dir = "./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR/checkpoint-1349/"
    # checkpoint_dir = "./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft2_full_num_eos_tokens_4_MV7M599S/checkpoint-10794/"
    # checkpoint_dir = "unsloth/Llama-3.2-3B"

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        default="./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR/checkpoint-1349/",
        help="Path or HF hub id of the model checkpoint.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["sentence", "vanilla"],
        default="sentence",
        help="Use 'sentence' for SentenceLlamaForCausalLM or 'vanilla' for LlamaForCausalLM.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/pg19_test",
        help="Path to the PG19 dataset saved with datasets.save_to_disk.",
    )
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples to evaluate (-1 for all).")
    parser.add_argument("--max-length", type=int, default=1000, help="Max sequence length for tokenization.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/ppl/",
        help="Directory to save outputs (CSV and PKL).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    checkpoint_dir = args.checkpoint_dir
    max_samples = args.max_samples
    max_tokens_length = args.max_length
    device = args.device
    dtype = str_to_dtype(args.dtype)

    if args.model_type == "sentence":
        model_class = SentenceLlamaForCausalLM
    else:
        model_class = LlamaForCausalLM

    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=dtype)
    model.eval()
    model.to(device)
    if args.model_type == "sentence":
        model.config._attn_implementation = "sentence_attention"
    # model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    dataset = datasets.Dataset.load_from_disk(args.dataset_path)
    if max_samples != -1:
        dataset = dataset.select(range(max_samples))

    tokens_log_probas = []
    samples_ppls = []

    per_sample_ppls = []
    all_samples_ppls = []
    all_input_ids = []
    all_kv_lengths = []

    # Compute PPL for prefixes: 1024 and all multiples up to max_length
    prefix_lengths = [length for length in range(1024, max_tokens_length + 1, 1024)]
    tokens_log_probas_by_prefix = {length: [] for length in prefix_lengths}
    samples_ppls_by_prefix = {length: [] for length in prefix_lengths}

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():

        for item in tqdm(dataset):

            current_tokens_log_probas = []

            input_ids = tokenizer.encode(item["text"], return_tensors="pt", max_length=max_tokens_length, truncation=True)
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            if args.model_type == "sentence":
                special_token_ids = tokenizer.end_of_sentence_token_ids
                special_embeddings_mask = torch.zeros_like(input_ids)
                for special_token_id in special_token_ids:
                    special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

                clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                    special_embeddings_mask,
                    num_special_tokens=len(special_token_ids),
                )

                def outputs_hook(input_ids, outputs, prev_sentence_i, sentence_i):
                    outputs_logits_normed = F.log_softmax(outputs.logits.float(), dim=-1)

                    if sentence_i == input_ids.shape[1]:
                        labels = input_ids[:, prev_sentence_i + 1 : sentence_i]
                        labels = labels.unsqueeze(-1)
                        log_probas = torch.gather(outputs_logits_normed[:, :-1, :], dim=-1, index=labels)
                    else:
                        labels = input_ids[:, prev_sentence_i + 1 : sentence_i + 1]
                        labels = labels.unsqueeze(-1)
                        log_probas = torch.gather(outputs_logits_normed, dim=-1, index=labels)

                    current_tokens_log_probas.extend(log_probas[0, :, 0].cpu().numpy().tolist())  # noqa: B023

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
                # Vanilla simple forward pass without special masks or scrooge prefill
                dynamic_cache = DynamicCache()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=dynamic_cache
                )
                logits = outputs.logits.float()
                log_probs = F.log_softmax(logits, dim=-1)
                labels = input_ids[:, 1:]
                gathered = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1))
                current_tokens_log_probas.extend(gathered[0, :, 0].cpu().numpy().tolist())
                kv_seq_len = input_ids.shape[1]

                del outputs

            ppl = np.exp(-np.mean(current_tokens_log_probas))
            samples_ppls.append(ppl)
            per_sample_ppls.append(current_tokens_log_probas)
            print("Sample PPL", ppl)
            print("input_ids", input_ids.shape)
            print("kv_length", kv_seq_len)
            print("compression ratio", input_ids.shape[1] / kv_seq_len)

            tokens_log_probas.extend(current_tokens_log_probas)
            all_samples_ppls.append(ppl)
            all_input_ids.append(input_ids.shape[1])
            all_kv_lengths.append(kv_seq_len)

            # Per-sample and aggregated PPL for each requested prefix length
            for pref_len in prefix_lengths:
                n_pred = min(len(current_tokens_log_probas), max(0, pref_len - 1))
                if n_pred > 0:
                    pref_ppl = np.exp(-np.mean(current_tokens_log_probas[:n_pred]))
                    samples_ppls_by_prefix[pref_len].append(pref_ppl)
                    tokens_log_probas_by_prefix[pref_len].extend(current_tokens_log_probas[:n_pred])
                else:
                    samples_ppls_by_prefix[pref_len].append(np.nan)

        ppl = np.exp(-np.mean(tokens_log_probas))
        print("Full PPL", ppl)
        print("Samples PPLs", np.mean(samples_ppls), "std", np.std(samples_ppls))

        # Print aggregated PPL for each prefix length
        for pref_len in prefix_lengths:
            token_logs = tokens_log_probas_by_prefix[pref_len]
            if len(token_logs) == 0:
                continue
            ppl_pref = np.exp(-np.mean(token_logs))
            print(f"PPL@{pref_len}", ppl_pref, "num_tokens", len(token_logs))

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "pg19_samples_ppls.csv")
    df = pd.DataFrame(samples_ppls)
    df.to_csv(csv_path, index=False)
    print("Saved sample-level PPLs to", csv_path)

    # Save sample-level PPLs by prefix length (columns per prefix)
    if len(prefix_lengths) > 0:
        by_prefix_csv_path = os.path.join(args.output_dir, "pg19_samples_ppls_by_prefix.csv")
        df_prefix = pd.DataFrame({f"ppl_at_{length}": samples_ppls_by_prefix[length] for length in prefix_lengths})
        # Also include full sequence sample PPLs for convenience
        df_prefix.insert(0, "ppl_full", samples_ppls)
        df_prefix.to_csv(by_prefix_csv_path, index=False)
        print("Saved sample-level PPLs by prefix length to", by_prefix_csv_path)

        # Save aggregated PPL by prefix length
        agg_csv_path = os.path.join(args.output_dir, "pg19_aggregated_ppl_by_prefix.csv")
        df_agg = pd.DataFrame(
            {
                "prefix_length": prefix_lengths,
                "ppl": [
                    (
                        np.exp(-np.mean(tokens_log_probas_by_prefix[length]))
                        if len(tokens_log_probas_by_prefix[length]) > 0
                        else np.nan
                    )
                    for length in prefix_lengths
                ],
                "num_tokens": [len(tokens_log_probas_by_prefix[length]) for length in prefix_lengths],
            }
        )
        df_agg.to_csv(agg_csv_path, index=False)
        print("Saved aggregated PPL by prefix length to", agg_csv_path)

    pkl_path = os.path.join(args.output_dir, "pg19_per_sample_ppls.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(per_sample_ppls, f)

    print("Max CUDA memory allocated", f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}", "GB")

    # Optionally save additional diagnostics if needed:
    # diag_df = pd.DataFrame({
    #     "input_ids_len": all_input_ids,
    #     "kv_length": all_kv_lengths,
    #     "ppl": all_samples_ppls,
    # })
    # diag_df.to_csv(os.path.join(args.output_dir, "pg19_samples_ppls_sentence_attention.csv"), index=False)
