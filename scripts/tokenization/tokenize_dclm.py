import os
import random
import sys
import time

import torch
from datasets import Dataset, load_dataset
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def process_stratified_dataset(dataset, tokenizer, max_length, num_eos_tokens, num_proc) -> Dataset:
    """
    Create a stratified subset (by sequence length) using parallel, batched operations.
    Avoids building large Python lists; relies on Arrow-backed Dataset ops.
    """

    total_dataset_size = 1000000
    num_bins = 100
    max_bin_size = total_dataset_size / num_bins

    # 1) Compute lengths, bins, and a deterministic random key per row (batched, parallel)
    def _compute_len_bin(batch, indices=None):
        texts = batch["text"]
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths = [len(ids) for ids in enc["input_ids"]]

        # Bin by length in [0, max_length], cap at last bin
        bins = [
            min(num_bins - 1, int((l * num_bins) / max_length)) if l <= max_length else num_bins - 1
            for l in lengths  # noqa: E741
        ]

        # Deterministic pseudo-random per index to enable stratified sampling
        seed = 42
        if indices is None:
            # Fall back: use per-batch positions if indices are not provided
            indices = list(range(len(lengths)))
        rnd = random.Random
        rands = [rnd(seed + int(i)).random() for i in indices]

        return {"_sa_length": lengths, "_sa_bin": bins, "_sa_rand": rands}

    dataset = dataset.map(
        _compute_len_bin,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Computing lengths and bins",
    )

    # 2) Filter out sequences longer than max_length (parallel, batched)
    dataset = dataset.filter(
        lambda l: [x <= max_length for x in l],  # noqa: E741
        input_columns=["_sa_length"],
        batched=True,
        num_proc=num_proc,
        desc="Filtering by max_length",
    )

    # 3) One pass to get counts per bin without materializing full columns
    bin_counts = [0] * num_bins
    for row in tqdm(dataset.to_iterable_dataset(num_shards=1), desc="Computing bin counts", total=len(dataset)):
        b = int(row["_sa_bin"])
        if 0 <= b < num_bins:
            bin_counts[b] += 1

    # 4) Compute per-bin keep probability to target max_bin_size items per bin
    keep_prob = []
    for count in bin_counts:
        if count <= 0:
            keep_prob.append(0.0)
        else:
            keep_prob.append(min(1.0, max_bin_size / float(count)))

    # 5) Probabilistic per-bin sampling using the precomputed random key (parallel, batched)
    def _keep_mask(batch):
        bins = batch["_sa_bin"]
        rands = batch["_sa_rand"]
        return [float(r) <= keep_prob[int(b)] for r, b in zip(rands, bins)]

    dataset = dataset.filter(
        _keep_mask,
        batched=True,
        num_proc=num_proc,
        desc="Stratified sampling by bin",
    )

    # 6) Drop helper columns
    dataset = (
        dataset.remove_columns(["_sa_length", "_sa_bin", "_sa_rand"])
        if set(["_sa_length", "_sa_bin", "_sa_rand"]).issubset(set(dataset.column_names))
        else dataset
    )

    print("Filtered stratified dataset length:", len(dataset))

    return dataset


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--with_eos_token", action="store_true")
    parser.add_argument("--only_stratified", action="store_true")
    parser.add_argument("--num_eos_tokens", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--timeout_minutes", type=int, default=-1)
    args = parser.parse_args()

    start_time = time.time()

    pretrained_model_name = args.pretrained_model_name  # HuggingFaceTB/SmolLM2-1.7B / unsloth/Llama-3.2-1B
    num_eos_tokens = args.num_eos_tokens
    max_length = args.max_length
    num_proc = args.num_proc
    pretrained_model_name_short = pretrained_model_name.split("/")[-1]

    print(f"pretrained_model_name_short: {pretrained_model_name_short}")

    suffix = f"_max_length_{max_length}"

    assert args.with_eos_token, "with_eos_token must be provided"

    if args.with_eos_token:
        suffix = f"{suffix}_with_eos_token_num_{num_eos_tokens}_merged"

    target_dir = f"./artifacts/data/dclm_tokenized_{pretrained_model_name_short}{suffix}"
    if args.num_shards > 1:
        target_dir = f"{target_dir}_shard_{args.shard_index}_of_{args.num_shards}"

    print("Will be saved to target_dir:", target_dir)

    tokenizer_class = type(AutoTokenizer.from_pretrained(pretrained_model_name)).__name__

    if args.with_eos_token:
        if tokenizer_class == "GPT2TokenizerFast":
            tokenizer_class = GPT2TokenizerFastEOS
        elif tokenizer_class == "PreTrainedTokenizerFast":
            tokenizer_class = PreTrainedTokenizerFastEOS
        elif tokenizer_class == "Qwen2TokenizerFast":
            tokenizer_class = Qwen2TokenizerFastEOS
        else:
            raise ValueError(f"Invalid tokenizer class: {tokenizer_class}")

    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name, num_eos_tokens=num_eos_tokens)
    assert tokenizer.num_eos_tokens == num_eos_tokens, "tokenizer num eos tokens set correctly"

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    special_token_ids = tokenizer.end_of_sentence_token_ids

    dataset_name = "mlfoundations/dclm-baseline-1.0"
    dataset_files = []

    assert args.num_shards == 1, "num_shards must be 1 for dclm"

    for shard_index_dataset in range(1, 6):
        for i in range(50):
            dataset_files.append(
                f"global-shard_{shard_index_dataset:02}_of_10/local-shard_0_of_10/shard_{i:08d}_processed.jsonl.zst"
            )

    dataset = load_dataset(dataset_name, num_proc=16, split="train", data_files=dataset_files)
    print(f"Loaded dataset from {dataset_name} with {len(dataset)} items")

    stratified_dataset_path = "./artifacts/data/dclm_tokenized_strarified"

    if os.path.exists(stratified_dataset_path):
        dataset = Dataset.load_from_disk(stratified_dataset_path)
        print(f"Loaded stratified dataset from {stratified_dataset_path}")
    else:
        dataset = process_stratified_dataset(dataset, tokenizer, max_length, num_eos_tokens, num_proc)
        dataset.save_to_disk(stratified_dataset_path)

    if args.only_stratified:
        sys.exit(0)

    def process_dataset_item(dataset_item):
        text = dataset_item["text"]

        tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

        input_ids = tokenized_inputs["input_ids"]

        special_embeddings_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for special_token_id in special_token_ids:
            special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            special_embeddings_mask, num_special_tokens=num_eos_tokens
        )

        return {
            "input_ids": input_ids[0].numpy().tolist(),
            "attention_mask": tokenized_inputs["attention_mask"],
            "special_embeddings_mask": special_embeddings_mask[0].numpy().tolist(),
            "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx[0].numpy().tolist(),
        }

    columns_to_keep = ["input_ids", "attention_mask", "special_embeddings_mask", "clothest_end_of_sentence_token_idx"]
    columns_to_remove = list(set(dataset.column_names) - set(columns_to_keep))

    dataset = dataset.map(process_dataset_item, num_proc=num_proc, remove_columns=columns_to_remove)

    dataset.save_to_disk(target_dir)
