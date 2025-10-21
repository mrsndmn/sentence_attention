import os
import random
import time

import torch
from datasets import Dataset, load_dataset
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def process_stratified_dataset(dataset, tokenizer, max_length, num_eos_tokens) -> Dataset:

    dclm_only_texts = []
    input_ids_lengths = []
    bins_counts = [0] * 100
    total_dataset_size = 1000000
    max_bin_size = total_dataset_size / len(bins_counts)

    pbar = tqdm(total=total_dataset_size)

    for item in dataset.shuffle(seed=42):
        # for item in tqdm(dataset.select(range(samples_count))):
        tokenized = tokenizer(item["text"])
        cur_len = len(tokenized["input_ids"])

        if cur_len > 16384:
            continue

        if pbar.n > total_dataset_size:
            break

        if pbar.n % 10 == 0:
            if args.timeout_minutes > 0:
                if time.time() - start_time > args.timeout_minutes * 60:
                    print("Timeout reached")
                    break

        current_bin = -1
        for i in range(len(bins_counts)):
            max_value = 16384 / len(bins_counts) * (i + 1)
            min_value = 16384 / len(bins_counts) * i
            if cur_len <= max_value and cur_len >= min_value:
                current_bin = i
                break

        assert current_bin != -1
        if bins_counts[current_bin] > max_bin_size:
            if cur_len > (8192 * (0.5 + random.random())):
                pass
            else:
                continue

        bins_counts[current_bin] += 1

        input_ids_lengths.append(cur_len)
        dclm_only_texts.append(item["text"])

        pbar.update(1)

    dataset = Dataset.from_dict({"text": dclm_only_texts})

    print("Filtered stratified dataset length: ", len(dataset))

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
        dataset = process_stratified_dataset(dataset, tokenizer, max_length, num_eos_tokens)
        dataset.save_to_disk(stratified_dataset_path)

    if args.only_stratified:
        os.exit(0)

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
