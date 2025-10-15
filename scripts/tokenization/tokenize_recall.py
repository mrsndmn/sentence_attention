import multiprocessing as mp
import os

import torch
import torch.nn as nn
from datasets import Dataset
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def _generate_one_sample(task):
    # task: Tuple[int, bool] -> (seed, no_answer)
    seed, no_answer = task
    import random as _random

    import numpy as _np
    import torch as _torch
    from sentence_attention.evaluation.my_recall import generate_random_sample_full as _gen
    from wonderwords import RandomWord as _RandomWord

    _random.seed(seed)
    _np.random.seed(seed % (2**32 - 1))
    _torch.manual_seed(seed)

    rw = _RandomWord()
    return _gen(num_examples=100, random_word=rw)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--with_eos_token", action="store_true")
    parser.add_argument("--num_eos_tokens", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--with_labels_on_answer", action="store_true")
    args = parser.parse_args()

    pretrained_model_name = args.pretrained_model_name  # HuggingFaceTB/SmolLM2-1.7B / unsloth/Llama-3.2-1B
    num_eos_tokens = args.num_eos_tokens
    max_length = args.max_length
    num_proc = args.num_proc
    if num_proc <= 0:
        num_proc = None

    pretrained_model_name_short = pretrained_model_name.split("/")[-1]

    print(f"pretrained_model_name_short: {pretrained_model_name_short}")

    suffix = f"_max_length_{max_length}_num_samples_{args.num_samples}"

    if args.with_eos_token:
        suffix = f"{suffix}_with_eos_token_num_{num_eos_tokens}_merged"

    if args.with_labels_on_answer:
        suffix = f"{suffix}_with_labels_on_answer"

    target_dir = f"./artifacts/data/synthetic_niah_tokenized_{pretrained_model_name_short}{suffix}"

    print("Will be saved to target_dir:", target_dir)

    tokenizer_class_name = type(AutoTokenizer.from_pretrained(pretrained_model_name)).__name__

    if not args.with_eos_token:
        raise ValueError("--with_eos_token must be provided for this script")

    if tokenizer_class_name == "GPT2TokenizerFast":
        tokenizer_class = GPT2TokenizerFastEOS
    elif tokenizer_class_name == "PreTrainedTokenizerFast":
        tokenizer_class = PreTrainedTokenizerFastEOS
    elif tokenizer_class_name == "Qwen2TokenizerFast":
        tokenizer_class = Qwen2TokenizerFastEOS
    else:
        raise ValueError(f"Invalid tokenizer class: {tokenizer_class_name}")

    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name, num_eos_tokens=num_eos_tokens)
    assert tokenizer.num_eos_tokens == num_eos_tokens, "tokenizer num eos tokens set correctly"

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    special_token_ids = tokenizer.end_of_sentence_token_ids

    # Parallel sample generation with multiprocessing, progress per sample
    num_workers = num_proc
    base_seed = int.from_bytes(os.urandom(8), "little")
    tasks = [(base_seed + 10007 * (i + 1), False) for i in range(args.num_samples)]

    generated_samples_with_answer = []
    generated_samples_without_answer = []

    print("num_workers", num_workers)
    if num_workers is not None:
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            for sample in tqdm(
                pool.imap_unordered(_generate_one_sample, tasks, chunksize=1),
                total=len(tasks),
                desc="Generating samples",
            ):
                generated_samples_with_answer.append(sample["sample_with_answer"])
                generated_samples_without_answer.append(sample["sample_without_answer"])
    else:
        for sample in tqdm(
            tasks,
            total=len(tasks),
            desc="Generating samples sequentially",
        ):
            result = _generate_one_sample(sample)
            generated_samples_with_answer.append(result["sample_with_answer"])
            generated_samples_without_answer.append(result["sample_without_answer"])

    # Validate counts and absence of duplicates
    assert (
        len(generated_samples_with_answer) == args.num_samples
    ), f"Generated {len(generated_samples_with_answer)} samples, expected {args.num_samples}"
    unique_count = len(set(generated_samples_with_answer))
    if unique_count != len(generated_samples_with_answer):
        dup_count = len(generated_samples_with_answer) - unique_count
        raise RuntimeError(f"Duplicate samples detected after pooling: {dup_count}")

    dataset = Dataset.from_dict(
        {"text": generated_samples_with_answer, "text_without_answer": generated_samples_without_answer}
    )

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

        attention_mask = tokenized_inputs["attention_mask"]
        result = {
            "input_ids": input_ids[0].numpy().tolist(),
            "attention_mask": attention_mask,
            "special_embeddings_mask": special_embeddings_mask[0].numpy().tolist(),
            "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx[0].numpy().tolist(),
        }

        if args.with_labels_on_answer:
            result_labels = input_ids.clone()

            text_without_answer = dataset_item["text_without_answer"]
            tokenized_inputs_without_answer = tokenizer(text_without_answer, truncation=False, return_tensors="pt")

            full_sample_tokens = attention_mask.sum().item()
            no_answer_tokens = tokenized_inputs_without_answer["attention_mask"].sum().item()

            answer_tokens = full_sample_tokens - no_answer_tokens

            result["labels"] = result_labels[:, 1:]
            result["labels"][:, :-answer_tokens] = -100
            result["labels"] = nn.functional.pad(result["labels"], (0, 1), value=-100)

        return result

    columns_to_keep = ["input_ids", "attention_mask", "special_embeddings_mask", "clothest_end_of_sentence_token_idx"]
    if args.with_labels_on_answer:
        columns_to_keep.append("labels")
    columns_to_remove = list(set(dataset.column_names) - set(columns_to_keep))

    dataset = dataset.map(process_dataset_item, num_proc=num_proc, remove_columns=columns_to_remove)

    dataset.save_to_disk(target_dir)

    print("dataset", dataset)
