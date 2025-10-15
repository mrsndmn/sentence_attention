import random

import torch
from datasets import Dataset
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from wonderwords import RandomWord


def generate_random_sample(num_examples=200, random_word=None):

    if random_word is None:
        random_word = RandomWord()

    keys_values = {}
    for _ in range(num_examples):
        word = "-".join(random_word.random_words(3))
        while word in keys_values:
            word = "-".join(random_word.random_words(3))
        keys_values[word] = random.randint(1, 1000000)

    query_key = random.choice(list(keys_values.keys()))
    query_answer = keys_values[query_key]

    haystack_samples_list = []
    keys_values_keys = list(keys_values.keys())
    random.shuffle(keys_values_keys)
    for key in keys_values_keys:
        value = keys_values[key]

        example = f"One of the special magic numbers for {key} is: {value}.\n"
        haystack_samples_list.append(example)

    haystack_samples = "\n".join(haystack_samples_list)
    full_sample = f"""
A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.
{haystack_samples}
The special magic number for {query_key} mentioned in the provided text is {query_answer}.
"""

    return full_sample


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--with_eos_token", action="store_true")
    parser.add_argument("--num_eos_tokens", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10000)
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

    target_dir = f"./artifacts/data/synthetic_niah_tokenized_{pretrained_model_name_short}{suffix}"

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

    random_word = RandomWord()
    s = generate_random_sample(random_word=random_word)

    generated_samples = []
    for _ in range(args.num_samples):
        generated_samples.append(generate_random_sample(random_word=random_word))

    dataset = Dataset.from_dict({"text": generated_samples})

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
