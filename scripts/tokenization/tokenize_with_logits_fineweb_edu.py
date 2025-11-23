import torch
from datetime import datetime


from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_eos_tokens", type=int, default=8)
    parser.add_argument("--max_cummulative_nlogits_value", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_model_name = args.pretrained_model_name  # HuggingFaceTB/SmolLM2-1.7B / unsloth/Llama-3.2-1B
    max_length = args.max_length
    num_proc = args.num_proc
    pretrained_model_name_short = pretrained_model_name.split("/")[-1]

    print(f"pretrained_model_name_short: {pretrained_model_name_short}")

    suffix = ""
    if max_length != 1024:
        suffix = f"_max_length_{max_length}"

    logits_target_dir = f"./artifacts/data/fineweb_edu_tokenized_ligits_{pretrained_model_name_short}{suffix}"
    if args.num_shards > 1:
        logits_target_dir = f"{logits_target_dir}_shard_{args.shard_index}_of_{args.num_shards}"
    print("logits_target_dir:", logits_target_dir)

    target_dir = f"./artifacts/data/fineweb_edu_tokenized_with_logits_{pretrained_model_name_short}{suffix}"
    if args.num_shards > 1:
        target_dir = f"{target_dir}_shard_{args.shard_index}_of_{args.num_shards}"

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset = load_dataset(dataset_name, "sample-10BT", num_proc=16, split="train")

    tokenizer_class = type(AutoTokenizer.from_pretrained(pretrained_model_name)).__name__

    if tokenizer_class == "GPT2TokenizerFast":
        tokenizer_class = GPT2TokenizerFastEOS
    elif tokenizer_class == "PreTrainedTokenizerFast":
        tokenizer_class = PreTrainedTokenizerFastEOS
    elif tokenizer_class == "Qwen2TokenizerFast":
        tokenizer_class = Qwen2TokenizerFastEOS
    else:
        raise ValueError(f"Invalid tokenizer class: {tokenizer_class}")

    num_eos_tokens = args.num_eos_tokens
    eo_tokenizer = tokenizer_class.from_pretrained(
        pretrained_model_name,
        num_eos_tokens=num_eos_tokens,
        # gist_placement="max_cummulative_nlogits",
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # Only half of data!
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)
    print("shard len", len(dataset))
    # dataset = dataset.remove_columns(columns_to_remove)

    logits_dataset = Dataset.load_from_disk(logits_target_dir)
    assert len(logits_dataset) == len(dataset), "logits and base dataset lengths are not the same"

    print("add column token_logprobs", datetime.now())
    dataset = dataset.add_column("token_logprobs", logits_dataset["token_logprobs"])
    print("add column seq_length", datetime.now())
    dataset = dataset.add_column("seq_length", logits_dataset["seq_length"])

    print("shard len", len(dataset), datetime.now())

    columns_to_keep = ["input_ids", "attention_mask", "token_logprobs", "seq_length"]
    columns_to_remove = list(set(dataset.column_names) - set(columns_to_keep))

    max_cummulative_nlogits_value = args.max_cummulative_nlogits_value
    list_eos_tokens_ids = eo_tokenizer.end_of_sentence_token_ids

    def process_dataset_item(item):
        tokenized_inputs = tokenizer(
            item["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )

        new_input_ids = tokenized_inputs.input_ids.clone()
        special_embeddings_mask = torch.zeros_like(tokenized_inputs.input_ids, dtype=torch.bool)

        # TODO
        # my_clothest_end_of_sentence_token_idx = torch.zeros_like(tokenized_inputs.input_ids)

        seq_length = item["seq_length"]
        token_logprobs = item["token_logprobs"]

        start_position_i = new_input_ids.shape[1] - seq_length
        sum_nlogits = 0
        target_position_i = start_position_i
        for orig_position_i in range(start_position_i, tokenized_inputs.input_ids.shape[-1]):
            if target_position_i >= new_input_ids.shape[1]:
                break

            if orig_position_i != tokenized_inputs.input_ids.shape[-1] - 1:
                logit = token_logprobs[orig_position_i]
                sum_nlogits -= logit
                if sum_nlogits > max_cummulative_nlogits_value:
                    # print("insert new eos tokens")
                    sum_nlogits = 0
                    for gist_token in list_eos_tokens_ids:
                        new_input_ids[0, target_position_i] = gist_token
                        special_embeddings_mask[0, target_position_i] = True
                        target_position_i += 1
                        if target_position_i >= new_input_ids.shape[1]:
                            break

            if target_position_i >= new_input_ids.shape[1]:
                break
            new_input_ids[0, target_position_i] = tokenized_inputs.input_ids[0, orig_position_i]
            target_position_i += 1

        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            special_embeddings_mask, num_special_tokens=num_eos_tokens
        )

        return {
            "input_ids": new_input_ids[0],
            "attention_mask": tokenized_inputs.attention_mask[0],
            "special_embeddings_mask": special_embeddings_mask[0].numpy().tolist(),
            "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx[0].numpy().tolist(),
            "token_logprobs": token_logprobs,
            "seq_length": seq_length,
        }

    dataset = dataset.map(process_dataset_item, num_proc=num_proc, remove_columns=columns_to_remove)
    dataset.save_to_disk(target_dir)
    print("Saved to", target_dir)

    print(dataset[0])
