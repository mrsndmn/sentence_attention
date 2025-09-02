import os

import pytest
import torch
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill
from transformers import AutoTokenizer

# from transformers import LlamaForCausalLM

ARTIFACTS_PREFIX = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/"


@pytest.mark.skip(reason="Skipping test_generate_country")
def test_generate_country():

    checkpoint = os.path.join(ARTIFACTS_PREFIX, "./experiments/eos_1/sentence_Llama-3.2-1B_ft_full_L1DB3Z21/checkpoint-1349/")

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode(
        "Great Brittain is a country in Europe. France is a country in Europe. Russia is", return_tensors="pt"
    )

    attention_mask = torch.ones_like(input_ids)

    generated_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        use_cache=False,
    )

    generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    print("Generated outputs", generated_output_text)

    assert generated_output_text.endswith("Russia is a country in Europe.")


def test_generate_number():

    checkpoint = os.path.join(ARTIFACTS_PREFIX, "./experiments/eos_1/sentence_Llama-3.2-1B_ft_full_L1DB3Z21/checkpoint-1349/")
    # checkpoint_4eos_tokens = os.path.join(ARTIFACTS_PREFIX, "./experiments_in_progress/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR/checkpoint-250")

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    input_ids = tokenizer.encode(
        # No instruction - Fails
        # "The special magic numbers for uninterested-cashier is: 2368710. The special magic number for uninterested-cashier mentioned in the provided text is",
        # Start from instruction - ok
        "Remember special magic number for uninterested-cashier. The special magic numbers for uninterested-cashier is: 2368710. The special magic number for uninterested-cashier mentioned in the provided text is",
        # Start from instruction, add noise - Fails
        # "Remember special magic number for uninterested-cashier. The special magic numbers for uninterested-cashier is: 2368710. The spechal number for lazy-cat is: 55822300. The special magic number for uninterested-cashier mentioned in the provided text is",
        return_tensors="pt",
    )

    attention_mask = torch.ones_like(input_ids).to(device)

    generated_outputs = model.generate(
        input_ids.to(device),
        attention_mask=attention_mask,
        max_new_tokens=5,
        use_cache=False,
    )

    generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    print("Generated outputs", generated_output_text)

    generated_output_text = generated_output_text.strip().removesuffix(".")

    assert generated_output_text.endswith("2368710")


def test_scrooge_prefill():

    checkpoint = os.path.join(ARTIFACTS_PREFIX, "./experiments/eos_1/sentence_Llama-3.2-1B_ft_full_L1DB3Z21/checkpoint-1349/")

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode(
        "Great Brittain is a country in Europe. France is a country in Europe. Russia is", return_tensors="pt"
    )

    attention_mask = torch.ones_like(input_ids)

    special_embeddings_mask = torch.zeros_like(attention_mask)
    if model.config.end_of_sentence_token_ids is not None:
        total_eos_tokens = 0
        for end_of_sentence_token_id in model.config.end_of_sentence_token_ids:
            special_embeddings_mask[input_ids == end_of_sentence_token_id] = 1
            total_eos_tokens += (input_ids == end_of_sentence_token_id).sum().item()
        print("number of end of sentence tokens", total_eos_tokens)

    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask,
        num_special_tokens=len(model.config.end_of_sentence_token_ids),
    )

    print("input_ids", input_ids.shape)
    print("clothest_end_of_sentence_token_idx", clothest_end_of_sentence_token_idx)

    outputs = scrooge_prefill(
        model,
        input_ids,
        attention_mask=attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
    )

    print("Scrooge prefill outputs kv seq_len", outputs["past_key_values"].get_seq_length())
    print("Input ids shape", outputs["input_ids"].shape)
    print("outputs[attention_mask]", outputs["attention_mask"].shape)
    print("outputs[cache_position]", outputs["cache_position"])

    generated_outputs = model.generate(
        outputs["input_ids"],
        attention_mask=outputs["attention_mask"],
        special_embeddings_mask=outputs["special_embeddings_mask"],
        clothest_end_of_sentence_token_idx=outputs["clothest_end_of_sentence_token_idx"],
        past_key_values=outputs["past_key_values"],
        cache_position=outputs["cache_position"],
        max_new_tokens=5,
    )

    generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    print("Generated outputs", generated_output_text)

    assert generated_output_text == "Russia is a country in Europe."

    no_kv_cache_generated_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        use_cache=False,
        max_new_tokens=5,
    )

    no_kv_cache_generated_output_text = tokenizer.decode(no_kv_cache_generated_outputs[0], skip_special_tokens=False)
    print("No kv cache generated outputs", no_kv_cache_generated_output_text)
