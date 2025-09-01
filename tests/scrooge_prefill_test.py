import os

import torch
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill
from transformers import AutoTokenizer

ARTIFACTS_PREFIX = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/"


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
