import os

import pytest
import torch
from transformers import AutoTokenizer

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill

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
    # checkpoint = os.path.join(
    #     ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR/checkpoint-1349"
    # )
    # checkpoint = os.path.join(
    #     ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_bos_token_full_num_eos_tokens_4_OPOKS8O7/checkpoint-336"
    # )
    # checkpoint = os.path.join(
    #     ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft2_full_num_eos_tokens_4_MV7M599S/checkpoint-10794/"
    # )
    checkpoint = os.path.join(
        ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-2000/"
    )

    save_maps = False

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    model.to(device)

    texts = [
        ("no_instruction", "The special magic numbers for uninterested-cashier is: 2368710. "),
        (
            "instruction",
            "Remember special magic number for uninterested-cashier. The special magic numbers for uninterested-cashier is: 2368710. ",
        ),
        (
            "instruction-noise",
            "Remember special magic number for uninterested-cashier. The special number for fat-squirrel is: 8244459. The special number for lazy-cat is: 55822300. The special magic numbers for uninterested-cashier is: 2368710. The special number for mega-boomber is: 2341887. The special number for jagger-ragger is: 555333110. ",
        ),
    ]

    failed = []

    print("Model config flexible_eos_tokens", model.config.flexible_eos_tokens)
    print("Model config ft_with_bos_token", model.config.ft_with_bos_token)

    with torch.no_grad():

        for task_type, task_prefix in texts:
            input_ids = tokenizer.encode(
                task_prefix + "The special magic number for uninterested-cashier mentioned in the provided text is",
                return_tensors="pt",
            )
            input_ids = input_ids.to(device)

            attention_mask = torch.ones_like(input_ids).to(device)

            if save_maps:
                fwd_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                # breakpoint()
                import matplotlib.pyplot as plt

                num_layers = len(fwd_outputs.attentions)
                plt.gcf().set_size_inches(50, 50)
                for layer_i, layer_attentions in enumerate(fwd_outputs.attentions):
                    layer_attentions_cpu = layer_attentions.float().cpu()

                    num_heads = layer_attentions.shape[1]
                    for head_num in range(num_heads):
                        plt.subplot(num_layers, layer_attentions.shape[1], num_heads * layer_i + head_num + 1)

                        plt.imshow(layer_attentions_cpu[0, head_num].detach().numpy())

                figure_path = f"/tmp/with_mask_sentence_attention_figure_{task_type}.png"
                plt.tight_layout()
                plt.savefig(figure_path)
                print("Saved attention maps:", figure_path)
                plt.clf()

                plt.figure(figsize=(60, 5))
                for layer_i, layer_attentions in enumerate(fwd_outputs.attentions):
                    layer_attentions_cpu = layer_attentions.float().cpu()
                    layer_attentions_cpu_mean = layer_attentions_cpu.mean(dim=1)
                    plt.subplot(1, len(fwd_outputs.attentions), layer_i + 1)
                    plt.imshow(layer_attentions_cpu_mean[0].detach().numpy())

                plt.tight_layout()
                plt.savefig(f"/tmp/with_mask_sentence_attention_figure_mean_{task_type}.png")
                print("Saved attention maps mean:", f"/tmp/with_mask_sentence_attention_figure_mean_{task_type}.png")
                plt.clf()

            generated_outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                use_cache=False,
            )

            generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
            print("Generated outputs", generated_output_text)

            generated_output_text = generated_output_text.strip().removesuffix(".")

            if generated_output_text.endswith("2368710"):
                print(f"\033[92mTest passed for {task_type}\033[0m")
            else:
                print(f"\033[91mTest failed for {task_type}\033[0m")
                failed.append(task_type)

        # assert len(failed) == 0, f"Failed tests: {failed}"


def test_scrooge_prefill():

    checkpoint = os.path.join(ARTIFACTS_PREFIX, "./experiments/eos_1/sentence_Llama-3.2-1B_ft_full_L1DB3Z21/checkpoint-1349/")

    device = "cuda"

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint).to(device)
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
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        special_embeddings_mask=special_embeddings_mask.to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.to(device),
    )

    print("Scrooge prefill outputs kv seq_len", outputs["past_key_values"].get_seq_length())
    print("Input ids shape", outputs["input_ids"].shape)
    print("outputs[attention_mask]", outputs["attention_mask"].shape)
    print("outputs[cache_position]", outputs["cache_position"])

    generated_outputs = model.generate(
        outputs["input_ids"].to(device),
        attention_mask=outputs["attention_mask"].to(device),
        special_embeddings_mask=outputs["special_embeddings_mask"].to(device),
        clothest_end_of_sentence_token_idx=outputs["clothest_end_of_sentence_token_idx"].to(device),
        past_key_values=outputs["past_key_values"],
        cache_position=outputs["cache_position"].to(device),
        max_new_tokens=5,
    )

    generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    print("Generated outputs", generated_output_text)

    assert generated_output_text == "Russia is a country in Europe."

    no_kv_cache_generated_outputs = model.generate(
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        special_embeddings_mask=special_embeddings_mask.to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.to(device),
        use_cache=False,
        max_new_tokens=5,
    )

    no_kv_cache_generated_output_text = tokenizer.decode(no_kv_cache_generated_outputs[0], skip_special_tokens=False)
    print("No kv cache generated outputs", no_kv_cache_generated_output_text)
