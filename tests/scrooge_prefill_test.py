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


def test_generate_summary():

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-4000/"
    )

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    model.to(device)

    base_story_text = "Jennifer is an earnest intelligent woman who makes a serious error in judgment when she chooses to marry Mina Loris, a pompous scholar many years her senior. Jennifer hopes to be actively involved in his work, but he wants her to serve as a secretary. She comes to doubt both his talent and his alleged magnum opus. Furthermore, the controlling Loris becomes jealous when she develops a friendship with Will Rihanna, his idealistic cousin. Although disappointed, Jennifer remains committed to the marriage and tries to appease her husband. After Loris has a heart attack, Jennifer is clearly devoted to him, but he bars Rihanna from visiting, believing that his cousin will pursue Jennifer when he dies. Loris subsequently seeks her promise that she will follow his wishes even after his death.She delays answering but ultimately decides that she should agree to his request. However, he dies before she can tell him. Jennifer later discovers that his will contains a provision that calls for her to be disinherited if she marries Rihanna. Afraid of scandal, Jennifer and Rihanna initially stay apart. However, they ultimately fall in love and marry. Rihanna later becomes a politician, and, despite her sacrifices, Jennifer is content, because the growing good of the world is partly dependent on unhistoric acts."

    texts = [
        ("instruction_last", base_story_text + "\n\nHere is the summary of previous text: "),
        (
            "instruction_fitst",
            "You are a summary writer. You are given a story text and you need to write a summary of the story. \n\nText:.\n"
            + base_story_text
            + "\n\nHere is the summary of previous text: ",
        ),
    ]

    print("Model config flexible_eos_tokens", model.config.flexible_eos_tokens)
    print("Model config ft_with_bos_token", model.config.ft_with_bos_token)

    with torch.no_grad():

        max_new_tokens = 100

        for task_type, task_prefix in texts:
            input_ids = tokenizer.encode(
                task_prefix,
                return_tensors="pt",
            )
            input_ids = input_ids.to(device)

            attention_mask = torch.ones_like(input_ids).to(device)

            # generated_outputs = model.generate(
            #     input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_new_tokens,
            #     use_cache=False,
            # )
            generated_output_text = '<end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>Jennifer is a young woman who marries a man named Mina Loris, a wealthy and distinguished scholar. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>She is initially excited about the marriage, but soon discovers that Loris is a controlling and jealous man who expects her to serve as his secretary. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>She begins to doubt his talent and his alleged magnum opus. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>The Loris also becomes jealous of Will Rihanna, a cousin who is a close'
            # generated_output_text = tokenizer.decode(generated_outputs[0, input_ids.shape[1] :], skip_special_tokens=False)

            print(task_type, "generated outputs", generated_output_text)

            # scrooge prefill
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

            outputs = scrooge_prefill(
                model,
                input_ids,
                attention_mask=attention_mask,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            )


            inputs = {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
                "special_embeddings_mask": outputs["special_embeddings_mask"],
                "clothest_end_of_sentence_token_idx": outputs["clothest_end_of_sentence_token_idx"],
                "past_key_values": outputs["past_key_values"],
                "cache_position": outputs["cache_position"],
            }

            # special_embeddings_mask = torch.zeros_like(inputs['attention_mask'])
            # if model.config.end_of_sentence_token_ids is not None:
            #     for end_of_sentence_token_id in model.config.end_of_sentence_token_ids:
            #         special_embeddings_mask[inputs['input_ids'] == end_of_sentence_token_id] = 1

            # clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            #     special_embeddings_mask,
            #     num_special_tokens=len(model.config.end_of_sentence_token_ids),
            # )

            # inputs["special_embeddings_mask"] = special_embeddings_mask
            # inputs["clothest_end_of_sentence_token_idx"] = clothest_end_of_sentence_token_idx

            scrooge_generated_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=False,
            )

            scrooge_prefill_generated_output_text = tokenizer.decode(scrooge_generated_outputs[0, outputs["input_ids"].shape[1] :], skip_special_tokens=False)

            print(task_type, "scrooge prefill generated output text", scrooge_prefill_generated_output_text)

            assert scrooge_prefill_generated_output_text == generated_output_text, "scrooge prefill generated output text should be the same as the generated output text"

def test_scrooge_prefill_kv_cache():

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-4000/"
    )

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    model.to(device)

    base_story_text = "Jennifer is an earnest intelligent woman who makes a serious error in judgment when she chooses to marry Mina Loris, a pompous scholar many years her senior. Jennifer hopes to be actively involved in his work, but he wants her to serve as a secretary. She comes to doubt both his talent and his alleged magnum opus. Furthermore, the controlling Loris becomes jealous when she develops a friendship with Will Rihanna, his idealistic cousin. Although disappointed, Jennifer remains committed to the marriage and tries to appease her husband. After Loris has a heart attack, Jennifer is clearly devoted to him, but he bars Rihanna from visiting, believing that his cousin will pursue Jennifer when he dies. Loris subsequently seeks her promise that she will follow his wishes even after his death.She delays answering but ultimately decides that she should agree to his request. However, he dies before she can tell him. Jennifer later discovers that his will contains a provision that calls for her to be disinherited if she marries Rihanna. Afraid of scandal, Jennifer and Rihanna initially stay apart. However, they ultimately fall in love and marry. Rihanna later becomes a politician, and, despite her sacrifices, Jennifer is content, because the growing good of the world is partly dependent on unhistoric acts."

    with torch.no_grad():

        input_ids = tokenizer.encode(
            base_story_text,
            return_tensors="pt",
        )
        input_ids = input_ids.to(device)

        attention_mask = torch.ones_like(input_ids).to(device)

        past_key_values = DynamicCache()

        forward_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )

        # scrooge prefill
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

        outputs = scrooge_prefill(
            model,
            input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        )


        inputs = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "special_embeddings_mask": outputs["special_embeddings_mask"],
            "clothest_end_of_sentence_token_idx": outputs["clothest_end_of_sentence_token_idx"],
            "past_key_values": outputs["past_key_values"],
            "cache_position": outputs["cache_position"],
            "use_cache": True,
        }

        # special_embeddings_mask = torch.zeros_like(inputs['attention_mask'])
        # if model.config.end_of_sentence_token_ids is not None:
        #     for end_of_sentence_token_id in model.config.end_of_sentence_token_ids:
        #         special_embeddings_mask[inputs['input_ids'] == end_of_sentence_token_id] = 1

        # clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        #     special_embeddings_mask,
        #     num_special_tokens=len(model.config.end_of_sentence_token_ids),
        # )

        # inputs["special_embeddings_mask"] = special_embeddings_mask
        # inputs["clothest_end_of_sentence_token_idx"] = clothest_end_of_sentence_token_idx

        scrooge_forward_outputs = model(
            **inputs,
        )

        # TODO check hidden states are close!
        breakpoint()


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
