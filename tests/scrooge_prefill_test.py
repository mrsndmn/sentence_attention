import os

import pytest
import torch
from transformers import AutoTokenizer, DynamicCache

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import full_kv_scrooge_prefill, scrooge_prefill

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
            generated_output_text = "<end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>Jennifer is a young woman who marries a man named Mina Loris, a wealthy and distinguished scholar. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>She is initially excited about the marriage, but soon discovers that Loris is a controlling and jealous man who expects her to serve as his secretary. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>She begins to doubt his talent and his alleged magnum opus. <end_of_sentence_0><end_of_sentence_1><end_of_sentence_2><end_of_sentence_3>The Loris also becomes jealous of Will Rihanna, a cousin who is a close"
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

            scrooge_prefill_generated_output_text = tokenizer.decode(
                scrooge_generated_outputs[0, outputs["input_ids"].shape[1] :], skip_special_tokens=False
            )

            print(task_type, "scrooge prefill generated output text", scrooge_prefill_generated_output_text)

            assert (
                scrooge_prefill_generated_output_text == generated_output_text
            ), "scrooge prefill generated output text should be the same as the generated output text"


@pytest.mark.skip(reason="Skipping test_scrooge_prefill_only")
def _new_test_scrooge_prefill_kv_cache():

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
        print("forward_outputs", forward_outputs.hidden_states[-1])
        print("scrooge_forward_outputs", scrooge_forward_outputs.hidden_states[-1])
        breakpoint()


@pytest.mark.skip(reason="Skipping test_scrooge_prefill_only")
def test_scrooge_prefill_only():

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX,
        "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-6000/",
    )

    device = "cuda"

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode(
        # "Jennifer is an earnest intelligent woman who makes a serious error in judgment when she chooses to marry Mina Loris, a pompous scholar many years her senior. Jennifer hopes to be actively involved in his work, but he wants her to serve as a secretary. She comes to doubt both his talent and his alleged magnum opus. Furthermore, the controlling Loris becomes jealous when she develops a friendship with Will Rihanna, his idealistic cousin. Although disappointed, Jennifer remains committed to the marriage and tries to appease her husband. After Loris has a heart attack, Jennifer is clearly devoted to him, but he bars Rihanna from visiting, believing that his cousin will pursue Jennifer when he dies. Loris subsequently seeks her promise that she will follow his wishes even after his death.She delays answering but ultimately decides that she should agree to his request. However, he dies before she can tell him. Jennifer later discovers that his will contains a provision that calls for her to be disinherited if she marries Rihanna. Afraid of scandal, Jennifer and Rihanna initially stay apart. However, they ultimately fall in love and marry. Rihanna later becomes a politician, and, despite her sacrifices, Jennifer is content, because the growing good of the world is partly dependent on unhistoric acts.\n\nHere is summary of the provided text:\nJennifer is a young woman who marries", return_tensors="pt"
        "Jennifer is an earnest intelligent woman. She think",
        return_tensors="pt",
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

    full_forward_cache_init = DynamicCache()
    model(
        input_ids=input_ids.clone().to(device),
        attention_mask=attention_mask.clone().to(device),
        special_embeddings_mask=special_embeddings_mask.clone().to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone().to(device),
        past_key_values=full_forward_cache_init,
    )

    full_forward_cache = DynamicCache()
    model(
        input_ids=input_ids.clone().to(device),
        attention_mask=attention_mask.clone().to(device),
        special_embeddings_mask=special_embeddings_mask.clone().to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone().to(device),
        past_key_values=full_forward_cache,
    )

    indices = torch.nonzero(special_embeddings_mask.to(device), as_tuple=False)
    # indices has shape [num_true_positions, 2] where each row is [batch_idx, seq_idx]
    # We need to extract the sequence indices for selecting
    seq_indices = indices[:, 1]  # Extract sequence dimension indices

    for i in range(len(full_forward_cache.key_cache)):
        # Get the current cache tensor shape: [batch_size, num_heads, seq_len, head_dim]
        cache_shape = full_forward_cache.key_cache[i].shape
        batch_size, num_heads, seq_len, head_dim = cache_shape

        # Use torch.index_select to select only the special embedding positions
        # We need to select along the sequence dimension (dim=-2)
        full_forward_cache.key_cache[i] = torch.index_select(full_forward_cache.key_cache[i], dim=-2, index=seq_indices)
        full_forward_cache.value_cache[i] = torch.index_select(full_forward_cache.value_cache[i], dim=-2, index=seq_indices)

    print("full_forward_cache_processed", full_forward_cache.get_seq_length())

    outputs_sp = scrooge_prefill(
        model,
        input_ids.clone().to(device),
        attention_mask=attention_mask.clone().to(device),
        special_embeddings_mask=special_embeddings_mask.clone().to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone().to(device),
    )

    outputs = full_kv_scrooge_prefill(
        model,
        input_ids.clone().to(device),
        attention_mask=attention_mask.clone().to(device),
        special_embeddings_mask=special_embeddings_mask.clone().to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone().to(device),
    )

    # scrooge_full_kv_cache = outputs["past_key_values"]

    # for i in range(len(scrooge_full_kv_cache.key_cache)):
    #     keys_are_same = torch.allclose(full_forward_cache.key_cache[i], scrooge_full_kv_cache.key_cache[i], atol=1e-5)
    #     if not keys_are_same:
    #         print("key cache diff:", (full_forward_cache.key_cache[i] - scrooge_full_kv_cache.key_cache[i]).abs().mean(1).mean(-1))
    #     assert keys_are_same, "key cache should be the same"

    #     values_are_same = torch.allclose(full_forward_cache.value_cache[i], scrooge_full_kv_cache.value_cache[i], atol=1e-5)
    #     if not values_are_same:
    #         print("value cache diff:", (full_forward_cache.value_cache[i] - scrooge_full_kv_cache.value_cache[i]).abs().mean(1).mean(-1))
    #     assert values_are_same, "value cache should be the same"

    #     keys_are_same = torch.allclose(full_forward_cache.key_cache[i], outputs_sp['past_key_values'].key_cache[i], atol=1e-5)
    #     if not keys_are_same:
    #         print("key cache diff:", (full_forward_cache.key_cache[i] - outputs_sp['past_key_values'].key_cache[i]).abs().mean(1).mean(-1))
    #     assert keys_are_same, "key cache should be the same"

    #     values_are_same = torch.allclose(full_forward_cache.value_cache[i], outputs_sp['past_key_values'].value_cache[i], atol=1e-5)
    #     if not values_are_same:
    #         print("value cache diff:", (full_forward_cache.value_cache[i] - outputs_sp['past_key_values'].value_cache[i]).abs().mean(1).mean(-1))
    #     assert values_are_same, "value cache should be the same"

    assert (outputs["input_ids"] == outputs_sp["input_ids"]).all(), "input ids should be the same"
    assert (outputs["attention_mask"] == outputs_sp["attention_mask"]).all(), "attention mask should be the same"
    assert (
        outputs["special_embeddings_mask"] == outputs_sp["special_embeddings_mask"]
    ).all(), "special embeddings mask should be the same"
    assert (
        outputs["clothest_end_of_sentence_token_idx"] == outputs_sp["clothest_end_of_sentence_token_idx"]
    ).all(), "clothest end of sentence token idx should be the same"
    assert (outputs["cache_position"] == outputs_sp["cache_position"]).all(), "cache position should be the same"
    assert (
        outputs["past_key_values"].get_seq_length() == outputs_sp["past_key_values"].get_seq_length()
    ), "past key values should be the same"

    print("Scrooge prefill outputs kv seq_len", outputs["past_key_values"].get_seq_length())
    print("Input ids shape", outputs["input_ids"].shape)
    print("outputs[attention_mask]", outputs["attention_mask"].shape)
    print("outputs[cache_position]", outputs["cache_position"])

    max_new_tokens = 100
    breakpoint()

    generated_outputs = model.generate(
        outputs["input_ids"].to(device),
        attention_mask=outputs["attention_mask"].to(device),
        special_embeddings_mask=outputs["special_embeddings_mask"].to(device),
        clothest_end_of_sentence_token_idx=outputs["clothest_end_of_sentence_token_idx"].to(device),
        past_key_values=outputs["past_key_values"],
        cache_position=outputs["cache_position"].to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=False)
    print("Generated outputs", generated_output_text)

    breakpoint()
    # assert generated_output_text == "Russia is a country in Europe."

    no_kv_cache_generated_outputs = model.generate(
        input_ids.clone().to(device),
        attention_mask=attention_mask.clone().to(device),
        special_embeddings_mask=special_embeddings_mask.clone().to(device),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone().to(device),
        use_cache=False,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    no_kv_cache_generated_output_text = tokenizer.decode(no_kv_cache_generated_outputs[0], skip_special_tokens=False)
    print("No kv cache generated outputs", no_kv_cache_generated_output_text)

    breakpoint()


def test_kv_cache_forward():

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX,
        "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-10794/",
    )

    device = "cuda"

    torch_dtype = torch.bfloat16
    # torch_dtype = torch.float16
    # torch_dtype = torch.float32

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch_dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode(
        # "Jennifer is an earnest intelligent woman who makes a serious error in judgment when she chooses to marry Mina Loris, a pompous scholar many years her senior. Jennifer hopes to be actively involved in his work, but he wants her to serve as a secretary. She comes to doubt both his talent and his alleged magnum opus. Furthermore, the controlling Loris becomes jealous when she develops a friendship with Will Rihanna, his idealistic cousin. Although disappointed, Jennifer remains committed to the marriage and tries to appease her husband. After Loris has a heart attack, Jennifer is clearly devoted to him, but he bars Rihanna from visiting, believing that his cousin will pursue Jennifer when he dies. Loris subsequently seeks her promise that she will follow his wishes even after his death.She delays answering but ultimately decides that she should agree to his request. However, he dies before she can tell him. Jennifer later discovers that his will contains a provision that calls for her to be disinherited if she marries Rihanna. Afraid of scandal, Jennifer and Rihanna initially stay apart. However, they ultimately fall in love and marry. Rihanna later becomes a politician, and, despite her sacrifices, Jennifer is content, because the growing good of the world is partly dependent on unhistoric acts.\n\nHere is summary of the provided text:\nJennifer is a young woman who marries", return_tensors="pt"
        "Jennifer is an earnest intelligent woman. She think",
        return_tensors="pt",
    )
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids, device=device)

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

    full_forward_cache_init = DynamicCache()
    init_out = model(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        special_embeddings_mask=special_embeddings_mask.clone(),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone(),
        past_key_values=full_forward_cache_init,
    )

    eos_only_forward_cache = DynamicCache()
    eos_only_out = model(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        special_embeddings_mask=special_embeddings_mask.clone(),
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx.clone(),
        past_key_values=eos_only_forward_cache,
    )

    # indices = torch.nonzero(special_embeddings_mask, as_tuple=False)
    # indices has shape [num_true_positions, 2] where each row is [batch_idx, seq_idx]
    # We need to extract the sequence indices for selecting
    # seq_indices = indices[:, 1]  # Extract sequence dimension indices

    num_last_kv_cache_tokens = 6

    for i in range(len(eos_only_forward_cache.key_cache)):
        # Get the current cache tensor shape: [batch_size, num_heads, seq_len, head_dim]
        cache_shape = eos_only_forward_cache.key_cache[i].shape
        batch_size, num_heads, seq_len, head_dim = cache_shape

        # Use torch.index_select to select only the special embedding positions
        # We need to select along the sequence dimension (dim=-2)
        eos_only_forward_cache.key_cache[i] = eos_only_forward_cache.key_cache[i][:, :, -num_last_kv_cache_tokens:, :]
        eos_only_forward_cache.value_cache[i] = eos_only_forward_cache.value_cache[i][:, :, -num_last_kv_cache_tokens:, :]

    for _ in range(len(model.config.end_of_sentence_token_ids)):
        for i in range(len(full_forward_cache_init.key_cache)):
            assert torch.allclose(
                full_forward_cache_init.key_cache[i][:, :, -num_last_kv_cache_tokens:, :], eos_only_forward_cache.key_cache[i]
            ), "key cache should be the same"
            assert torch.allclose(
                full_forward_cache_init.value_cache[i][:, :, -num_last_kv_cache_tokens:, :],
                eos_only_forward_cache.value_cache[i],
            ), "value cache should be the same"

    # breakpoint()
    max_new_tokens = 50

    init_input_ids = init_out.logits[:, -1:].argmax(dim=-1)
    eos_only_input_ids = eos_only_out.logits[:, -1:].argmax(dim=-1)

    init_generated_tokens = [init_input_ids.item()]
    eos_only_generated_tokens = [eos_only_input_ids.item()]

    init_attention_mask_continuation = [1]
    init_special_embeddings_mask_continuation = [0]

    eos_only_attention_mask_continuation = [1]
    eos_only_special_embeddings_mask_continuation = [0]

    for new_token_id in range(max_new_tokens):

        print("full_forward_cache_init", full_forward_cache_init.get_seq_length())
        full_attention_mask = torch.cat(
            [attention_mask, torch.tensor([init_attention_mask_continuation], device=device, dtype=torch.long)], dim=-1
        )
        full_special_embeddings_mask = torch.cat(
            [
                special_embeddings_mask,
                torch.tensor([init_special_embeddings_mask_continuation], device=device, dtype=torch.long),
            ],
            dim=-1,
        )
        full_clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            full_special_embeddings_mask,
            num_special_tokens=len(model.config.end_of_sentence_token_ids),
        )

        init_out = model(
            input_ids=init_input_ids,
            attention_mask=full_attention_mask,
            special_embeddings_mask=full_special_embeddings_mask,
            clothest_end_of_sentence_token_idx=full_clothest_end_of_sentence_token_idx,
            past_key_values=full_forward_cache_init,
            cache_position=torch.tensor([input_ids.shape[1] + 1], device=input_ids.device, dtype=torch.long),
            output_hidden_states=True,
        )

        init_input_ids = None
        if init_generated_tokens[-1] in model.config.end_of_sentence_token_ids:
            prev_token_is_eos_idx = model.config.end_of_sentence_token_ids.index(init_generated_tokens[-1])
            if prev_token_is_eos_idx < len(model.config.end_of_sentence_token_ids) - 1:
                init_input_ids = torch.tensor(
                    [[model.config.end_of_sentence_token_ids[prev_token_is_eos_idx + 1]]],
                    device=input_ids.device,
                    dtype=torch.long,
                )

        if init_input_ids is None:
            init_input_ids = init_out.logits[:, -1:].argmax(dim=-1)

        init_next_token = init_input_ids.item()
        init_generated_tokens.append(init_next_token)

        init_attention_mask_continuation.append(1)
        if init_next_token in model.config.end_of_sentence_token_ids:
            init_special_embeddings_mask_continuation.append(1)
        else:
            init_special_embeddings_mask_continuation.append(0)

        eos_only_attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1] + eos_only_attention_mask_continuation], device=device, dtype=torch.long
        )
        eos_only_special_embeddings_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0] + eos_only_special_embeddings_mask_continuation], device=device, dtype=torch.long
        )
        eos_only_clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            eos_only_special_embeddings_mask,
            num_special_tokens=len(model.config.end_of_sentence_token_ids),
        )

        print("eos_only_forward_cache", eos_only_forward_cache.get_seq_length())

        eos_only_out = model(
            input_ids=eos_only_input_ids,
            attention_mask=eos_only_attention_mask,
            special_embeddings_mask=eos_only_special_embeddings_mask,
            clothest_end_of_sentence_token_idx=eos_only_clothest_end_of_sentence_token_idx,
            past_key_values=eos_only_forward_cache,
            cache_position=torch.tensor([input_ids.shape[1] + 1], device=input_ids.device, dtype=torch.long),
            output_hidden_states=True,
        )

        eos_only_input_ids = None

        if eos_only_generated_tokens[-1] in model.config.end_of_sentence_token_ids:
            prev_token_is_eos_idx = model.config.end_of_sentence_token_ids.index(eos_only_generated_tokens[-1])
            if prev_token_is_eos_idx < len(model.config.end_of_sentence_token_ids) - 1:
                eos_only_input_ids = torch.tensor(
                    [[model.config.end_of_sentence_token_ids[prev_token_is_eos_idx + 1]]],
                    device=input_ids.device,
                    dtype=torch.long,
                )

        if eos_only_input_ids is None:
            eos_only_input_ids = eos_only_out.logits[:, -1:].argmax(dim=-1)

        eos_only_next_token = eos_only_input_ids.item()
        eos_only_generated_tokens.append(eos_only_next_token)

        eos_only_attention_mask_continuation.append(1)
        if eos_only_next_token in model.config.end_of_sentence_token_ids:
            eos_only_special_embeddings_mask_continuation.append(1)
        else:
            eos_only_special_embeddings_mask_continuation.append(0)

        # for hs_i in range(len(init_out.hidden_states)):
        #     print(
        #         "HS i",
        #         hs_i,
        #         "diff",
        #         (init_out.hidden_states[hs_i] - eos_only_out.hidden_states[hs_i]).abs().mean(1).mean(-1),
        #     )

        print("Logits max diff", (init_out.logits - eos_only_out.logits).abs().max())
        # print(
        #     "Logits max",
        #     init_out.logits[:, -1].argsort(dim=-1, descending=True)[:, :5],
        #     eos_only_out.logits[:, -1].argsort(dim=-1, descending=True)[:, :5],
        # )
        if not (
            init_out.logits[:, -1].argsort(dim=-1, descending=True)[:, :5]
            == eos_only_out.logits[:, -1].argsort(dim=-1, descending=True)[:, :5]
        ).all():
            print(f"{new_token_id} WARNING! ⚠️ Logist are affected! Order of tokens is changed!")
            # breakpoint()
        else:
            print(f"{new_token_id} ✅ Logist are the same")

    init_generated = tokenizer.decode(init_generated_tokens, skip_special_tokens=False)
    eos_only_generated = tokenizer.decode(eos_only_generated_tokens, skip_special_tokens=False)

    print("init_generated    ", init_generated)
    print("eos_only_generated", eos_only_generated)

    assert init_generated == eos_only_generated, "generated tokens should be the same"

    breakpoint()
