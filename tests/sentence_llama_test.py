import os

import torch
from sentence_attention.artifacts.experiments import ARTIFACTS_PREFIX
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    SentenceLlamaModel,
    special_token_mask_to_clothest_token_idx_slow,
)
from transformers import AutoTokenizer


def test_sentence_attention_4d_mask():
    torch.manual_seed(0)

    batch_size = 1
    num_heads = 1
    seq_len = 8
    head_dim = 8

    # Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # 2D attention mask (no padding)
    attention_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mark a few EOS tokens in the sequence
    special_embeddings_mask = torch.zeros_like(attention_mask_2d)
    special_positions = [3, 4]
    for pos in special_positions:
        special_embeddings_mask[:, pos] = 1

    num_special_tokens = 2
    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask, num_special_tokens=num_special_tokens
    )

    cache_position = torch.arange(seq_len)
    causal_mask_4d = SentenceLlamaModel._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
        attention_mask=attention_mask_2d,
        sequence_length=seq_len,
        target_length=seq_len,
        dtype=query.dtype,
        device=query.device,
        cache_position=cache_position,
        batch_size=batch_size,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    expected_mask = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 1],
            ]
        ]
    )

    assert ((causal_mask_4d == 0) == expected_mask).all()


@torch.no_grad()
def test_sentence_llama_model_generate_with_eos_token():

    device = "cuda"

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX, "experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-10794/"
    )
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")

    input_text = "Russia - Moscow. France - Paris. Germany - Berlin. Italy - "

    # TODO flex attention differs if there is any EOS token!
    input_text = input_text.replace(".", " ")

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    end_of_sentence_token_id = tokenizer.convert_tokens_to_ids("<end_of_sentence_0>")

    if "." in input_text:
        assert (input_ids == end_of_sentence_token_id).sum().item() == 3

    outputs_logits = []
    tokens_orders = []
    outputs_hidden_states = []

    model.eval()

    for attn_impl in ["sentence_attention", "sentence_attention_flex"]:
        model.config._attn_implementation = attn_impl

        print("input_ids", input_ids)
        output = model.forward(input_ids, output_hidden_states=True)

        outputs_logits.append(output.logits)

        tokens_order = torch.argsort(output.logits[:, -1], dim=-1, descending=True)[:, :1000].cpu().numpy().tolist()
        tokens_orders.append(tokens_order)
        print(attn_impl, "logits", tokens_order)

        outputs_hidden_states.append(output.hidden_states)

        # print("attn_impl", attn_impl, "output", tokenizer.decode(output[0], skip_special_tokens=False))

    assert tokens_orders[0] == tokens_orders[1], "tokens orders are not equal"
    assert torch.allclose(outputs_logits[0][:, -1], outputs_logits[1][:, -1], atol=1e-4), "logits are not equal"

    diffs_hidden_states = []
    for h_i in range(len(outputs_hidden_states[0])):
        diffs_hidden_states.append((outputs_hidden_states[0][h_i] - outputs_hidden_states[1][h_i]).norm(2, dim=-1))

    for h_i in range(len(outputs_hidden_states[0])):
        assert torch.allclose(
            outputs_hidden_states[0][h_i], outputs_hidden_states[1][h_i], atol=1e-4
        ), f"hidden_states[{h_i}] are not equal"


def test_sentence_llama_model_generate_with_eos_token_and_attention_mask_pad():

    device = "cuda"

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_ids = [tokenizer.convert_tokens_to_ids("<end_of_sentence_0>")]

    # model.config._attn_implementation = "sentence_attention"
    model.config._attn_implementation = "sentence_attention_flex"

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    input_ids = input_ids.to(device)
    assert (input_ids == model.config.end_of_sentence_token_ids[0]).sum().item() == 3

    seq_len = input_ids.shape[1]

    # print("input_ids", input_ids)
    output1 = model(
        input_ids,
        labels=input_ids,
        use_cache=False,
        output_hidden_states=True,
    )

    # inputs_ids_padded = torch.cat([torch.zeros_like(input_ids), input_ids], dim=-1)
    inputs_ids_padded = torch.cat([input_ids, torch.zeros_like(input_ids)], dim=-1)
    # attention_mask_padded = torch.cat([torch.zeros_like(input_ids), torch.ones_like(input_ids)], dim=-1)
    attention_mask_padded = torch.cat([torch.ones_like(input_ids), torch.zeros_like(input_ids)], dim=-1)
    # labels_padded = torch.cat([torch.ones_like(input_ids) * -100, input_ids], dim=-1)
    labels_padded = torch.cat([input_ids, torch.ones_like(input_ids) * -100], dim=-1)

    output2 = model(
        inputs_ids_padded,
        labels=labels_padded,
        attention_mask=attention_mask_padded,
        use_cache=False,
        output_hidden_states=True,
    )

    for i in range(len(output1.hidden_states)):
        assert torch.allclose(
            output1.hidden_states[i], output2.hidden_states[i][:, :seq_len], atol=1e-2
        ), f"hidden_states[{i}] are not equal"

    assert torch.allclose(
        output1.loss, output2.loss, atol=1e-3
    ), f"output losses are not equal, l1={output1.loss}, l2={output2.loss}"


def test_sentence_llama_model_generate_with_eos_token_and_attention_mask_partial_logits():

    device = "cuda"

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_ids = [tokenizer.convert_tokens_to_ids("<end_of_sentence_0>")]

    model.config._attn_implementation = "sentence_attention_flex"

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    input_ids = input_ids.to(device)
    assert (input_ids == model.config.end_of_sentence_token_ids[0]).sum().item() == 3

    seq_len = input_ids.shape[1]

    model.eval()

    # print("input_ids", input_ids)
    output1 = model(
        input_ids,
        use_cache=False,
        output_hidden_states=True,
    )

    output2 = model(
        input_ids[:, : seq_len // 2],
        use_cache=False,
        output_hidden_states=True,
    )

    tokens_1 = output1.logits[:, : seq_len // 2, :].argmax(dim=-1)
    tokens_2 = output2.logits.argmax(dim=-1)

    print("tokens_1", tokens_1)
    print("tokens_2", tokens_2)

    assert (tokens_1 == tokens_2).all(), "tokens are not equal"

    logits_diff = (output1.logits[:, : seq_len // 2, :] - output2.logits).norm()
    assert logits_diff < 0.04, "logits diff is low"
    assert torch.allclose(output1.logits[:, : seq_len // 2, :], output2.logits, atol=1e-2), "logits are not equal"


@torch.no_grad()
def test_generate_flex_attention():

    checkpoint = os.path.join(
        ARTIFACTS_PREFIX, "./experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-10794/"
    )

    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model.to(device)

    base_story_text = "Jennifer is an earnest intelligent woman who makes a serious error in judgment when she chooses to marry Mina Loris, a pompous scholar many years her senior. Jennifer hopes to be actively involved in his work, but he wants her to serve as a secretary. She comes to doubt both his talent and his alleged magnum opus. Furthermore, the controlling Loris becomes jealous when she develops a friendship with Will Rihanna, his idealistic cousin. Although disappointed, Jennifer remains committed to the marriage and tries to appease her husband. After Loris has a heart attack, Jennifer is clearly devoted to him, but he bars Rihanna from visiting, believing that his cousin will pursue Jennifer when he dies. Loris subsequently seeks her promise that she will follow his wishes even after his death.She delays answering but ultimately decides that she should agree to his request. However, he dies before she can tell him. Jennifer later discovers that his will contains a provision that calls for her to be disinherited if she marries Rihanna. Afraid of scandal, Jennifer and Rihanna initially stay apart. However, they ultimately fall in love and marry. Rihanna later becomes a politician, and, despite her sacrifices, Jennifer is content, because the growing good of the world is partly dependent on unhistoric acts."

    texts = [
        ("instruction_last", base_story_text + "\n\nHere is the summary of previous text: "),
        # (
        #     "instruction_fitst",
        #     "You are a summary writer. You are given a story text and you need to write a summary of the story. \n\nText:.\n"
        #     + base_story_text
        #     + "\n\nHere is the summary of previous text: ",
        # ),
    ]

    print("Model config flexible_eos_tokens", model.config.flexible_eos_tokens)
    print("Model config ft_with_bos_token", model.config.ft_with_bos_token)

    max_new_tokens = 10

    for sattn_impl in ["sentence_attention_flex", "sentence_attention"]:
        for task_type, task_prefix in texts:
            model.config._attn_implementation = sattn_impl
            input_ids = tokenizer.encode(
                task_prefix,
                return_tensors="pt",
            )
            input_ids = input_ids.to(device)

            attention_mask = torch.ones_like(input_ids).to(device)

            scrooge_generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_scores=False,
            )

            # scrooge_prefill_generated_output_text = tokenizer.decode(scrooge_generated_outputs[0,], skip_special_tokens=False)
            scrooge_prefill_generated_output_text = tokenizer.decode(scrooge_generated_outputs[0,], skip_special_tokens=True)

            print(sattn_impl, task_type, "\n", scrooge_generated_outputs[0].cpu().numpy().tolist())

            print(sattn_impl, task_type, "\n", scrooge_prefill_generated_output_text)


if __name__ == "__main__":
    test_generate_flex_attention()
