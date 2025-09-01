import pytest
import torch
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    SentenceLlamaModel,
    sentence_attention_forward,
    sentence_attention_forward_flex,
    special_token_mask_to_clothest_token_idx_slow,
)
from transformers.utils import is_torch_flex_attn_available

ARTIFACTS_PREFIX = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/"


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


@pytest.mark.skipif(not is_torch_flex_attn_available(), reason="Flex attention not available")
def test_sentence_attention_impl_equivalence():
    torch.manual_seed(0)

    batch_size = 3
    num_heads = 4
    seq_len = 128
    head_dim = 8

    # Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # 2D attention mask (no padding)
    attention_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Mark a few EOS tokens in the sequence
    special_embeddings_mask = torch.zeros_like(attention_mask_2d)
    special_positions = [3, 4, 5, 6, 100, 101, 102, 103]
    for pos in special_positions:
        special_embeddings_mask[:, pos] = 1

    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
        special_embeddings_mask,
        num_special_tokens=4,
    )

    # Build the 4D mask used by the SDPA path
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

    class Dummy:
        # Ensure no KV repetition
        num_key_value_groups = 1

    scaling = 1.0 / (head_dim**0.5)

    out_sdpa, _ = sentence_attention_forward(
        Dummy(), query, key, value, causal_mask_4d, dropout=0.0, scaling=scaling, is_causal=None
    )

    out_flex, _ = sentence_attention_forward_flex(
        Dummy(),
        query,
        key,
        value,
        attention_mask=attention_mask_2d,
        scaling=scaling,
        special_embeddings_mask=special_embeddings_mask,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
    )

    diff = (out_sdpa - out_flex).norm(2, dim=-1)
    print("diff", diff)
    print("out_sdpa", out_sdpa.shape)

    assert torch.allclose(out_sdpa, out_flex, atol=1e-5, rtol=1e-5)


def test_sentence_llama_model_generate_with_eos_token():

    device = "cuda"

    checkpoint = ARTIFACTS_PREFIX + "experiments/eos_4/sentence_Llama-3.2-1B_ft_full_R7NAB8H0/checkpoint-1349/"
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

    outputs = []

    model.eval()

    for attn_impl in ["sentence_attention", "sentence_attention_flex"]:
        model.config._attn_implementation = attn_impl

        # print("input_ids", input_ids)
        output = model.forward(input_ids, output_hidden_states=True)

        outputs.append(output.hidden_states)

        # print("attn_impl", attn_impl, "output", tokenizer.decode(output[0], skip_special_tokens=False))

    diffs = []
    for h_i in range(len(outputs[0])):
        diffs.append((outputs[0][h_i] - outputs[1][h_i]).norm(2, dim=-1))

    for h_i in range(len(outputs[0])):
        assert torch.allclose(outputs[0][h_i], outputs[1][h_i], atol=1e-5), f"hidden_states[{h_i}] are not equal"


def test_sentence_llama_model_generate_with_eos_token_and_attention_mask_pad():

    device = "cuda"

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_ids = [tokenizer.convert_tokens_to_ids("<end_of_sentence>")]

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
    model.config.end_of_sentence_token_ids = [tokenizer.convert_tokens_to_ids("<end_of_sentence>")]

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


if __name__ == "__main__":
    test_sentence_llama_model_generate_with_eos_token_and_attention_mask_partial_logits()
