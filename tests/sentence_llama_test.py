from transformers import AutoTokenizer

from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS

from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM, sentence_attention_forward, special_token_mask_to_clothest_token_idx_slow

import torch


def test_sentence_llama_model_generate():

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=5,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=False)
    print(response)

    assert 'rome' in response.lower()


def test_sentence_llama_model_generate_with_eos_token():

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')

    model.config._attn_implementation = 'sentence_attention'

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    assert (input_ids == model.config.end_of_sentence_token_id).sum().item() == 3

    print("input_ids", input_ids)
    output = model.generate(
        input_ids,
        max_new_tokens=5,
    )

def test_sentence_llama_model_generate_with_eos_token_and_attention_mask_pad():

    device = 'cuda'

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')

    model.config._attn_implementation = 'sentence_attention'

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    input_ids = input_ids.to(device)
    assert (input_ids == model.config.end_of_sentence_token_id).sum().item() == 3

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
        assert torch.allclose(output1.hidden_states[i], output2.hidden_states[i][:, :seq_len], atol=1e-2), f"hidden_states[{i}] are not equal"

    assert torch.allclose(output1.loss, output2.loss, atol=1e-3), f"output losses are not equal, l1={output1.loss}, l2={output2.loss}"


def test_sentence_llama_model_generate_with_eos_token_and_attention_mask_partial_logits():

    device = 'cuda'

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')

    # model.config._attn_implementation = 'eager'
    # model.config._attn_implementation = 'sdpa'
    model.config._attn_implementation = 'sentence_attention'

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    input_ids = input_ids.to(device)
    assert (input_ids == model.config.end_of_sentence_token_id).sum().item() == 3

    seq_len = input_ids.shape[1]

    model.eval()

    # print("input_ids", input_ids)
    output1 = model(
        input_ids,
        use_cache=False,
        output_hidden_states=True,
    )

    output2 = model(
        input_ids[:, :seq_len // 2],
        use_cache=False,
        output_hidden_states=True,
    )

    tokens_1 = output1.logits[:, :seq_len // 2, :].argmax(dim=-1)
    tokens_2 = output2.logits.argmax(dim=-1)

    print('tokens_1', tokens_1)
    print('tokens_2', tokens_2)

    assert (tokens_1 == tokens_2).all(), "tokens are not equal"

    logits_diff = (output1.logits[:, :seq_len // 2, :] - output2.logits).norm()
    assert logits_diff < 0.04, "logits diff is low"
    assert torch.allclose(output1.logits[:, :seq_len // 2, :], output2.logits, atol=1e-2), "logits are not equal"



def test_sentence_attention_attention_mask():

    hidden_size = 128
    batch_size = 4
    seq_len = 7
    num_heads = 8

    scaling = 1.0 / hidden_size ** 0.5

    q = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    k = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    v = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    special_embeddings_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    special_embeddings_mask[:, 2] = 1
    special_embeddings_mask[:, 5] = 1
    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

    output1, _ = sentence_attention_forward(
        None,
        q, k, v,
        attention_mask,
        scaling=scaling,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    seq_len2 = seq_len * 2
    q2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    k2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    v2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    q2[:, :, :seq_len] = q
    k2[:, :, :seq_len] = k
    v2[:, :, :seq_len] = v
    attention_mask2 = torch.ones(batch_size, 1, seq_len2, seq_len2)
    attention_mask2[:, :, seq_len:, seq_len:] = 0

    special_embeddings_mask2 = torch.zeros(batch_size, seq_len2, dtype=torch.long)
    special_embeddings_mask2[:, :seq_len] = special_embeddings_mask
    clothest_end_of_sentence_token_idx2 = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask2)

    output2, _ = sentence_attention_forward(
        None,
        q2, k2, v2,
        attention_mask2,
        scaling=scaling,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx2,
        special_embeddings_mask=special_embeddings_mask2,
    )

    assert torch.allclose(output1, output2[:, :, :seq_len, :]), "output1 and output2 are not equal"


def test_sentence_attention_causal_mask():

    hidden_size = 128
    batch_size = 4
    seq_len = 7
    num_heads = 8

    scaling = 1.0 / hidden_size ** 0.5

    q = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    k = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    v = torch.rand(batch_size, num_heads, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    special_embeddings_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    special_embeddings_mask[:, 2] = 1
    special_embeddings_mask[:, 5] = 1
    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

    output1, _ = sentence_attention_forward(
        None,
        q, k, v,
        attention_mask,
        scaling=scaling,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        special_embeddings_mask=special_embeddings_mask,
    )

    seq_len2 = seq_len * 2
    q2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    k2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    v2 = torch.rand(batch_size, num_heads, seq_len2, hidden_size)
    q2[:, :, :seq_len] = q
    k2[:, :, :seq_len] = k
    v2[:, :, :seq_len] = v
    attention_mask2 = torch.ones(batch_size, seq_len2)

    special_embeddings_mask2 = torch.zeros(batch_size, seq_len2, dtype=torch.long)
    special_embeddings_mask2[:, :seq_len] = special_embeddings_mask
    clothest_end_of_sentence_token_idx2 = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask2)

    output2, _ = sentence_attention_forward(
        None,
        q2, k2, v2,
        attention_mask2,
        scaling=scaling,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx2,
        special_embeddings_mask=special_embeddings_mask2,
    )

    assert torch.allclose(output1, output2[:, :, :seq_len, :]), "output1 and output2 are not equal"


if __name__ == "__main__":
    test_sentence_llama_model_generate_with_eos_token_and_attention_mask_partial_logits()