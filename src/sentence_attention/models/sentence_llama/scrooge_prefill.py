import torch
from tqdm.auto import tqdm
from transformers import DynamicCache


def full_prefill_small_kv_cache(
    model, input_ids, attention_mask, special_embeddings_mask, clothest_end_of_sentence_token_idx, output_hidden_states=False
):

    assert clothest_end_of_sentence_token_idx.shape[0] == 1, "only single size batch is supported"

    # TODO Sentence Cache - saves only the last sentence embedding
    past_key_values = DynamicCache()

    # TODO Calculate Last Chunk with possibly no sentence id

    # model.model.
    # full_ones_attention_mask = torch.ones_like(input_ids)

    hidden_states = []

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        past_key_values=past_key_values,
        output_hidden_states=output_hidden_states,
    )
    last_outputs = outputs

    kv_length = past_key_values.get_seq_length()
    special_embeddings_mask_kv_cache = special_embeddings_mask.clone()

    last_eos_token_idx = clothest_end_of_sentence_token_idx[0][-1].item()
    len_last_normal_tokens = input_ids.shape[1] - (last_eos_token_idx + 1)

    special_embeddings_mask_kv_cache[:, last_eos_token_idx:] = 1

    print("full_prefill_small_kv_cache special_embeddings_mask_kv_cache", special_embeddings_mask_kv_cache)

    indices = torch.nonzero(special_embeddings_mask_kv_cache, as_tuple=False)
    seq_indices = indices[:, 1]  # Extract sequence dimension indices

    for i in range(len(past_key_values.key_cache)):
        # Get the current cache tensor shape: [batch_size, num_heads, seq_len, head_dim]
        cache_shape = past_key_values.key_cache[i].shape
        batch_size, num_heads, seq_len, head_dim = cache_shape

        # Use torch.index_select to select only the special embedding positions
        # We need to select along the sequence dimension (dim=-2)
        past_key_values.key_cache[i] = torch.index_select(past_key_values.key_cache[i], dim=-2, index=seq_indices)
        past_key_values.value_cache[i] = torch.index_select(past_key_values.value_cache[i], dim=-2, index=seq_indices)

    print("past_key_values.key_cache", past_key_values.get_seq_length())

    kv_length = past_key_values.get_seq_length()

    next_token = outputs.logits[:, -1:].argmax(dim=-1)

    last_input_ids = torch.tensor([[next_token]], device=input_ids.device, dtype=input_ids.dtype)
    attention_mask = torch.ones([1, kv_length + 1], device=input_ids.device, dtype=torch.long)

    special_embeddings_mask_prefix = torch.ones(
        [1, kv_length - len_last_normal_tokens], device=special_embeddings_mask.device, dtype=special_embeddings_mask.dtype
    )
    special_embeddings_mask_suffix = [0]
    if last_input_ids[0, 0].item() in model.config.end_of_sentence_token_ids:
        special_embeddings_mask_suffix = [1]

    special_embeddings_mask_suffix = torch.tensor(
        [special_embeddings_mask_suffix], device=special_embeddings_mask.device, dtype=special_embeddings_mask.dtype
    )

    special_embeddings_mask_current = torch.cat(
        [special_embeddings_mask_prefix, special_embeddings_mask[:, last_eos_token_idx + 1 :], special_embeddings_mask_suffix],
        dim=-1,
    )

    assert (
        attention_mask.shape == special_embeddings_mask_current.shape
    ), "attention mask and special embeddings mask should have the same shape"

    clothest_end_of_sentence_token_idx_current = torch.zeros_like(
        last_input_ids, device=clothest_end_of_sentence_token_idx.device, dtype=clothest_end_of_sentence_token_idx.dtype
    )
    past_key_values = past_key_values
    cache_position = torch.arange(input_ids.shape[1], input_ids.shape[1] + 1, device=input_ids.device)

    assert cache_position.shape[0] == last_input_ids.shape[1], "cache position should have the same length as last input ids"

    return {
        "input_ids": last_input_ids,
        "attention_mask": attention_mask,
        "special_embeddings_mask": special_embeddings_mask_current,
        "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx_current,
        "cache_position": cache_position,
        "last_outputs": last_outputs,
        "past_key_values": past_key_values,
        "hidden_states": hidden_states,
    }


def scrooge_prefill(
    model,
    input_ids,
    attention_mask,
    special_embeddings_mask,
    clothest_end_of_sentence_token_idx,
    output_hidden_states=False,
    outputs_hook=None,
):

    assert clothest_end_of_sentence_token_idx.shape[0] == 1, "only single size batch is supported"

    eos_tokens_idxs = set(clothest_end_of_sentence_token_idx.cpu().numpy().tolist()[0])
    eos_tokens_idxs.remove(0)
    eos_tokens_idxs.add(input_ids.shape[1] - 1)
    # if input_ids.shape[1] in eos_tokens_idxs:
    #     eos_tokens_idxs.remove(input_ids.shape[1])
    eos_tokens_idxs = sorted(eos_tokens_idxs)

    num_eos_tokens = len(model.config.end_of_sentence_token_ids)

    # TODO Sentence Cache - saves only the last sentence embedding
    past_key_values = DynamicCache()

    # TODO Calculate Last Chunk with possibly no sentence id

    # model.model.
    # full_ones_attention_mask = torch.ones_like(input_ids)

    if output_hidden_states:
        hidden_states = []

    prev_sentence_i = (attention_mask == 0).sum().item()
    if prev_sentence_i > 0:
        assert attention_mask[0, 0].item() == 0, "attention mask is left padded"

    last_outputs = None

    len_last_normal_tokens = None

    assert eos_tokens_idxs[-1] == input_ids.shape[1] - 1, "last eos token idx should be the last token idx"

    for i, sentence_i in tqdm(enumerate(eos_tokens_idxs), desc="Scrooge prefill", total=len(eos_tokens_idxs)):

        kv_length = past_key_values.get_seq_length()

        sentence_i = min(sentence_i + 1, input_ids.shape[1])

        attention_mask = torch.ones([1, sentence_i - prev_sentence_i + kv_length], device=input_ids.device, dtype=torch.long)

        special_embeddings_mask_prefix = torch.ones(
            [1, kv_length], device=special_embeddings_mask.device, dtype=special_embeddings_mask.dtype
        )
        special_embeddings_mask_current = torch.cat(
            [special_embeddings_mask_prefix, special_embeddings_mask[:, prev_sentence_i:sentence_i]], dim=-1
        )

        clothest_end_of_sentence_token_idx_current = torch.zeros(
            [1, sentence_i - prev_sentence_i + kv_length],
            device=clothest_end_of_sentence_token_idx.device,
            dtype=clothest_end_of_sentence_token_idx.dtype,
        )
        # clothest_end_of_sentence_token_idx_current = torch.cat([clothest_end_of_sentence_token_idx_prefix, clothest_end_of_sentence_token_idx[:, prev_sentence_i:sentence_i]], dim=-1)

        # print("prev_sentence_i:sentence_i", prev_sentence_i, sentence_i)
        # print("processing", input_ids[:, prev_sentence_i:sentence_i])
        # print("attention_mask", attention_mask)
        # print("special_embeddings_mask_current", special_embeddings_mask_current)
        # print("clothest_end_of_sentence_token_idx_current", clothest_end_of_sentence_token_idx_current)
        # print("past_key_values", past_key_values.get_seq_length())

        outputs = model(
            input_ids=input_ids[:, prev_sentence_i:sentence_i],
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask_current,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx_current,
            past_key_values=past_key_values,
            cache_position=torch.arange(prev_sentence_i, sentence_i, device=input_ids.device),
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            hidden_states.append(outputs.hidden_states)

        if outputs_hook is not None:
            outputs_hook(input_ids, outputs, prev_sentence_i, sentence_i)

        last_outputs = outputs

        if sentence_i == input_ids.shape[1]:
            len_last_normal_tokens = sentence_i - prev_sentence_i
            break

        # Leave only sentence attention cache
        for idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[idx] != []:
                past_key_values.key_cache[idx] = torch.cat(
                    [
                        past_key_values.key_cache[idx][..., : i * num_eos_tokens, :],
                        past_key_values.key_cache[idx][..., -num_eos_tokens:, :],
                    ],
                    dim=-2,
                )
                past_key_values.value_cache[idx] = torch.cat(
                    [
                        past_key_values.value_cache[idx][..., : i * num_eos_tokens, :],
                        past_key_values.value_cache[idx][..., -num_eos_tokens:, :],
                    ],
                    dim=-2,
                )
                # if idx == 0:
                #     breakpoint()

        assert past_key_values.get_seq_length() == num_eos_tokens * (
            i + 1
        ), "cache seq len should be equal to number of sentences"

        # End of loop
        prev_sentence_i = sentence_i

        # print("past_key_values.get_seq_length()", past_key_values.get_seq_length())

    assert len_last_normal_tokens is not None, "len_last_normal_tokens should be set"
    assert len_last_normal_tokens > 0

    kv_length = past_key_values.get_seq_length()
    last_input_ids = torch.zeros_like(input_ids[:, -1:])
    last_input_ids[0, 0] = last_outputs.logits[:, -1:].argmax(dim=-1)

    attention_mask = torch.ones([1, kv_length + 1], device=input_ids.device, dtype=torch.long)

    special_embeddings_mask_prefix = torch.ones(
        [1, kv_length - len_last_normal_tokens], device=special_embeddings_mask.device, dtype=special_embeddings_mask.dtype
    )
    special_embeddings_mask_suffix = [0]
    if last_input_ids[0, 0].item() in model.config.end_of_sentence_token_ids:
        special_embeddings_mask_suffix = [1]

    special_embeddings_mask_suffix = torch.tensor(
        [special_embeddings_mask_suffix], device=special_embeddings_mask.device, dtype=special_embeddings_mask.dtype
    )

    special_embeddings_mask_current = torch.cat(
        [special_embeddings_mask_prefix, special_embeddings_mask[:, prev_sentence_i:], special_embeddings_mask_suffix], dim=-1
    )

    assert (
        attention_mask.shape == special_embeddings_mask_current.shape
    ), "attention mask and special embeddings mask should have the same shape"

    clothest_end_of_sentence_token_idx_current = torch.zeros_like(
        last_input_ids, device=clothest_end_of_sentence_token_idx.device, dtype=clothest_end_of_sentence_token_idx.dtype
    )
    past_key_values = past_key_values
    cache_position = torch.arange(sentence_i, sentence_i + 1, device=input_ids.device)

    assert cache_position.shape[0] == last_input_ids.shape[1], "cache position should have the same length as last input ids"

    return {
        "last_outputs": last_outputs,
        "hidden_states": hidden_states if output_hidden_states else None,
        "input_ids": last_input_ids,
        "attention_mask": attention_mask,
        "special_embeddings_mask": special_embeddings_mask_current,
        "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx_current,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
    }
