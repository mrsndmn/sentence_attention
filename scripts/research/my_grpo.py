import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
    SentenceLlamaForCausalLM,
)
from sentence_attention.trainer.build_model_tokenizer import _load_boundary_predictor_model


def reward(input_ids):

    # TODO count negative number if gist tokens
    num_gist_tokens = 0  # TODO

    return -num_gist_tokens


def modify_gist_tokens(special_embeddings_mask, input_ids, attention_mask, num_gist_tokens_to_keep):
    """
    Modify special_embeddings_mask and input_ids to keep only the first num_gist_tokens_to_keep
    gist tokens in each consecutive group of gist tokens. Also removes extra gist tokens from input_ids.

    Args:
        special_embeddings_mask: torch.Tensor of shape [batch, seq_len] with 1s for gist tokens
        input_ids: torch.Tensor of shape [batch, seq_len] with token IDs
        attention_mask: torch.Tensor of shape [batch, seq_len] with attention mask
        num_gist_tokens_to_keep: int, number of gist tokens to keep per group

    Returns:
        Tuple of (modified_special_embeddings_mask, modified_input_ids, modified_attention_mask)
    """
    batch_size, seq_len = special_embeddings_mask.shape
    device = special_embeddings_mask.device

    # Process each batch separately
    modified_masks = []
    modified_input_ids_list = []
    modified_attention_masks = []

    for batch_idx in range(batch_size):
        mask = special_embeddings_mask[batch_idx]
        input_ids_batch = input_ids[batch_idx]
        attention_mask_batch = attention_mask[batch_idx]

        # Build mask of indices to keep
        indices_to_keep = []
        i = 0
        while i < seq_len:
            if mask[i].item() == 1:  # Start of a gist token group
                # Find the end of this group
                group_start = i
                while i < seq_len and mask[i].item() == 1:
                    i += 1
                group_end = i
                group_length = group_end - group_start

                # Keep only the first num_gist_tokens_to_keep tokens
                tokens_to_keep_in_group = min(group_length, num_gist_tokens_to_keep)
                for idx in range(group_start, group_start + tokens_to_keep_in_group):
                    indices_to_keep.append(idx)
            else:
                # Keep all non-gist tokens
                indices_to_keep.append(i)
                i += 1

        # Create new tensors with only kept indices
        indices_to_keep_tensor = torch.tensor(indices_to_keep, device=device, dtype=torch.long)
        modified_masks.append(mask[indices_to_keep_tensor].unsqueeze(0))
        modified_input_ids_list.append(input_ids_batch[indices_to_keep_tensor].unsqueeze(0))
        modified_attention_masks.append(attention_mask_batch[indices_to_keep_tensor].unsqueeze(0))

    # Concatenate batches
    modified_mask = torch.cat(modified_masks, dim=0)
    modified_input_ids = torch.cat(modified_input_ids_list, dim=0)
    modified_attention_mask = torch.cat(modified_attention_masks, dim=0)

    return modified_mask, modified_input_ids, modified_attention_mask


if __name__ == "__main__":
    device = "cuda"

    checkpoint = "artifacts/experiments/eos_8/sentence_Llama-3.2-1B_boundaries_prediction_full_boundary_predictor__num_eos_tokens_8_KHTDPJ7Z/checkpoint-1000"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model.config._attn_implementation = "sentence_attention"

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    bp_checkpoint = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments/bp/unsloth__Llama-3.2-1B/model.pt"
    boundary_predictor_model = _load_boundary_predictor_model(bp_checkpoint, model)
    boundary_predictor_model.to(device)
    tokenizer.set_boundary_predictor_model(boundary_predictor_model)
    # Set gist_placement to boundary_predictor to enable gist token insertion
    tokenizer.gist_placement = "boundary_predictor"
    print(f"Tokenizer gist_placement: {tokenizer.gist_placement}")

    # pass
    test_text = "Elizabeth's refusal of Mr. Collins offer of marriage showed an independence seldom seen in heroines of the day. Her refusal of Mr. Darcy while triggered by anger showed a level of independence that left him shocked and stunned."
    # Call encode_plus for single text (not batch) to ensure gist tokens are inserted
    encoding = tokenizer.encode_plus(test_text)
    print(f"Encoding keys (before tensor conversion): {list(encoding.keys())}")

    # Convert to tensors manually
    input_ids_list = encoding["input_ids"]
    if isinstance(input_ids_list[0], list):
        input_ids_list = input_ids_list[0]
    attention_mask_list = encoding.get("attention_mask", [1] * len(input_ids_list))
    if isinstance(attention_mask_list[0], list):
        attention_mask_list = attention_mask_list[0]

    # Convert to tensors and add batch dimension
    input_ids_base = torch.tensor([input_ids_list], device=device)
    attention_mask_tensor = torch.tensor([attention_mask_list], device=device)

    # Get base special_embeddings_mask - if not present, compute from special_tokens_mask or gist token IDs
    special_embeddings_mask_base = encoding.get("special_embeddings_mask")
    if special_embeddings_mask_base is None:
        # Compute special_embeddings_mask from special_tokens_mask or gist token IDs
        special_tokens_mask = encoding.get("special_tokens_mask")
        if special_tokens_mask is not None:
            # Use special_tokens_mask as special_embeddings_mask
            special_embeddings_mask_base = special_tokens_mask
        else:
            # Compute from gist token IDs
            gist_token_ids = set(tokenizer.end_of_sentence_token_ids)
            special_embeddings_mask_base = [1 if token_id in gist_token_ids else 0 for token_id in input_ids_list]

    if isinstance(special_embeddings_mask_base, list):
        special_embeddings_mask_base = torch.tensor(special_embeddings_mask_base, device=device)
    else:
        special_embeddings_mask_base = special_embeddings_mask_base.to(device)
    if special_embeddings_mask_base.dim() == 1:
        special_embeddings_mask_base = special_embeddings_mask_base.unsqueeze(0)

    # Use attention mask from encoding, ensure correct dtype (long for transformers, will be converted internally)
    attention_mask = attention_mask_tensor.to(torch.long)

    # Store results for plotting
    num_gist_tokens_list = []
    losses_non_gist = []
    losses_full = []

    # Loop over different numbers of gist tokens (1 to 8)
    for num_gist_tokens_to_keep in range(1, 9):
        print(f"\n=== Testing with {num_gist_tokens_to_keep} gist tokens ===")

        # Modify special_embeddings_mask, input_ids, and attention_mask to keep only first num_gist_tokens_to_keep gist tokens per group
        special_embeddings_mask, input_ids, attention_mask_modified = modify_gist_tokens(
            special_embeddings_mask_base, input_ids_base, attention_mask, num_gist_tokens_to_keep
        )

        # Recompute clothest_end_of_sentence_token_idx with modified mask
        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            special_embeddings_mask.bool(), num_special_tokens=num_gist_tokens_to_keep
        )
        if clothest_end_of_sentence_token_idx.dim() == 1:
            clothest_end_of_sentence_token_idx = clothest_end_of_sentence_token_idx.unsqueeze(0)

        # Create labels (shifted input_ids for next token prediction)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Last token has no next token

        # Mask out gist tokens in labels for non-gist loss (set to -100 where the target token is a gist token)
        gist_mask_next_token = special_embeddings_mask[:, 1:].bool()  # [batch, seq_len-1]
        labels_non_gist = labels.clone()
        labels_non_gist[:, :-1][gist_mask_next_token] = -100

        # Run model forward pass for non-gist loss
        model.eval()
        with torch.no_grad():
            outputs_non_gist = model(
                input_ids=input_ids,
                attention_mask=attention_mask_modified.bool(),
                labels=labels_non_gist,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            )

        # Run model forward pass for full loss (including gist tokens)
        with torch.no_grad():
            outputs_full = model(
                input_ids=input_ids,
                attention_mask=attention_mask_modified.bool(),
                labels=labels,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            )

        loss_non_gist = outputs_non_gist.loss
        loss_full = outputs_full.loss
        losses_non_gist.append(loss_non_gist.item())
        losses_full.append(loss_full.item())
        num_gist_tokens_list.append(num_gist_tokens_to_keep)

        # Count tokens for context
        num_non_gist_tokens = (~special_embeddings_mask.bool() & attention_mask_modified.bool()).sum().item()
        num_gist_tokens = (special_embeddings_mask.bool() & attention_mask_modified.bool()).sum().item()
        print(f"Loss for non-gist tokens: {loss_non_gist.item():.4f}")
        print(f"Full loss (with gist tokens): {loss_full.item():.4f}")
        print(f"Total tokens: {attention_mask_modified.sum().item()}")
        print(f"Non-gist tokens: {num_non_gist_tokens}")
        print(f"Gist tokens: {num_gist_tokens}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(
        num_gist_tokens_list,
        losses_non_gist,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Loss (non-gist tokens)",
        color="blue",
    )
    plt.plot(
        num_gist_tokens_list,
        losses_full,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Full loss (with gist tokens)",
        color="red",
    )
    plt.xlabel("Number of Gist Tokens", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss vs Number of Gist Tokens", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(num_gist_tokens_list)
    plt.tight_layout()
    plt.savefig("/tmp/loss_vs_gist_tokens.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to '/tmp/loss_vs_gist_tokens.png'")
    plt.show()
