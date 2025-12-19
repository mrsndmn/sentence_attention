import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
    SentenceLlamaForCausalLM,
)
from sentence_attention.trainer.build_model_tokenizer import _load_boundary_predictor_model


def modify_gist_tokens_with_per_boundary_counts(special_embeddings_mask, input_ids, attention_mask, gist_counts_per_boundary):
    """
    Modify special_embeddings_mask and input_ids to keep only the specified number of gist tokens
    for each boundary group. Also removes extra gist tokens from input_ids.

    Args:
        special_embeddings_mask: torch.Tensor of shape [batch, seq_len] with 1s for gist tokens
        input_ids: torch.Tensor of shape [batch, seq_len] with token IDs
        attention_mask: torch.Tensor of shape [batch, seq_len] with attention mask
        gist_counts_per_boundary: list[int], number of gist tokens to keep for each boundary group

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
        boundary_idx = 0
        i = 0
        while i < seq_len:
            if mask[i].item() == 1:  # Start of a gist token group
                # Find the end of this group
                group_start = i
                while i < seq_len and mask[i].item() == 1:
                    i += 1
                group_end = i
                group_length = group_end - group_start

                # Get the number of gist tokens to keep for this boundary
                if boundary_idx < len(gist_counts_per_boundary):
                    num_gist_tokens_to_keep = gist_counts_per_boundary[boundary_idx]
                else:
                    num_gist_tokens_to_keep = group_length  # Keep all if not specified
                boundary_idx += 1

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


def get_boundary_positions(special_embeddings_mask):
    """Extract boundary positions (start of each gist token group) from special_embeddings_mask."""
    boundary_positions = []
    seq_len = special_embeddings_mask.shape[1]
    i = 0
    while i < seq_len:
        if special_embeddings_mask[0, i].item() == 1:
            boundary_positions.append(i)
            # Skip the rest of this gist token group
            while i < seq_len and special_embeddings_mask[0, i].item() == 1:
                i += 1
        else:
            i += 1
    return boundary_positions


def compute_losses(model, input_ids, attention_mask, special_embeddings_mask, clothest_end_of_sentence_token_idx):
    """Compute full loss, gist tokens loss, and non-gist tokens loss."""
    # Create labels (shifted input_ids for next token prediction)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Last token has no next token

    # Mask out gist tokens in labels for non-gist loss
    gist_mask_next_token = special_embeddings_mask[:, 1:].bool()  # [batch, seq_len-1]
    labels_non_gist = labels.clone()
    labels_non_gist[:, :-1][gist_mask_next_token] = -100

    # Mask out non-gist tokens in labels for gist loss
    labels_gist = labels.clone()
    labels_gist[:, :-1][~gist_mask_next_token] = -100

    # Run model forward pass for non-gist loss
    with torch.no_grad():
        outputs_non_gist = model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            labels=labels_non_gist,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        )

    # Run model forward pass for gist loss
    with torch.no_grad():
        outputs_gist = model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            labels=labels_gist,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        )

    # Run model forward pass for full loss
    with torch.no_grad():
        outputs_full = model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            labels=labels,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        )

    return outputs_full.loss.item(), outputs_gist.loss.item(), outputs_non_gist.loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample gist tokens per boundary and compute loss statistics")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/experiments/eos_8/sentence_Llama-3.2-1B_boundaries_prediction_full_boundary_predictor__num_eos_tokens_8_manual_base_weights_KH000000/checkpoint-1000",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--bp_checkpoint",
        type=str,
        default="artifacts/experiments/bp/unsloth__Llama-3.2-1B_drop_punkt_0.0_bs_4_train_samples_10000/model.pt",
        help="Boundary predictor checkpoint path",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Elizabeth's refusal of Mr. Collins offer of marriage showed an independence seldom seen in heroines of the day. Her refusal of Mr. Darcy while triggered by anger showed a level of independence that left him shocked and stunned.",
        help="Text to process",
    )
    args = parser.parse_args()

    device = "cuda"

    model = SentenceLlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.config._attn_implementation = "sentence_attention"
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    boundary_predictor_model = _load_boundary_predictor_model(args.bp_checkpoint, model)
    boundary_predictor_model.to(device)
    tokenizer.set_boundary_predictor_model(boundary_predictor_model)
    tokenizer.gist_placement = "boundary_predictor"
    print(f"Tokenizer gist_placement: {tokenizer.gist_placement}")
    print(f"Number of EOS tokens: {tokenizer.num_eos_tokens}")

    # Get boundaries prediction from pretrained model
    encoding = tokenizer.encode_plus(args.text)
    print(f"Encoding keys: {list(encoding.keys())}")

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

    # Get base special_embeddings_mask
    special_embeddings_mask_base = encoding.get("special_embeddings_mask")
    if special_embeddings_mask_base is None:
        special_tokens_mask = encoding.get("special_tokens_mask")
        if special_tokens_mask is not None:
            special_embeddings_mask_base = special_tokens_mask
        else:
            gist_token_ids = set(tokenizer.end_of_sentence_token_ids)
            special_embeddings_mask_base = [1 if token_id in gist_token_ids else 0 for token_id in input_ids_list]

    if isinstance(special_embeddings_mask_base, list):
        special_embeddings_mask_base = torch.tensor(special_embeddings_mask_base, device=device)
    else:
        special_embeddings_mask_base = special_embeddings_mask_base.to(device)
    if special_embeddings_mask_base.dim() == 1:
        special_embeddings_mask_base = special_embeddings_mask_base.unsqueeze(0)

    attention_mask = attention_mask_tensor.to(torch.long)

    # Get boundary positions
    boundary_positions = get_boundary_positions(special_embeddings_mask_base)
    num_boundaries = len(boundary_positions)
    print(f"Number of boundaries found: {num_boundaries}")

    # Store metrics for all samples
    compression_rates = []
    losses_full = []
    losses_gist = []
    losses_non_gist = []

    # Generate N samples
    for sample_idx in range(args.num_samples):
        # Sample uniformly the number of gist tokens for each boundary (from 1 to num_eos_tokens)
        gist_counts_per_boundary = [np.random.randint(1, tokenizer.num_eos_tokens + 1) for _ in range(num_boundaries)]
        print("gist_counts_per_boundary", gist_counts_per_boundary)

        # Modify gist tokens based on sampled counts
        special_embeddings_mask, input_ids, attention_mask_modified = modify_gist_tokens_with_per_boundary_counts(
            special_embeddings_mask_base, input_ids_base, attention_mask, gist_counts_per_boundary
        )

        # Find max number of gist tokens in any boundary for computing clothest_end_of_sentence_token_idx
        max_gist_tokens = max(gist_counts_per_boundary) if gist_counts_per_boundary else tokenizer.num_eos_tokens

        # Recompute clothest_end_of_sentence_token_idx with modified mask
        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            special_embeddings_mask.bool(), num_special_tokens=max_gist_tokens
        )
        if clothest_end_of_sentence_token_idx.dim() == 1:
            clothest_end_of_sentence_token_idx = clothest_end_of_sentence_token_idx.unsqueeze(0)

        # Compute losses
        loss_full, loss_gist, loss_non_gist = compute_losses(
            model, input_ids, attention_mask_modified, special_embeddings_mask, clothest_end_of_sentence_token_idx
        )

        # Count tokens
        num_non_gist_tokens = (~special_embeddings_mask.bool() & attention_mask_modified.bool()).sum().item()
        num_gist_tokens = (special_embeddings_mask.bool() & attention_mask_modified.bool()).sum().item()

        # Compute compression rate
        compression_rate = num_non_gist_tokens / (num_gist_tokens + 1) if num_non_gist_tokens > 0 else 0.0

        # Store metrics
        compression_rates.append(compression_rate)
        losses_full.append(loss_full)
        losses_gist.append(loss_gist)
        losses_non_gist.append(loss_non_gist)

        if (sample_idx + 1) % 10 == 0:
            print(f"Processed {sample_idx + 1}/{args.num_samples} samples")

    # Compute and print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Number of samples: {args.num_samples}")
    print("\nCompression Rate (gist_tokens / non_gist_tokens):")
    compression_rates_valid = [x for x in compression_rates if not np.isnan(x)]
    if compression_rates_valid:
        print(f"  Mean: {np.mean(compression_rates_valid):.4f}")
        print(f"  Std:  {np.std(compression_rates_valid):.4f}")
        print(f"  Min:  {np.min(compression_rates_valid):.4f}")
        print(f"  Max:  {np.max(compression_rates_valid):.4f}")
    else:
        print("  No valid values")
    print("\nFull Loss:")
    losses_full_valid = [x for x in losses_full if not np.isnan(x)]
    if losses_full_valid:
        print(f"  Mean: {np.mean(losses_full_valid):.4f}")
        print(f"  Std:  {np.std(losses_full_valid):.4f}")
    else:
        print("  No valid values")
    print("\nGist Tokens Loss:")
    losses_gist_valid = [x for x in losses_gist if not np.isnan(x)]
    if losses_gist_valid:
        print(f"  Mean: {np.mean(losses_gist_valid):.4f}")
        print(f"  Std:  {np.std(losses_gist_valid):.4f}")
    else:
        print("  No valid values")
    print("\nNon-Gist Tokens Loss:")
    losses_non_gist_valid = [x for x in losses_non_gist if not np.isnan(x)]
    if losses_non_gist_valid:
        print(f"  Mean: {np.mean(losses_non_gist_valid):.4f}")
        print(f"  Std:  {np.std(losses_non_gist_valid):.4f}")
    else:
        print("  No valid values")
    print("=" * 80)
