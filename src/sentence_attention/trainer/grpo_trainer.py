import time

import torch
from transformers import Trainer

from typing import Any

from transformers.trainer_pt_utils import distributed_concat


class GRPOTrainer(Trainer):

    def sample_gist_tokens_for_input(self, model, inputs):
        """
        For each contiguous sequence of gist tokens, sample a new number of gist tokens k in [1, run_len],
        replace that gist sequence with `<end_of_sentence_{i}_{k}>` tokens, and right-pad to preserve tensor shapes.

        Repeats the sampling `self.args.grpo_num_generate_samples` times and returns a list of new input dicts.
        """

        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer is not available on trainer (expected `processing_class` or `tokenizer`).")

        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        special_embeddings_mask: torch.Tensor = inputs["special_embeddings_mask"]

        if input_ids.ndim != 2:
            raise ValueError(f"Expected 2D input_ids, got shape={tuple(input_ids.shape)}")
        if attention_mask.shape != input_ids.shape or special_embeddings_mask.shape != input_ids.shape:
            raise ValueError("input_ids, attention_mask, and special_embeddings_mask must have the same shape.")

        # Right padded (pads on the right, always attend to first token).
        if not (attention_mask[:, 0] == 1).all():
            raise ValueError("Input must be right padded (attention_mask[:, 0] must be 1).")

        batch_size, seq_len = input_ids.shape

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", 0) or 0

        def compute_closest_boundary_idx_from_runs(mask: torch.Tensor) -> torch.Tensor:
            # mask: [bs, seq_len] bool where True marks gist tokens.
            # Boundary is the last token in each contiguous run of gist tokens.
            is_special = mask.to(torch.bool)
            boundary = is_special & torch.logical_not(torch.roll(is_special, shifts=-1, dims=1))
            boundary[:, -1] = is_special[:, -1]

            positions = torch.arange(seq_len, device=mask.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            boundary_indices = torch.where(boundary, positions, torch.zeros_like(positions))
            last_boundary_inclusive, _ = torch.cummax(boundary_indices, dim=1)
            last_boundary_prev = torch.roll(last_boundary_inclusive, shifts=1, dims=1)
            last_boundary_prev[:, 0] = 0
            return last_boundary_prev

        def sample_one() -> dict:
            new_input_ids = input_ids.clone()
            new_attention_mask = attention_mask.clone()
            new_special_mask = special_embeddings_mask.clone()

            new_labels = None
            if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor) and inputs["labels"].shape == input_ids.shape:
                new_labels = inputs["labels"].clone()

            # Operate per-example to keep logic simple and robust.
            for b in range(batch_size):
                valid_len = int(new_attention_mask[b].sum().item())
                if valid_len <= 0:
                    continue

                ids = new_input_ids[b, :valid_len].tolist()
                spec = new_special_mask[b, :valid_len].to(torch.bool).tolist()
                lbls = new_labels[b, :valid_len].tolist() if new_labels is not None else None

                out_ids: list[int] = []
                out_spec: list[int] = []
                out_lbls: list[int] | None = [] if lbls is not None else None

                i = 0
                while i < valid_len:
                    if spec[i]:
                        # Consume a contiguous gist run.
                        j = i
                        while j < valid_len and spec[j]:
                            j += 1
                        run_len = j - i

                        # Sample k in [1, run_len] and substitute the entire run with k new gist tokens.
                        k = int(torch.randint(low=1, high=run_len + 1, size=(1,), device=input_ids.device).item())
                        for tok_i in range(k):
                            token_str = f"<end_of_sentence_{tok_i}_{k}>"
                            tok_id = tokenizer.convert_tokens_to_ids(token_str)
                            # HF tokenizers typically return unk_token_id for unknown tokens (not None).
                            unk_id = getattr(tokenizer, "unk_token_id", None)
                            if tok_id is None or (
                                unk_id is not None and tok_id == unk_id and token_str != getattr(tokenizer, "unk_token", None)
                            ):
                                raise ValueError(
                                    f"Token id not found for {token_str}. Did you call prepare_model_before_training?"
                                )
                            out_ids.append(int(tok_id))
                            out_spec.append(1)
                            if out_lbls is not None:
                                out_lbls.append(int(tok_id))
                        i = j
                    else:
                        out_ids.append(int(ids[i]))
                        out_spec.append(0)
                        if out_lbls is not None:
                            out_lbls.append(int(lbls[i]))
                        i += 1

                # Right-pad back to seq_len.
                if len(out_ids) > seq_len:
                    out_ids = out_ids[:seq_len]
                    out_spec = out_spec[:seq_len]
                    if out_lbls is not None:
                        out_lbls = out_lbls[:seq_len]

                pad_count = seq_len - len(out_ids)
                out_ids = out_ids + [int(pad_token_id)] * pad_count
                out_spec = out_spec + [0] * pad_count
                if out_lbls is not None:
                    # Preserve common label convention for padded positions.
                    out_lbls = out_lbls + [-100] * pad_count

                new_input_ids[b] = torch.tensor(out_ids, device=input_ids.device, dtype=input_ids.dtype)
                new_special_mask[b] = torch.tensor(out_spec, device=input_ids.device, dtype=new_special_mask.dtype)
                new_attention_mask[b] = (new_input_ids[b] != pad_token_id).to(dtype=new_attention_mask.dtype)
                if new_labels is not None and out_lbls is not None:
                    new_labels[b] = torch.tensor(out_lbls, device=input_ids.device, dtype=new_labels.dtype)

            result = dict(inputs)
            result["input_ids"] = new_input_ids
            result["attention_mask"] = new_attention_mask
            result["special_embeddings_mask"] = new_special_mask
            result["clothest_end_of_sentence_token_idx"] = compute_closest_boundary_idx_from_runs(
                new_special_mask.to(torch.bool) & new_attention_mask.to(torch.bool)
            )
            if new_labels is not None:
                result["labels"] = new_labels
            return result

        return [sample_one() for _ in range(self.args.grpo_num_generate_samples)]

    @staticmethod
    def prepare_model_before_training(model, tokenizer):

        # Add separate tokens with different sequence length
        additional_special_tokens = []
        for num_gist_tokens in range(1, tokenizer.num_eos_tokens + 1):
            for i in range(num_gist_tokens):
                additional_special_tokens.append(f"<end_of_sentence_{i}_{num_gist_tokens}>")

        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        # Resize embeddings for newly added special tokens.
        # Note: must re-fetch weights AFTER resizing, since resize_token_embeddings may replace modules.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        with torch.no_grad():
            input_weight = input_embeddings.weight

            output_weight = None
            output_bias = None
            if output_embeddings is not None and not model.config.tie_word_embeddings:
                output_weight = output_embeddings.weight
                if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
                    output_bias = output_embeddings.bias

            for num_gist_tokens in range(1, tokenizer.num_eos_tokens + 1):
                for i in range(num_gist_tokens):
                    new_gist_token = f"<end_of_sentence_{i}_{num_gist_tokens}>"
                    orig_gist_token = f"<end_of_sentence_{i}>"

                    new_id = tokenizer.convert_tokens_to_ids(new_gist_token)
                    orig_id = tokenizer.convert_tokens_to_ids(orig_gist_token)
                    if new_id is None or orig_id is None:
                        raise ValueError(f"Failed to resolve token ids for {new_gist_token=} {orig_gist_token=}")

                    # Initialize input embeddings for new tokens from the original gist token embeddings.
                    input_weight[new_id].copy_(input_weight[orig_id])

                    # For untied models, also initialize the LM head weights/bias for the new tokens.
                    if output_weight is not None:
                        if output_weight.shape[0] == len(tokenizer):
                            output_weight[new_id].copy_(output_weight[orig_id])
                        elif output_weight.ndim == 2 and output_weight.shape[1] == len(tokenizer):
                            output_weight[:, new_id].copy_(output_weight[:, orig_id])
                        else:
                            raise ValueError(
                                "Unexpected LM head weight shape while initializing new tokens: "
                                f"{tuple(output_weight.shape)} vs vocab={len(tokenizer)}"
                            )

                        if output_bias is not None:
                            if output_bias.shape[0] == len(tokenizer):
                                output_bias[new_id].copy_(output_bias[orig_id])
                            else:
                                raise ValueError(
                                    "Unexpected LM head bias shape while initializing new tokens: "
                                    f"{tuple(output_bias.shape)} vs vocab={len(tokenizer)}"
                                )

        return

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        log_metrics=True,
        log_prefix="debug",
        force_log=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        import torch.nn.functional as F

        labels = inputs.pop("labels")

        special_embeddings_mask = inputs.get("special_embeddings_mask")
        attention_mask = inputs["attention_mask"]
        if special_embeddings_mask is None:
            raise ValueError("special_embeddings_mask must be provided for GRPO training.")
        if special_embeddings_mask.shape != attention_mask.shape:
            raise ValueError("special_embeddings_mask and attention_mask must have the same shape.")

        # Generate multiple sampled "compressions" per input.
        base_inputs = dict(inputs)
        base_inputs["labels"] = labels
        sampled_inputs = self.sample_gist_tokens_for_input(model, base_inputs)
        if not isinstance(sampled_inputs, list) or len(sampled_inputs) == 0:
            raise ValueError("sample_gist_tokens_for_input must return a non-empty list of input dicts.")

        num_samples = len(sampled_inputs)
        batch_size, seq_len = inputs["input_ids"].shape

        # Flatten samples into a single batch for a single forward pass.
        flat_input_ids = torch.cat([s["input_ids"] for s in sampled_inputs], dim=0)
        flat_attention_mask = torch.cat([s["attention_mask"] for s in sampled_inputs], dim=0)
        flat_special_mask = torch.cat([s["special_embeddings_mask"] for s in sampled_inputs], dim=0)
        flat_closest_idx = torch.cat([s["clothest_end_of_sentence_token_idx"] for s in sampled_inputs], dim=0)
        flat_labels = torch.cat([s["labels"] for s in sampled_inputs], dim=0)

        model_kwargs = {
            "input_ids": flat_input_ids,
            "attention_mask": flat_attention_mask,
            "special_embeddings_mask": flat_special_mask,
            "clothest_end_of_sentence_token_idx": flat_closest_idx,
            "use_cache": False,
            "output_attentions": False,
        }

        if self.model_accepts_loss_kwargs:
            if num_items_in_batch is not None:
                model_kwargs["num_items_in_batch"] = num_items_in_batch

        torch.compiler.cudagraph_mark_step_begin()
        outputs = model(
            fused_linear_cross_entropy=False,
            labels=None,
            **model_kwargs,
        )

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if getattr(outputs, "logits", None) is None:
            raise ValueError("Model must return logits for GRPO loss computation (fused_linear_cross_entropy=False).")

        logits = outputs.logits  # [N*B, seq_len, vocab]

        # Per-example LM loss (causal shift), ignoring -100.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = flat_labels[:, 1:].contiguous()
        vocab_size = shift_logits.shape[-1]
        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        per_example_lm_loss = token_losses.sum(dim=1) / num_items_in_batch  # [N*B]

        # Reward = compression rate = (#non-gist tokens) / (#gist tokens)
        gist_tokens = (flat_special_mask.bool() & flat_attention_mask.bool()).sum(dim=1)
        total_tokens = flat_attention_mask.sum(dim=1)
        non_gist_tokens = (total_tokens - gist_tokens).clamp(min=0)
        reward = non_gist_tokens.float() / gist_tokens.clamp(min=1).float()  # [N*B]

        # Group rewards/losses per original example, and take reward-weighted LM loss.
        reward_grouped = reward.view(num_samples, batch_size).transpose(0, 1)  # [B, N]
        loss_grouped = per_example_lm_loss.view(num_samples, batch_size).transpose(0, 1)  # [B, N]
        reward_sum = reward_grouped.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = (reward_grouped / reward_sum).detach()
        weighted_lm_loss_per_example = (weights * loss_grouped).sum(dim=1)  # [B]
        loss = weighted_lm_loss_per_example.mean()

        # Mirror legacy metric naming.
        lm_loss_value = per_example_lm_loss.detach().mean()

        total_tokens_all = total_tokens.sum().clamp(min=1)
        gist_tokens_all = gist_tokens.sum()
        gist_tokens_percent = (gist_tokens_all.float() / total_tokens_all.float()) * 100 / self.args.gradient_accumulation_steps

        metrics = {
            "lm_loss": lm_loss_value,
            "gist_tokens_percent": gist_tokens_percent.detach(),
            "grpo_reward_mean": (reward.detach().mean() / self.args.gradient_accumulation_steps),
            "grpo_reward_std": (reward.detach().std(unbiased=False) / self.args.gradient_accumulation_steps),
        }

        outputs.loss = loss
        return (loss, outputs) if return_outputs else (loss, metrics)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):

        while True:
            try:
                super().save_model(output_dir, _internal_call)
                break
            except Exception as e:
                print("Error in saving model", e)
                time.sleep(300)

    def _nested_gather(self, tensors, name: str | None = None):
        """
        Gather arbitrary nested structures across processes, ensuring tensors are on the accelerator device.

        This avoids NCCL errors when attempting to all_gather CPU tensors by moving any tensors/scalars
        to `self.args.device` prior to using distributed collectives.
        """
        # Local import to avoid hard dependency if not using distributed
        import torch.distributed as dist

        # Fast path: no distributed initialized or single process
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return tensors

        device = (
            getattr(self.args, "device", None)
            or getattr(self.accelerator, "device", None)
            or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        def to_device_tensor(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                t = x
            else:
                try:
                    t = torch.tensor(x)
                except Exception:
                    # If it can't be tensorized, return as-is (will not be gathered)
                    return x  # type: ignore[return-value]
            if t.device != device:
                t = t.to(device)
            # Ensure at least 1D for concatenation semantics
            if t.dim() == 0:
                t = t.view(1)
            return t

        if isinstance(tensors, dict):
            return type(tensors)({k: self._nested_gather(v) for k, v in tensors.items()})
        if isinstance(tensors, (list, tuple)):
            gathered_list = [self._nested_gather(v) for v in tensors]
            return type(tensors)(gathered_list)

        # Base case: attempt to gather a tensor; if not tensorizable, return as-is
        t = to_device_tensor(tensors)
        if not isinstance(t, torch.Tensor):
            return tensors
        return distributed_concat(t)
