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

        Always returns one "baseline" sample with the original gist token placement (i.e. keep all gist tokens),
        and uses sampling for the remaining `self.args.grpo_num_generate_samples - 1` samples.
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

        temperature = float(self.args.grpo_sampling_temperature)
        top_p = float(self.args.grpo_sampling_top_p)
        if temperature <= 0:
            raise ValueError("grpo_sampling_temperature must be > 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("grpo_sampling_top_p must be in (0, 1]")

        # Anneal sampling strategy: start with uniform sampling for a few steps, then switch to logits-based sampling.
        pretrain_steps = self.args.grpo_pretrain_sampling_steps
        if pretrain_steps < 0:
            pretrain_steps = 0
        global_step = self.state.global_step
        use_model_based_sampling = global_step >= pretrain_steps

        def compute_closest_boundary_idx_from_runs(mask: torch.Tensor) -> torch.Tensor:
            # mask: [bs, L] bool where True marks gist tokens.
            # Boundary is the last token in each contiguous run of gist tokens.
            if mask.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape={tuple(mask.shape)}")
            bs, length = mask.shape
            is_special = mask.to(torch.bool)
            boundary = is_special & torch.logical_not(torch.roll(is_special, shifts=-1, dims=1))
            boundary[:, -1] = is_special[:, -1]

            positions = torch.arange(length, device=mask.device, dtype=torch.long).unsqueeze(0).expand(bs, -1)
            boundary_indices = torch.where(boundary, positions, torch.zeros_like(positions))
            last_boundary_inclusive, _ = torch.cummax(boundary_indices, dim=1)
            last_boundary_prev = torch.roll(last_boundary_inclusive, shifts=1, dims=1)
            last_boundary_prev[:, 0] = 0
            return last_boundary_prev

        def _resolve_token_id(token_str: str) -> int:
            tok_id = tokenizer.convert_tokens_to_ids(token_str)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if tok_id is None or (
                unk_id is not None and tok_id == unk_id and token_str != getattr(tokenizer, "unk_token", None)
            ):
                raise ValueError(f"Token id not found for {token_str}. Did you call prepare_model_before_training?")
            return int(tok_id)

        # Build candidate ids for sampling the *first* token in a gist block: <end_of_sentence_0_{k}>.
        # We only ever sample among tokens that exist in the tokenizer.
        max_eos = int(getattr(tokenizer, "num_eos_tokens", 0) or 0)
        if max_eos <= 0:
            raise ValueError("Tokenizer must expose num_eos_tokens > 0 for GRPO gist sampling.")

        # Precompute all possible first-token ids (k=1..max_eos). We'll later slice to run_len.
        first_token_ids_all: list[int] = []
        first_token_id_to_k: dict[int, int] = {}
        for k in range(1, max_eos + 1):
            tok_id = _resolve_token_id(f"<end_of_sentence_0_{k}>")
            first_token_ids_all.append(tok_id)
            first_token_id_to_k[tok_id] = k

        def _sample_first_gist_token_id(next_logits: torch.Tensor | None, run_len: int) -> int:
            # Candidates are <end_of_sentence_0_{k}> for k in [1, run_len], bounded by tokenizer.num_eos_tokens.
            run_len = int(min(run_len, len(first_token_ids_all)))
            if run_len <= 0:
                raise ValueError(f"Invalid run_len={run_len} for gist sampling.")

            if next_logits is None:
                idx = int(torch.randint(low=0, high=run_len, size=(1,), device=input_ids.device).item())
                return int(first_token_ids_all[idx])

            candidate_ids = torch.tensor(first_token_ids_all[:run_len], device=next_logits.device, dtype=torch.long)
            cand_logits = next_logits.index_select(0, candidate_ids).to(torch.float32) / temperature
            probs = torch.softmax(cand_logits, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            if float(probs.sum().detach().cpu().item()) <= 0.0:
                idx = int(torch.randint(low=0, high=run_len, size=(1,), device="cpu").item())
                return int(first_token_ids_all[idx])

            # Nucleus (top-p) sampling on the candidate set.
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            if keep.numel() > 0:
                keep[0] = True  # always keep at least one token
            kept_probs = sorted_probs[keep]
            kept_idx = sorted_idx[keep]
            kept_sum = kept_probs.sum()
            if not torch.isfinite(kept_sum) or float(kept_sum.detach().cpu().item()) <= 0.0:
                idx = int(torch.randint(low=0, high=int(kept_idx.numel()), size=(1,), device="cpu").item())
                sampled_candidate_pos = int(kept_idx[idx].item())
                return int(candidate_ids[sampled_candidate_pos].item())
            kept_probs = (kept_probs / kept_sum).to(torch.float32)

            # Sample on CPU to avoid CUDA device-side asserts on invalid distributions.
            sampled_rel = int(torch.multinomial(kept_probs.detach().cpu(), num_samples=1).item())
            sampled_candidate_pos = int(kept_idx[sampled_rel].item())
            return int(candidate_ids[sampled_candidate_pos].item())

        def sample_one() -> dict:
            new_input_ids = input_ids.clone()
            new_attention_mask = attention_mask.clone()
            new_special_mask = special_embeddings_mask.clone()

            new_labels = None
            if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor) and inputs["labels"].shape == input_ids.shape:
                new_labels = inputs["labels"].clone()

            was_training = model.training
            try:
                model.eval()
                with torch.no_grad():
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

                        # Maintain an incremental cache over the *constructed* output prefix so that each sampled
                        # gist-run is conditioned on previously sampled gist tokens (not the original ones).
                        past = None
                        fed_spec: list[int] = []
                        next_logits: torch.Tensor | None = None  # logits for the next token to generate

                        pending_ids: list[int] = []
                        pending_spec: list[int] = []

                        def flush_pending() -> None:
                            nonlocal past, next_logits, fed_spec, pending_ids, pending_spec
                            if len(pending_ids) == 0:
                                return

                            total_spec = fed_spec + pending_spec
                            total_len = len(total_spec)
                            attention = torch.ones((1, total_len), device=input_ids.device, dtype=torch.long)
                            special = torch.tensor(total_spec, device=input_ids.device, dtype=torch.long).unsqueeze(0)
                            closest = compute_closest_boundary_idx_from_runs(special.to(torch.bool) & attention.to(torch.bool))

                            chunk = torch.tensor(pending_ids, device=input_ids.device, dtype=new_input_ids.dtype).unsqueeze(0)
                            outputs = model(
                                fused_linear_cross_entropy=False,
                                labels=None,
                                input_ids=chunk,
                                attention_mask=attention,
                                special_embeddings_mask=special,
                                clothest_end_of_sentence_token_idx=closest,
                                past_key_values=past,
                                use_cache=True,
                                logits_to_keep=1,
                                output_attentions=False,
                            )
                            if getattr(outputs, "logits", None) is None:
                                raise ValueError(
                                    "Model must return logits for GRPO sampling (fused_linear_cross_entropy=False)."
                                )
                            if getattr(outputs, "past_key_values", None) is None:
                                raise ValueError("Model must return past_key_values when use_cache=True for GRPO sampling.")

                            past = outputs.past_key_values
                            next_logits = outputs.logits[0, -1, :].detach()
                            fed_spec = total_spec
                            pending_ids = []
                            pending_spec = []

                        i = 0
                        while i < valid_len:
                            if spec[i]:
                                # Consume a contiguous gist run (from the original input) and replace it with k tokens.
                                j = i
                                while j < valid_len and spec[j]:
                                    j += 1
                                run_len = j - i

                                # Ensure next_logits is conditioned on everything we've already written to out_*.
                                flush_pending()

                                logits_for_sampling = next_logits if use_model_based_sampling else None
                                first_tok_id = _sample_first_gist_token_id(next_logits=logits_for_sampling, run_len=run_len)
                                k = int(first_token_id_to_k.get(first_tok_id, 0))
                                if k <= 0 or k > run_len:
                                    raise ValueError(
                                        "Sampled an invalid first gist token id. "
                                        f"{first_tok_id=} {k=} {run_len=}. This should not happen."
                                    )

                                for tok_i in range(k):
                                    token_str = f"<end_of_sentence_{tok_i}_{k}>"
                                    tok_id = _resolve_token_id(token_str)
                                    out_ids.append(int(tok_id))
                                    out_spec.append(1)
                                    if out_lbls is not None:
                                        out_lbls.append(int(tok_id))
                                    pending_ids.append(int(tok_id))
                                    pending_spec.append(1)

                                # Update cache with the inserted gist tokens so later sampling sees them in context.
                                flush_pending()
                                i = j
                            else:
                                tok = int(ids[i])
                                out_ids.append(tok)
                                out_spec.append(0)
                                if out_lbls is not None:
                                    out_lbls.append(int(lbls[i]))
                                pending_ids.append(tok)
                                pending_spec.append(0)
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
            finally:
                model.train(was_training)

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

        orig_attn_impl = model.config._attn_implementation
        model.config._attn_implementation = "sentence_attention"

        num_generate_samples = int(self.args.grpo_num_generate_samples)
        if num_generate_samples <= 0:
            raise ValueError("grpo_num_generate_samples must be > 0")

        # Baseline sample: keep the original gist token placement unchanged (all gist tokens present).
        baseline = dict(inputs)
        if (
            "clothest_end_of_sentence_token_idx" not in baseline
            or not isinstance(baseline["clothest_end_of_sentence_token_idx"], torch.Tensor)
            or baseline["clothest_end_of_sentence_token_idx"].shape != input_ids.shape
        ):
            baseline["clothest_end_of_sentence_token_idx"] = compute_closest_boundary_idx_from_runs(
                baseline["special_embeddings_mask"].to(torch.bool) & baseline["attention_mask"].to(torch.bool)
            )

        result = [baseline]
        result.extend(sample_one() for _ in range(num_generate_samples - 1))

        model.config._attn_implementation = orig_attn_impl
        # torch.cuda.empty_cache()

        return result

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
        num_start_gist_tokens_items_in_batch=None,
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

        unwrapped_model = self.accelerator.unwrap_model(model)
        sampled_inputs = self.sample_gist_tokens_for_input(unwrapped_model, base_inputs)
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

        # Masks aligned to the shifted targets (positions 1..L-1).
        flat_special_bool = flat_special_mask.bool() & flat_attention_mask.bool()  # [N*B, L]
        prev_special = torch.zeros_like(flat_special_bool)
        prev_special[:, 1:] = flat_special_bool[:, :-1]
        starts = flat_special_bool & (~prev_special)  # [N*B, L]

        target_is_valid = (flat_attention_mask[:, 1:].bool()) & (shift_labels != -100)  # [N*B, L-1]
        starts_target = starts[:, 1:] & target_is_valid  # [N*B, L-1]

        # Supervised CE: by default keep it on all tokens EXCEPT the action tokens (first token of each gist-run).
        supervised_mask = target_is_valid
        if bool(self.args.grpo_supervised_exclude_action_tokens):
            supervised_mask = supervised_mask & (~starts_target)

        # supervised_denom = supervised_mask.sum().clamp(min=1)
        lm_loss = (token_losses * supervised_mask).sum() / (num_items_in_batch * num_samples)

        # Group rewards/losses per original example, and take reward-weighted LM loss.
        # Compute non-gist causal LM loss (quality proxy) and shape reward as compression_gain - beta * penalty.
        # Baseline sample is expected to be sample index 0 (see sample_gist_tokens_for_input).
        target_is_special = flat_special_mask[:, 1:].bool()  # [N*B, L-1]
        non_gist_target_mask = (~target_is_special) & target_is_valid  # [N*B, L-1]
        non_gist_denom = non_gist_target_mask.sum(dim=1).clamp(min=1)
        non_gist_causal_lm_loss = (token_losses * non_gist_target_mask).sum(dim=1) / non_gist_denom  # [N*B]

        non_gist_loss_grouped = non_gist_causal_lm_loss.view(num_samples, batch_size)  # [N, B]
        baseline_non_gist_loss = non_gist_loss_grouped[0]  # [B]
        better_than_baseline = non_gist_loss_grouped < baseline_non_gist_loss.unsqueeze(0)  # [N, B]
        better_than_baseline[0, :] = False
        better_than_baseline_percent = better_than_baseline.sum() / better_than_baseline.numel()

        # Compression signal: fewer gist tokens is better (baseline has all original gist tokens).
        gist_tokens = (flat_special_mask.bool() & flat_attention_mask.bool()).sum(dim=1)  # [N*B]
        gist_grouped = gist_tokens.view(num_samples, batch_size).transpose(0, 1).to(torch.float32)  # [B, N]
        gist_baseline = gist_grouped[:, :1].clamp(min=1.0)  # [B, 1]
        compression_gain = (gist_baseline - gist_grouped) / (
            gist_baseline * self.args.gradient_accumulation_steps * self.args.world_size
        )  # [B, N] in [0, 1]

        # Quality penalty: increase in non-gist CE vs baseline.
        non_gist_grouped = non_gist_loss_grouped.transpose(0, 1).to(torch.float32)  # [B, N]
        non_gist_delta = non_gist_grouped - baseline_non_gist_loss.to(torch.float32).unsqueeze(1)  # [B, N]
        margin = float(self.args.grpo_quality_margin)
        quality_penalty = torch.relu(non_gist_delta - margin)  # [B, N]
        beta = float(self.args.grpo_quality_beta)

        reward_grouped = compression_gain - beta * quality_penalty  # [B, N]
        clip = float(self.args.grpo_reward_clip)
        if clip > 0:
            reward_grouped = reward_grouped.clamp(min=-clip, max=clip)

        reward_std = reward_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantage = ((reward_grouped - reward_grouped.mean(dim=1, keepdim=True)) / reward_std).detach()

        # GRPO loss on the "action" tokens only: first token of each gist-run.
        global_action_count = num_start_gist_tokens_items_in_batch * num_samples
        if global_action_count is None:
            action_count = starts_target.sum(dim=1).clamp(min=1)
            global_action_count = action_count * self.args.gradient_accumulation_steps * self.args.world_size

        action_nll = (token_losses * starts_target).sum(dim=1) / global_action_count  # [N*B]
        action_nll_grouped = action_nll.view(num_samples, batch_size).transpose(0, 1)  # [B, N]

        grpo_loss_per_example = (advantage.detach() * action_nll_grouped).mean(dim=1) * float(self.args.grpo_loss_alpha)  # [B]

        loss = lm_loss
        if not (reward_std <= 1e-8).all():
            loss = loss + grpo_loss_per_example.mean()

        if torch.isnan(loss):
            print("Found nan loss")
            breakpoint()

        # Mirror legacy metric naming.
        lm_loss_value = lm_loss.detach()

        total_tokens = flat_attention_mask.sum(dim=1)
        gist_tokens_all = gist_tokens.sum()
        gist_tokens_percent = (
            gist_tokens_all.float() * 100 / (total_tokens * self.args.gradient_accumulation_steps * self.args.world_size)
        )

        # Distribution of "first gist token" types (<end_of_sentence_0_{k}>) as a percent of ALL gist tokens.
        # We define "first gist token" as the first token in each contiguous gist-run in the sampled sequence.
        tokenizer = self.processing_class
        max_eos: int = tokenizer.num_eos_tokens

        denom_gist = starts.sum().float().clamp(min=1.0)
        first_gist_metrics: dict[str, torch.Tensor] = {}
        if starts.any():
            start_ids = flat_input_ids[starts]
            for k in range(1, max_eos + 1):
                tok_id = tokenizer.convert_tokens_to_ids(f"<end_of_sentence_0_{k}>")
                count_k = (start_ids == int(tok_id)).sum() if tok_id is not None else torch.tensor(0, device=start_ids.device)
                first_gist_metrics[f"grpo_first_gist_token_{k}_percent"] = (
                    count_k.float() * 100 / (num_start_gist_tokens_items_in_batch * num_samples)
                ).detach()
        else:
            for k in range(1, max_eos + 1):
                first_gist_metrics[f"grpo_first_gist_token_{k}_percent"] = torch.tensor(
                    0.0, device=loss.device, dtype=torch.float32
                )

        # breakpoint()
        metrics = {
            "lm_loss": lm_loss_value,
            "starts": denom_gist.sum(),
            "grpo_loss": grpo_loss_per_example.detach().mean(),
            "grpo_better_than_baseline_percent": better_than_baseline_percent.detach(),
            "gist_tokens_percent": gist_tokens_percent.detach(),
            "grpo_reward_mean": (reward_grouped.detach().mean() / self.args.gradient_accumulation_steps),
            "grpo_reward_std": (reward_grouped.detach().std(unbiased=False) / self.args.gradient_accumulation_steps),
            "grpo_compression_gain_mean": (compression_gain.detach().mean() / self.args.gradient_accumulation_steps),
            "grpo_non_gist_delta_mean": (non_gist_delta.detach().mean() / self.args.gradient_accumulation_steps),
        }
        metrics.update(first_gist_metrics)

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
            self.args.device
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
