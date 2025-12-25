import time

import torch
from transformers import Trainer

from typing import Any

from transformers.trainer_pt_utils import distributed_concat

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
)


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

        temperature = float(getattr(self.args, "grpo_sampling_temperature", 0.8))
        top_p = float(getattr(self.args, "grpo_sampling_top_p", 0.95))
        if temperature <= 0:
            raise ValueError("grpo_sampling_temperature must be > 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("grpo_sampling_top_p must be in (0, 1]")

        # Anneal sampling strategy: start with uniform sampling for a few steps, then switch to logits-based sampling.
        pretrain_steps = int(getattr(self.args, "grpo_pretrain_sampling_steps", 0))
        if pretrain_steps < 0:
            pretrain_steps = 0
        global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
        use_model_based_sampling = global_step >= pretrain_steps

        def _resolve_token_id(token_str: str) -> int:
            tok_id = tokenizer.convert_tokens_to_ids(token_str)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if tok_id is None or (
                unk_id is not None and tok_id == unk_id and token_str != getattr(tokenizer, "unk_token", None)
            ):
                raise ValueError(f"Token id not found for {token_str}. Did you call prepare_model_before_training?")
            return int(tok_id)

        # Precompute ids for variable-length gist tokens once (avoid tokenizer lookups in the hot loop).
        max_eos = int(getattr(tokenizer, "num_eos_tokens", 0) or 0)
        if max_eos <= 0:
            raise ValueError("Tokenizer must expose num_eos_tokens > 0 for GRPO gist sampling.")

        # eos_token_ids_by_k[k] -> list of ids for <end_of_sentence_{i}_{k}> (i=0..k-1). Index 0 unused.
        eos_token_ids_by_k: list[list[int]] = [[]]
        first_token_ids_all: list[int] = []
        for k in range(max_eos):
            ids_k: list[int] = []
            for i in range(k + 1):
                ids_k.append(_resolve_token_id(f"<end_of_sentence_{i}_{k}>"))
            eos_token_ids_by_k.append(ids_k)
            first_token_ids_all.append(ids_k[0])

        first_token_ids_tensor = torch.tensor(first_token_ids_all, device=input_ids.device, dtype=torch.long)
        eos_token_tensors_by_k: list[torch.Tensor] = [torch.empty((0,), device=input_ids.device, dtype=input_ids.dtype)]
        eos_spec_tensors_by_k: list[torch.Tensor] = [
            torch.empty((0,), device=input_ids.device, dtype=special_embeddings_mask.dtype)
        ]
        for k in range(max_eos):
            eos_token_tensors_by_k.append(torch.tensor(eos_token_ids_by_k[k], device=input_ids.device, dtype=input_ids.dtype))
            eos_spec_tensors_by_k.append(torch.ones((k,), device=input_ids.device, dtype=special_embeddings_mask.dtype))

        def _sample_k(next_logits: torch.Tensor | None, run_len: int) -> int:
            # Sample k in [1, min(run_len, max_eos)].
            k_max = int(min(run_len, max_eos))
            if k_max <= 0:
                raise ValueError(f"Invalid run_len={run_len} for gist sampling.")

            if next_logits is None:
                # Uniform over k.
                return int(torch.randint(low=1, high=k_max + 1, size=(1,), device="cpu").item())

            candidate_ids = first_token_ids_tensor[:k_max]  # <end_of_sentence_0_{k}> ids
            cand_logits = next_logits.index_select(0, candidate_ids).to(torch.float32) / temperature
            probs = torch.softmax(cand_logits, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            if float(probs.sum().detach().cpu().item()) <= 0.0:
                return int(torch.randint(low=1, high=k_max + 1, size=(1,), device="cpu").item())

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
                sampled_rel = int(torch.randint(low=0, high=int(kept_idx.numel()), size=(1,), device="cpu").item())
                sampled_pos = int(kept_idx[sampled_rel].item())
                return int(sampled_pos + 1)

            kept_probs = (kept_probs / kept_sum).to(torch.float32)
            sampled_rel = int(torch.multinomial(kept_probs.detach().cpu(), num_samples=1).item())
            sampled_pos = int(kept_idx[sampled_rel].item())
            return int(sampled_pos + 1)

        supports_model_based_sampling = (
            bool(use_model_based_sampling)
            and callable(model)
            and hasattr(model, "eval")
            and hasattr(model, "train")
            and hasattr(model, "training")
        )

        def _gist_runs_1d(spec_1d: torch.Tensor) -> list[tuple[int, int]]:
            # spec_1d: [L] bool. Returns (start, end) with end exclusive.
            if spec_1d.numel() == 0 or not bool(spec_1d.any().item()):
                return []
            prev = torch.zeros_like(spec_1d)
            prev[1:] = spec_1d[:-1]
            starts = spec_1d & (~prev)
            nxt = torch.zeros_like(spec_1d)
            nxt[:-1] = spec_1d[1:]
            ends = spec_1d & (~nxt)
            start_idx = starts.nonzero(as_tuple=False).flatten().tolist()
            end_idx = ends.nonzero(as_tuple=False).flatten().tolist()
            if len(start_idx) != len(end_idx):
                raise ValueError("Invalid gist run structure (mismatched starts/ends).")
            return [(int(s), int(e) + 1) for s, e in zip(start_idx, end_idx)]

        def sample_one() -> dict:
            new_input_ids = input_ids.new_full((batch_size, seq_len), int(pad_token_id))
            new_attention_mask = attention_mask.new_zeros((batch_size, seq_len))
            new_special_mask = special_embeddings_mask.new_zeros((batch_size, seq_len))

            new_labels = None
            if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor) and inputs["labels"].shape == input_ids.shape:
                new_labels = inputs["labels"].new_full((batch_size, seq_len), -100)

            was_training = bool(getattr(model, "training", False))
            try:
                if supports_model_based_sampling:
                    model.eval()
                with torch.no_grad():
                    for b in range(batch_size):
                        valid_len = int(attention_mask[b].sum().item())
                        if valid_len <= 0:
                            continue

                        ids_1d = input_ids[b, :valid_len]
                        spec_1d = special_embeddings_mask[b, :valid_len].to(torch.bool)
                        lbls_1d = inputs["labels"][b, :valid_len] if new_labels is not None else None

                        runs = _gist_runs_1d(spec_1d)

                        out_id_parts: list[torch.Tensor] = []
                        out_spec_parts: list[torch.Tensor] = []
                        out_lbl_parts: list[torch.Tensor] | None = [] if lbls_1d is not None else None

                        # Incremental context for model-based sampling.
                        past = None
                        fed_spec = torch.empty((0,), device=input_ids.device, dtype=torch.long)
                        next_logits: torch.Tensor | None = None
                        pending_id_parts: list[torch.Tensor] = []
                        pending_spec_parts: list[torch.Tensor] = []

                        def flush_pending() -> None:
                            nonlocal past, fed_spec, next_logits, pending_id_parts, pending_spec_parts
                            if not supports_model_based_sampling or len(pending_id_parts) == 0:
                                pending_id_parts = []
                                pending_spec_parts = []
                                return

                            chunk_ids = torch.cat(pending_id_parts, dim=0).unsqueeze(0)
                            chunk_spec = torch.cat(pending_spec_parts, dim=0)
                            total_spec = torch.cat([fed_spec, chunk_spec], dim=0)
                            total_len = int(total_spec.numel())
                            attention = torch.ones((1, total_len), device=input_ids.device, dtype=torch.long)
                            special = total_spec.unsqueeze(0)
                            closest = special_token_mask_to_clothest_token_idx_slow(special.to(torch.bool))

                            outputs = model(
                                fused_linear_cross_entropy=False,
                                labels=None,
                                input_ids=chunk_ids,
                                attention_mask=attention,
                                special_embeddings_mask=special.to(dtype=special_embeddings_mask.dtype),
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
                            pending_id_parts = []
                            pending_spec_parts = []

                        pos = 0
                        for run_start, run_end in runs:
                            if run_start > pos:
                                seg_ids = ids_1d[pos:run_start]
                                seg_len = int(seg_ids.numel())
                                seg_spec = torch.zeros((seg_len,), device=input_ids.device, dtype=torch.long)

                                out_id_parts.append(seg_ids)
                                out_spec_parts.append(seg_spec)
                                if out_lbl_parts is not None and lbls_1d is not None:
                                    out_lbl_parts.append(lbls_1d[pos:run_start])
                                pending_id_parts.append(seg_ids)
                                pending_spec_parts.append(seg_spec)

                            flush_pending()

                            run_len = int(run_end - run_start)
                            k = _sample_k(next_logits if supports_model_based_sampling else None, run_len=run_len)
                            tok_ids = eos_token_tensors_by_k[k]
                            tok_spec = eos_spec_tensors_by_k[k]

                            out_id_parts.append(tok_ids)
                            out_spec_parts.append(tok_spec)
                            if out_lbl_parts is not None:
                                out_lbl_parts.append(tok_ids.to(dtype=inputs["labels"].dtype))

                            pending_id_parts.append(tok_ids)
                            pending_spec_parts.append(tok_spec)
                            flush_pending()
                            pos = run_end

                        if pos < valid_len:
                            seg_ids = ids_1d[pos:valid_len]
                            seg_len = int(seg_ids.numel())
                            seg_spec = torch.zeros((seg_len,), device=input_ids.device, dtype=torch.long)

                            out_id_parts.append(seg_ids)
                            out_spec_parts.append(seg_spec)
                            if out_lbl_parts is not None and lbls_1d is not None:
                                out_lbl_parts.append(lbls_1d[pos:valid_len])

                        if len(out_id_parts) == 0:
                            out_ids = ids_1d.new_empty((0,))
                            out_spec = special_embeddings_mask.new_empty((0,), dtype=torch.long)
                            out_lbl = None
                        else:
                            out_ids = torch.cat(out_id_parts, dim=0)
                            out_spec = torch.cat(out_spec_parts, dim=0).to(dtype=special_embeddings_mask.dtype)
                            out_lbl = torch.cat(out_lbl_parts, dim=0) if out_lbl_parts is not None else None

                        if out_ids.numel() > seq_len:
                            out_ids = out_ids[:seq_len]
                            out_spec = out_spec[:seq_len]
                            if out_lbl is not None:
                                out_lbl = out_lbl[:seq_len]

                        new_input_ids[b, : out_ids.numel()] = out_ids
                        new_special_mask[b, : out_spec.numel()] = out_spec
                        new_attention_mask[b] = (new_input_ids[b] != int(pad_token_id)).to(dtype=new_attention_mask.dtype)
                        if new_labels is not None and out_lbl is not None:
                            new_labels[b, : out_lbl.numel()] = out_lbl.to(dtype=new_labels.dtype)
            finally:
                if supports_model_based_sampling:
                    model.train(was_training)

            result = dict(inputs)
            result["input_ids"] = new_input_ids
            result["attention_mask"] = new_attention_mask
            result["special_embeddings_mask"] = new_special_mask
            result["clothest_end_of_sentence_token_idx"] = special_token_mask_to_clothest_token_idx_slow(
                new_special_mask.to(torch.bool) & new_attention_mask.to(torch.bool)
            )
            if new_labels is not None:
                result["labels"] = new_labels
            return result

        orig_attn_impl = None
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "_attn_implementation"):
            orig_attn_impl = model_config._attn_implementation
            model_config._attn_implementation = "sentence_attention"

        num_generate_samples = int(getattr(self.args, "grpo_num_generate_samples", 1))
        if num_generate_samples <= 0:
            raise ValueError("grpo_num_generate_samples must be > 0")

        # Baseline sample: keep the original gist token placement unchanged (all gist tokens present).
        baseline = dict(inputs)
        if (
            "clothest_end_of_sentence_token_idx" not in baseline
            or not isinstance(baseline["clothest_end_of_sentence_token_idx"], torch.Tensor)
            or baseline["clothest_end_of_sentence_token_idx"].shape != input_ids.shape
        ):
            baseline["clothest_end_of_sentence_token_idx"] = special_token_mask_to_clothest_token_idx_slow(
                baseline["special_embeddings_mask"].to(torch.bool) & baseline["attention_mask"].to(torch.bool)
            )

        if use_model_based_sampling:
            result = [baseline]
            result.extend(sample_one() for _ in range(num_generate_samples - 1))
        else:
            result = [sample_one()]

        if orig_attn_impl is not None and model_config is not None:
            model_config._attn_implementation = orig_attn_impl
        # torch.cuda.empty_cache()

        return result

    @staticmethod
    def prepare_model_before_training(model, tokenizer):
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

        global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
        use_model_based_sampling = global_step >= self.args.grpo_pretrain_sampling_steps
        if use_model_based_sampling and bool(self.args.grpo_supervised_exclude_action_tokens):
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

        # loss = 0
        if use_model_based_sampling and not (reward_std <= 1e-8).all():
            loss = grpo_loss_per_example.mean()
        else:
            loss = lm_loss

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

        if self.args.average_tokens_across_devices:
            loss *= self.accelerator.num_processes
            lm_loss_value *= self.accelerator.num_processes
            grpo_loss_per_example *= self.accelerator.num_processes

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
