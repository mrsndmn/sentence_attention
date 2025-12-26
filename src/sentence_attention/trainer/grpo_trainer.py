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

        Always returns one "baseline" sample at index 0 (no compression: k == run_len for every run),
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

        # Anneal sampling strategy:
        # - For the first `grpo_pretrain_sampling_steps`, we still generate multiple samples, but sample k uniformly.
        # - After that, we sample k using the model's logits (optionally with nucleus top-p).
        pretrain_steps = int(getattr(self.args, "grpo_pretrain_sampling_steps", 0))
        if pretrain_steps < 0:
            pretrain_steps = 0
        global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
        model_based_sampling_requested = global_step >= pretrain_steps

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

        # eos_token_tensors_by_k[k] holds a 1D tensor of length k:
        #   <end_of_sentence_0_{k}>, <end_of_sentence_1_{k}>, ..., <end_of_sentence_{k-1}_{k}>
        eos_token_tensors_by_k: list[torch.Tensor] = []
        eos_spec_tensors_by_k: list[torch.Tensor] = []
        first_token_id_by_k: list[int] = [0]
        for k in range(max_eos):
            ids_k = [_resolve_token_id(f"<end_of_sentence_{i}_{k}>") for i in range(k + 1)]
            eos_token_tensors_by_k.append(torch.tensor(ids_k, device=input_ids.device, dtype=input_ids.dtype))
            eos_spec_tensors_by_k.append(torch.ones((k + 1,), device=input_ids.device, dtype=special_embeddings_mask.dtype))
            first_token_id_by_k.append(int(ids_k[0]))

        first_token_ids_by_k = torch.tensor(first_token_id_by_k, device=input_ids.device, dtype=torch.long)

        def _sample_k_uniform(run_len: int) -> int:
            k_max = max_eos
            return int(torch.randint(low=0, high=k_max, size=(1,), device="cpu").item())

        def _sample_k_from_logits(next_logits: torch.Tensor, run_len: int) -> int:
            # Sample k in [1, min(run_len, max_eos)] based on logits for <end_of_sentence_0_{k}>.
            k_max = int(min(run_len, max_eos))
            if k_max <= 0:
                raise ValueError(f"Invalid run_len={run_len} for gist sampling.")

            candidate_ids = first_token_ids_by_k[1 : k_max + 1]  # ids for k=1..k_max
            cand_logits = next_logits.index_select(0, candidate_ids).to(torch.float32) / temperature
            probs = torch.softmax(cand_logits, dim=-1)

            sampled_pos = int(torch.multinomial(probs.detach().cpu(), num_samples=1).item())
            return int(sampled_pos)

        model_based_sampling_enabled = (
            bool(model_based_sampling_requested) and callable(model) and hasattr(model, "eval") and hasattr(model, "train")
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

        def _next_logits_for_prefix(prefix_ids_1d: torch.Tensor, prefix_spec_1d: torch.Tensor) -> torch.Tensor | None:
            """
            Full-prefill forward (no KV-cache): return logits for the next token given the provided prefix.
            """
            if prefix_ids_1d.numel() == 0:
                return None

            prefix_len = int(prefix_ids_1d.numel())
            attention = torch.ones((1, prefix_len), device=input_ids.device, dtype=torch.long)
            special = prefix_spec_1d.to(dtype=special_embeddings_mask.dtype).unsqueeze(0)
            closest = special_token_mask_to_clothest_token_idx_slow(prefix_spec_1d.to(torch.bool).unsqueeze(0))

            outputs = model(
                fused_linear_cross_entropy=False,
                labels=None,
                input_ids=prefix_ids_1d.unsqueeze(0),
                attention_mask=attention,
                special_embeddings_mask=special,
                clothest_end_of_sentence_token_idx=closest,
                use_cache=False,
                logits_to_keep=1,
                output_attentions=False,
            )
            if getattr(outputs, "logits", None) is None:
                return None
            return outputs.logits[0, -1, :].detach()

        def _build_sample(*, kind: str) -> dict:
            if kind not in {"baseline", "sample"}:
                raise ValueError(f"Invalid sample kind: {kind}")

            new_input_ids = input_ids.new_full((batch_size, seq_len), int(pad_token_id))
            new_attention_mask = attention_mask.new_zeros((batch_size, seq_len))
            new_special_mask = special_embeddings_mask.new_zeros((batch_size, seq_len))

            was_training = model.training

            if model_based_sampling_enabled:
                model.eval()

            with torch.no_grad():
                for b in range(batch_size):
                    valid_len = int(attention_mask[b].sum().item())

                    ids_1d = input_ids[b, :valid_len]
                    spec_1d = special_embeddings_mask[b, :valid_len].to(torch.bool)

                    runs = _gist_runs_1d(spec_1d)

                    prefix_id_parts: list[torch.Tensor] = []
                    prefix_spec_parts: list[torch.Tensor] = []

                    pos = 0
                    for run_start, run_end in runs:
                        if run_start > pos:
                            seg_ids = ids_1d[pos:run_start]
                            seg_len = int(seg_ids.numel())
                            seg_spec = torch.zeros(
                                (seg_len,),
                                device=input_ids.device,
                                dtype=special_embeddings_mask.dtype,
                            )

                            prefix_id_parts.append(seg_ids)
                            prefix_spec_parts.append(seg_spec)

                        run_len = int(run_end - run_start)
                        if kind == "baseline":
                            k = max_eos - 1
                        else:
                            if model_based_sampling_enabled:
                                prefix_ids = torch.cat(prefix_id_parts, dim=0)
                                prefix_spec = torch.cat(prefix_spec_parts, dim=0)
                                next_logits = _next_logits_for_prefix(prefix_ids, prefix_spec)
                                k = _sample_k_from_logits(next_logits, run_len=run_len)
                            else:
                                k = _sample_k_uniform(run_len=run_len)

                        tok_ids = eos_token_tensors_by_k[k]
                        tok_spec = eos_spec_tensors_by_k[k]

                        prefix_id_parts.append(tok_ids)
                        prefix_spec_parts.append(tok_spec)

                        pos = run_end

                    if len(runs) == 0:
                        prefix_id_parts.append(ids_1d)
                        prefix_spec_parts.append(spec_1d)

                    out_ids = torch.cat(prefix_id_parts, dim=0)
                    out_spec = torch.cat(prefix_spec_parts, dim=0).to(dtype=special_embeddings_mask.dtype)

                    current_seq_len = min(out_ids.numel(), new_input_ids.shape[-1])

                    new_input_ids[b, :current_seq_len] = out_ids[:current_seq_len]
                    new_special_mask[b, :current_seq_len] = out_spec[:current_seq_len]
                    new_attention_mask[b, :current_seq_len] = 1

                    if out_ids.numel() >= new_input_ids.shape[-1]:
                        continue

            if model_based_sampling_enabled:
                model.train(was_training)

            result = dict(inputs)
            result["input_ids"] = new_input_ids
            result["attention_mask"] = new_attention_mask
            result["special_embeddings_mask"] = new_special_mask
            result["clothest_end_of_sentence_token_idx"] = special_token_mask_to_clothest_token_idx_slow(
                new_special_mask.to(torch.bool) & new_attention_mask.to(torch.bool)
            )

            result["labels"] = new_input_ids.clone()
            result["labels"][:, :-1] = result["labels"][:, 1:].clone()
            result["labels"][new_attention_mask == 0] = -100

            return result

        orig_attn_impl = None
        model_config = model.config
        orig_attn_impl = model_config._attn_implementation
        model_config._attn_implementation = "sentence_attention"

        num_generate_samples = self.args.grpo_num_generate_samples
        if num_generate_samples <= 0:
            raise ValueError("grpo_num_generate_samples must be > 0")

        # Always return a baseline at index 0 (see compute_loss which assumes this).
        result = [_build_sample(kind="baseline")]
        result.extend(_build_sample(kind="sample") for _ in range(num_generate_samples - 1))

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
        with torch.no_grad():
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

        token_losses = token_losses.to(torch.float32)

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
        non_gist_causal_lm_loss = (token_losses.to(torch.float32) * non_gist_target_mask).sum(dim=1) / non_gist_denom  # [N*B]

        non_gist_loss_grouped = non_gist_causal_lm_loss.view(num_samples, batch_size)  # [N, B]
        baseline_non_gist_loss = non_gist_loss_grouped[0]  # [B]
        better_than_baseline = non_gist_loss_grouped < baseline_non_gist_loss.unsqueeze(0)  # [N, B]
        better_than_baseline[0, :] = False
        better_than_baseline_percent = (
            100
            * better_than_baseline.sum()
            / (better_than_baseline.numel() * self.accelerator.num_processes * self.args.gradient_accumulation_steps)
        )

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

        loss = lm_loss.clone()
        if use_model_based_sampling and not (reward_std <= 1e-8).all():
            loss += grpo_loss_per_example.mean()

        if torch.isnan(loss):
            print("Found nan loss")
            breakpoint()

        # Mirror legacy metric naming.
        lm_loss_value = lm_loss.detach()

        total_tokens = flat_attention_mask.sum()
        gist_tokens_all = gist_tokens.sum()
        gist_tokens_percent = gist_tokens_all.float() * 100 / total_tokens

        # Distribution of "first gist token" types (<end_of_sentence_0_{k}>) as a percent of ALL gist tokens.
        # We define "first gist token" as the first token in each contiguous gist-run in the sampled sequence.
        tokenizer = self.processing_class
        max_eos: int = tokenizer.num_eos_tokens

        denom_gist = starts.sum().float().clamp(min=1.0)
        first_gist_metrics: dict[str, torch.Tensor] = {}
        if starts.any():
            start_ids = flat_input_ids[starts]
            for k, start_eos_token_id in enumerate(tokenizer.start_end_of_sentence_token_ids):
                count_k = (start_ids == int(start_eos_token_id)).sum()
                first_gist_metrics[f"grpo_first_gist_token_{k}_percent"] = (
                    count_k.float() * 100 / (num_start_gist_tokens_items_in_batch * num_samples)
                ).detach()
        else:
            for k in range(max_eos):
                first_gist_metrics[f"grpo_first_gist_token_{k}_percent"] = torch.tensor(
                    0.0, device=loss.device, dtype=torch.float32
                )

        if self.args.average_tokens_across_devices:
            loss /= self.accelerator.num_processes
            lm_loss_value /= self.accelerator.num_processes
            grpo_loss_per_example /= self.accelerator.num_processes

        metrics = {
            "lm_loss": lm_loss_value,
            "starts": denom_gist.sum(),
            "grpo_loss": grpo_loss_per_example.detach().mean(),
            "grpo_better_than_baseline_percent": better_than_baseline_percent.detach(),
            "gist_tokens_percent": gist_tokens_percent.detach() / self.args.gradient_accumulation_steps,
            "grpo_reward_mean": reward_grouped.detach().mean()
            / self.args.gradient_accumulation_steps
            / self.accelerator.num_processes,
            "grpo_reward_std": reward_grouped.detach().std(unbiased=False)
            / self.args.gradient_accumulation_steps
            / self.accelerator.num_processes,
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
