import time

import torch
from transformers import Trainer

from typing import Any

from transformers.trainer_pt_utils import distributed_concat


class GRPOTrainer(Trainer):

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
            sampled_inputs = unwrapped_model.sample_gist_tokens_for_input(
                tokenizer=self.processing_class,
                inputs=base_inputs,
                num_generate_samples=self.args.grpo_num_generate_samples,
                add_baseline=self.args.grpo_loss_alpha > 0,
                sampling_temperature=getattr(self.args, "grpo_sampling_temperature", 0.8),
                sampling_top_p=getattr(self.args, "grpo_sampling_top_p", 0.95),
                pretrain_sampling_steps=getattr(self.args, "grpo_pretrain_sampling_steps", 0),
                global_step=int(getattr(getattr(self, "state", None), "global_step", 0)),
                greedy=self.args.grpo_loss_alpha == 0,
            )
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

        loss = lm_loss.clone()

        grpo_loss_per_example_log = 0
        better_than_baseline_percent = 0

        gist_tokens = (flat_special_mask.bool() & flat_attention_mask.bool()).sum(dim=1)  # [N*B]
        total_tokens = flat_attention_mask.sum()
        gist_tokens_all = gist_tokens.sum()
        gist_tokens_percent = (gist_tokens_all.float() * 100 / total_tokens).detach()

        if self.args.grpo_loss_alpha > 0:
            # Group rewards/losses per original example, and take reward-weighted LM loss.
            # Compute non-gist causal LM loss (quality proxy) and shape reward as compression_gain - beta * penalty.
            # Baseline sample is expected to be sample index 0 (see sample_gist_tokens_for_input).
            target_is_special = flat_special_mask[:, 1:].bool()  # [N*B, L-1]
            non_gist_target_mask = (~target_is_special) & target_is_valid  # [N*B, L-1]
            non_gist_denom = non_gist_target_mask.sum(dim=1).clamp(min=1)
            non_gist_causal_lm_loss = (token_losses.to(torch.float32) * non_gist_target_mask).sum(
                dim=1
            ) / non_gist_denom  # [N*B]

            non_gist_loss_grouped = non_gist_causal_lm_loss.view(num_samples, batch_size)  # [N, B]
            baseline_non_gist_loss = non_gist_loss_grouped[0]  # [B]
            better_than_baseline = non_gist_loss_grouped < baseline_non_gist_loss.unsqueeze(0)  # [N, B]
            better_than_baseline[0, :] = False
            better_than_baseline_percent = (
                100
                * better_than_baseline.sum()
                / (better_than_baseline.numel() * self.accelerator.num_processes * self.args.gradient_accumulation_steps)
            ).detach()

            # Compression signal: fewer gist tokens is better (baseline has all original gist tokens).
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

            grpo_loss_per_example = (advantage.detach() * action_nll_grouped).mean(dim=1) * float(
                self.args.grpo_loss_alpha
            )  # [B]
            grpo_loss_per_example_log = grpo_loss_per_example.detach().mean()

            if use_model_based_sampling and not (reward_std <= 1e-8).all():
                loss += grpo_loss_per_example.mean()

        if torch.isnan(loss):
            print("Found nan loss")
            breakpoint()

        # Mirror legacy metric naming.
        lm_loss_value = lm_loss.detach()

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
            loss *= self.accelerator.num_processes
            # loss /= self.accelerator.num_processes
            # lm_loss_value /= self.accelerator.num_processes
            # grpo_loss_per_example /= self.accelerator.num_processes

        metrics = {
            "lm_loss": lm_loss_value,
            "starts": denom_gist.sum(),
            "gist_tokens_percent": gist_tokens_percent / self.args.gradient_accumulation_steps,
        }

        if self.args.grpo_loss_alpha > 0:
            metrics.update(
                {
                    "grpo_loss": grpo_loss_per_example_log,
                    "grpo_better_than_baseline_percent": better_than_baseline_percent,
                    "grpo_reward_mean": reward_grouped.detach().mean()
                    / self.args.gradient_accumulation_steps
                    / self.accelerator.num_processes,
                    "grpo_reward_std": reward_grouped.detach().std(unbiased=False)
                    / self.args.gradient_accumulation_steps
                    / self.accelerator.num_processes,
                    "grpo_compression_gain_mean": (compression_gain.detach().mean() / self.args.gradient_accumulation_steps),
                    "grpo_non_gist_delta_mean": (non_gist_delta.detach().mean() / self.args.gradient_accumulation_steps),
                }
            )

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
