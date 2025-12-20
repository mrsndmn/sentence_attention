import time

import torch
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model


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

    def prepare_model_before_training(self, model, tokenizer):

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

        # if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:

        labels = inputs.pop("labels")

        unwrapped_model = self.accelerator.unwrap_model(model)

        # Optionally disable loss on EOS tokens when using multiple EOS tokens
        special_embeddings_mask = inputs.get("special_embeddings_mask")

        attention_mask = inputs["attention_mask"]
        # token_frequency = inputs.get('token_frequency', None)
        model_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_attentions": False,
        }

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            model_kwargs = {**model_kwargs, **loss_kwargs}

        assert special_embeddings_mask is not None
        model_kwargs["special_embeddings_mask"] = special_embeddings_mask

        model_kwargs["clothest_end_of_sentence_token_idx"] = inputs["clothest_end_of_sentence_token_idx"]

        assert special_embeddings_mask.shape == attention_mask.shape

        fused_linear_cross_entropy = True

        torch.compiler.cudagraph_mark_step_begin()
        # breakpoint()
        # print("model_kwargs input ids", model_kwargs["input_ids"])
        outputs = model(
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            labels=labels,
            **model_kwargs,
        )
        # [ bs, seq_len, 2 ]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if fused_linear_cross_entropy:
            loss = outputs.loss
        elif labels is not None and self.label_smoother is not None or self.compute_loss_func is not None:
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(
                    outputs.logits,
                    labels,
                    vocab_size=unwrapped_model.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                )
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        moe_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        lm_loss = loss.detach()
        if self.args.moe_special_embeddings_layer_idx is not None and outputs.moe_aux_loss is not None:
            # Ensure moe_aux_loss is on the same device as loss
            moe_aux_loss = outputs.moe_aux_loss
            if isinstance(moe_aux_loss, torch.Tensor):
                moe_aux_loss = moe_aux_loss.to(device=loss.device, dtype=loss.dtype)
            else:
                # Convert scalar to tensor if needed
                moe_aux_loss = torch.tensor(moe_aux_loss, device=loss.device, dtype=loss.dtype)
            moe_loss = moe_aux_loss * self.args.moe_aux_loss_weight
            loss += moe_loss

        if self.args.average_tokens_across_devices and (self.model_accepts_loss_kwargs or self.compute_loss_func):
            loss *= self.accelerator.num_processes

        outputs.loss = loss

        # Convert to scalars to avoid device issues during distributed gathering
        # The transformers library will handle scalar metrics properly
        lm_loss_value = lm_loss.detach()
        moe_loss_value = moe_loss.detach()

        total_tokens = attention_mask.sum().clamp(min=1)
        gist_tokens = (special_embeddings_mask.bool() & attention_mask.bool()).sum()
        gist_tokens_percent = (gist_tokens.float() / total_tokens.float()) * 100 / self.args.gradient_accumulation_steps

        metrics = {
            "lm_loss": lm_loss_value,
            "moe_loss": moe_loss_value,
            "gist_tokens_percent": gist_tokens_percent.detach(),
        }

        return (loss, outputs) if return_outputs else (loss, metrics)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):

        while True:
            try:
                super().save_model(output_dir, _internal_call)
                break
            except Exception as e:
                print("Error in saving model", e)
                time.sleep(300)
