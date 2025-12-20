import time

import torch
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model


class GRPOTrainer(Trainer):

    def sample_gist_tokens_for_input(self, model, inputs):

        input_ids = inputs["input_ids"].cpu()
        attention_masks_length = inputs["attention_mask"].sum(dim=-1).cpu()
        assert (inputs["attention_mask"][:, 0] == 1).all(), "input must be right padded"
        for batch_i in range(inputs.shape[0]):
            # seq length
            input_ids = input_ids[batch_i, : attention_masks_length[batch_i]]

        return None

    def prepare_model_before_training(self, model, tokenizer):

        # Add separate tokens with different sequence length
        additional_special_tokens = []
        for num_gist_tokens in range(1, tokenizer.num_eos_tokens + 1):
            for i in range(num_gist_tokens):
                additional_special_tokens.append(f"<end_of_sentence_{i}_{num_gist_tokens}>")

        print("additional_special_tokens", additional_special_tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        # Initialize new embeddings with pretrained gist embeddings values
        base_embedding_weight = model.get_input_embeddings().weight
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        for num_gist_tokens in range(1, tokenizer.num_eos_tokens + 1):
            for i in range(num_gist_tokens):
                new_gist_token = f"<end_of_sentence_{i}_{num_gist_tokens}>"
                orig_gist_token = f"<end_of_sentence_{i}>"
                gist_token_id = tokenizer.convert_tokens_to_ids([new_gist_token])[0]
                orig_gist_token_id = tokenizer.convert_tokens_to_ids([orig_gist_token])[0]
                print(f"Initialize {new_gist_token} with {orig_gist_token}")
                base_embedding_weight[gist_token_id] = base_embedding_weight[orig_gist_token_id]

        # TODO also reinitialize LM Head for untied LMs
        if not model.config.tie_word_embeddings:
            raise ValueError("Not implemented model preparation for non tie_word_embeddings models")

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
