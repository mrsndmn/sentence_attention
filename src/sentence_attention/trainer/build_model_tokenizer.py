import torch
import torch.nn as nn
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import (
    GPT2TokenizerFastEOS,
)
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
)
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import (
    Qwen2TokenizerFastEOS,
)
from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from sentence_attention.trainer.arguments import (
    AVAILABLE_OPTIMIZED_PARAMS,
    SentenceTrainingArguments,
)
from transformers import AutoTokenizer


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def build_model_tokenizer(training_args: SentenceTrainingArguments):
    tokenizer = None
    model_checkpoint = training_args.model_checkpoint

    number_of_eos_tokens = training_args.number_of_eos_tokens

    if training_args.add_end_of_sentence_token:

        tokenizer_class = type(AutoTokenizer.from_pretrained(model_checkpoint)).__name__

        if tokenizer_class in ["GPT2TokenizerFast", "GPT2TokenizerFastEOS"]:
            tokenizer_class = GPT2TokenizerFastEOS
        elif tokenizer_class in ["PreTrainedTokenizerFast", "PreTrainedTokenizerFastEOS"]:
            tokenizer_class = PreTrainedTokenizerFastEOS
        elif tokenizer_class in ["Qwen2TokenizerFast", "Qwen2TokenizerFastEOS"]:
            tokenizer_class = Qwen2TokenizerFastEOS
        else:
            raise ValueError(f"Invalid tokenizer class: {tokenizer_class}")

        print("tokenizer_class", tokenizer_class, "number_of_eos_tokens", number_of_eos_tokens)
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint, num_eos_tokens=number_of_eos_tokens)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("num_eos_tokens", tokenizer.num_eos_tokens, "end_of_sentence_tokens_list", tokenizer.end_of_sentence_tokens_list)
    print("tokenizer", tokenizer)

    torch_dtype = torch.bfloat16

    if training_args.model_type == "sentence_pretrained_checkpoint":
        model_checkpoint = training_args.model_checkpoint
        print("Load sentence llama model from", model_checkpoint)
        model_class = None
        if "lama" in model_checkpoint.lower() or "smollm2" in model_checkpoint.lower():
            model_class = SentenceLlamaForCausalLM
        elif "qwen" in model_checkpoint.lower():
            model_class = SentenceQwen2ForCausalLM

        print("model_class", model_class)
        model = model_class.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"{training_args.model_type} is not supported")

    model.config._attn_implementation = training_args.sentence_attention_implementation

    if training_args.moe_special_embeddings_layer_idx is not None:
        model.config.moe_special_embeddings_layer_idx = training_args.moe_special_embeddings_layer_idx
        model.config.moe_num_experts = training_args.moe_num_experts
        torch.set_default_dtype(torch_dtype)
        model.init_gist_moe()
        torch.set_default_dtype(torch.float32)

    print("model.config._attn_implementation", model.config._attn_implementation)

    # tokenizer.padding_side = "right"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if training_args.add_end_of_sentence_token and model.config.vocab_size != len(tokenizer):
        # breakpoint()
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
        model.config.end_of_sentence_token_ids = tokenizer.end_of_sentence_token_ids
        print("model.config.end_of_sentence_token_ids", model.config.end_of_sentence_token_ids)

    model.config.flexible_eos_tokens = training_args.flexible_eos_tokens
    model.config.ft_with_bos_token = training_args.ft_with_bos_token

    if training_args.model_type == "sentence_pretrained_checkpoint":
        optimized_params = training_args.optimized_params
        print("optimized_params", optimized_params)

        assert any(
            param in AVAILABLE_OPTIMIZED_PARAMS for param in optimized_params.split(",")
        ), f"unknown optimized_params value: {optimized_params}. available ones: {AVAILABLE_OPTIMIZED_PARAMS}"

        if optimized_params == "full":
            pass

        if "only_eos_embedding" in optimized_params:
            freeze_model(model)
            for p in model.model.embed_tokens.parameters():
                p.requires_grad = True

            for p in model.lm_head.parameters():
                p.requires_grad = True

        if "lora" in optimized_params:
            from peft import LoraConfig, TaskType

            # create LoRA configuration object
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # type of task to train on
                inference_mode=False,  # set to False for training
                exclude_modules=["lm_head", "model.embed_tokens"],
                modules_to_save=["lm_head", "model.embed_tokens"],
                r=32,  # dimension of the smaller matrices
                lora_alpha=64,  # scaling factor
                lora_dropout=0.1,  # dropout of LoRA layers
            )
            model.add_adapter(lora_config, adapter_name="lora_1")

    # Always unfreeze MoE parameters if they exist
    if training_args.moe_special_embeddings_layer_idx is not None:
        if hasattr(model.model, "gist_moe") and model.model.gist_moe is not None:
            for p in model.model.gist_moe.parameters():
                p.requires_grad = True
            print("Unfrozen MoE parameters (gist_moe)")

    print("model", type(model))
    print("model", model)
    print("num trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("num freezed model parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad))

    return model, tokenizer
