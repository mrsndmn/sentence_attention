import torch
from peft import PeftConfig, PeftModel
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM


def load_model_from_checkpoint(checkpoint_path, attention_implementation=None):
    if "lora" in checkpoint_path:
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        base_model, tokenizer = _load_model_from_checkpoint(
            peft_config.base_model_name_or_path, attention_implementation=attention_implementation
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        model, tokenizer = _load_model_from_checkpoint(checkpoint_path, attention_implementation=attention_implementation)

    return model, tokenizer


def _load_model_from_checkpoint(checkpoint_path, attention_implementation=None):

    config = AutoConfig.from_pretrained(checkpoint_path)
    model_class_name = config.architectures[0]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if hasattr(tokenizer, "num_eos_tokens"):
        print(
            "tokenizer num_eos_tokens",
            tokenizer.num_eos_tokens,
            "end_of_sentence_token_ids",
            tokenizer.end_of_sentence_token_ids,
        )
    else:
        print("tokenizer does not have num_eos_tokens", type(tokenizer))

    if model_class_name == "SentenceLlamaForCausalLM":
        model_class = SentenceLlamaForCausalLM
    elif model_class_name == "SentenceQwen2ForCausalLM":
        model_class = SentenceQwen2ForCausalLM
    elif model_class_name == "LlamaForCausalLM":
        model_class = LlamaForCausalLM
    elif model_class_name == "Qwen2ForCausalLM":
        model_class = Qwen2ForCausalLM
    elif model_class_name == "LlamaAutoCompressorModel":
        from auto_compressor import LlamaAutoCompressorModel

        model_class = LlamaAutoCompressorModel
    else:
        raise ValueError(f"Model class {model_class_name} not supported")

    model = model_class.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()

    if attention_implementation is not None:
        model.config._attn_implementation = attention_implementation

    print("model", model, "attention_implementation", attention_implementation)

    return model, tokenizer
