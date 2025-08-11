import argparse

from peft import PeftConfig, PeftModel
from sentence_attention.evaluation.benchmarks import all_benchmarks
from sentence_attention.evaluation.evaluation import evaluate_lighteval_task, evaluate_lighteval_task_save_results
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM
from transformers import AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM


def load_model_from_checkpoint(checkpoint_path):

    config = AutoConfig.from_pretrained(checkpoint_path)
    model_class_name = config.architectures[0]

    if model_class_name == "SentenceLlamaForCausalLM":
        model_class = SentenceLlamaForCausalLM
    elif model_class_name == "SentenceQwen2ForCausalLM":
        model_class = SentenceQwen2ForCausalLM
    elif model_class_name == "LlamaForCausalLM":
        model_class = LlamaForCausalLM
    elif model_class_name == "Qwen2ForCausalLM":
        model_class = Qwen2ForCausalLM
    else:
        raise ValueError(f"Model class {model_class_name} not supported")

    model = model_class.from_pretrained(checkpoint_path)
    model.eval()

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True, choices=all_benchmarks)
    parser.add_argument("--no-save-results", action="store_true", default=False)

    args = parser.parse_args()

    if "lora" in args.checkpoint:
        peft_config = PeftConfig.from_pretrained(args.checkpoint)
        base_model = load_model_from_checkpoint(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model = load_model_from_checkpoint(args.checkpoint)

    if args.no_save_results:
        print("Evaluating without saving results")
        results = evaluate_lighteval_task(model, args.benchmark)
    else:
        results = evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark)

    print(results)
