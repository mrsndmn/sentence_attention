import argparse

from sentence_attention.evaluation.evaluation import evaluate_lighteval_task_save_results
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM
from transformers import AutoConfig, AutoTokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.checkpoint)

    model_class_name = config.architectures[0]

    if model_class_name == "SentenceLlamaForCausalLM":
        model_class = SentenceLlamaForCausalLM
    elif model_class_name == "SentenceQwen2ForCausalLM":
        model_class = SentenceQwen2ForCausalLM
    else:
        raise ValueError(f"Model class {model_class_name} not supported")

    model = model_class.from_pretrained(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model.eval()

    evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark)
