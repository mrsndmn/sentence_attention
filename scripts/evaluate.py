import argparse
import os

from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM

from sentence_attention.evaluation.benchmarks import all_benchmarks
from sentence_attention.evaluation.evaluation import evaluate_lighteval_task, evaluate_lighteval_task_save_results
from sentence_attention.evaluation.pg19 import evaluate_pg19_ppl, save_pg19_results_json
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from sentence_attention.models.sentence_qwen2.modeling_sentence_qwen2 import SentenceQwen2ForCausalLM


def load_model_from_checkpoint(checkpoint_path):

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
    # PG19-specific optional args (ignored for other benchmarks)
    parser.add_argument(
        "--pg19-dataset-path",
        type=str,
        default="/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/pg19_test",
    )
    parser.add_argument("--pg19-model-type", type=str, choices=["sentence", "vanilla"], default="sentence")
    parser.add_argument("--pg19-max-samples", type=int, default=10)
    parser.add_argument("--pg19-max-length", type=int, default=1000)
    parser.add_argument("--pg19-device", type=str, default=None)
    parser.add_argument("--pg19-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")

    args = parser.parse_args()

    if "lora" in args.checkpoint:
        peft_config = PeftConfig.from_pretrained(args.checkpoint)
        base_model = load_model_from_checkpoint(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model = load_model_from_checkpoint(args.checkpoint)

    if args.benchmark == "pg19_ppl":
        results = evaluate_pg19_ppl(
            checkpoint_dir=args.checkpoint,
            dataset_path=args.pg19_dataset_path,
            model_type=args.pg19_model_type,
            max_samples=args.pg19_max_samples,
            max_length=args.pg19_max_length,
            device=args.pg19_device,
            dtype=args.pg19_dtype,
        )
        if not args.no_save_results:
            out_dir = os.path.join(args.checkpoint, "evaluation")
            os.makedirs(out_dir, exist_ok=True)
            out_path = save_pg19_results_json(out_dir, results)
            print(f"Results saved to {out_path}")
    else:
        if args.no_save_results:
            print("Evaluating without saving results")
            results = evaluate_lighteval_task(model, args.benchmark)
        else:
            results = evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark)

    print(results)
