import argparse
import os

import transformers
from peft import PeftConfig, PeftModel
from sentence_attention.evaluation.benchmarks import all_benchmarks
from sentence_attention.evaluation.evaluation import evaluate_lighteval_task, evaluate_lighteval_task_save_results
from sentence_attention.evaluation.my_recall import evaluate_synthetic_my_recall
from sentence_attention.evaluation.pg19 import evaluate_pg19_ppl, save_pg19_results_json
from sentence_attention.models.checkpoint import load_model_from_checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sel-llm", action="store_true", default=False)
    parser.add_argument("--benchmark", type=str, required=True, choices=all_benchmarks + ["synthetic_my_recall"])
    parser.add_argument("--no-save-results", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attention-implementation", type=str, default=None)

    args = parser.parse_args()

    if "lora" in args.checkpoint:
        peft_config = PeftConfig.from_pretrained(args.checkpoint)
        base_model, tokenizer = load_model_from_checkpoint(
            peft_config.base_model_name_or_path, attention_implementation=args.attention_implementation
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, attention_implementation=args.attention_implementation)

    if args.sel_llm:
        assert transformers.__version__ == "4.53.0", "transformers version must be 4.53.0"
        import os

        os.environ["SEPCACHE_ENABLED"] = "1"

    if args.benchmark == "pg19":

        model_type = "vanilla"
        if "sentence" in str(type(model)).lower():
            model_type = "sentence"

        results = evaluate_pg19_ppl(
            model,
            tokenizer,
            dataset_path="/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/pg19_test",  # TODO move to HF?
            model_type=model_type,
            max_samples=args.max_samples,
            max_length=32000,
            # max_length=64000,
        )
        if not args.no_save_results:
            out_dir = os.path.join(args.checkpoint, "evaluation")
            os.makedirs(out_dir, exist_ok=True)
            out_path = save_pg19_results_json(out_dir, results)
            print(f"Results saved to {out_path}")
    elif args.benchmark == "synthetic_my_recall":
        results = evaluate_synthetic_my_recall(model, tokenizer, max_samples=args.max_samples)
        print("results", results)
    else:
        if args.no_save_results:
            print("Evaluating without saving results")
            results = evaluate_lighteval_task(model, args.benchmark, max_samples=args.max_samples)
        else:
            results = evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark, max_samples=args.max_samples)

    print(results)
