import argparse
import os
from tqdm.auto import tqdm

import transformers
from sentence_attention.evaluation.benchmarks import all_benchmarks
from sentence_attention.evaluation.evaluation import evaluate_lighteval_task, evaluate_lighteval_task_save_results
from sentence_attention.evaluation.my_recall import evaluate_synthetic_my_recall
from sentence_attention.evaluation.pg19 import evaluate_pg19_ppl, save_pg19_results_json
from sentence_attention.models.checkpoint import load_model_from_checkpoint

import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sel-llm", action="store_true", default=False)
    parser.add_argument(
        "--benchmark", type=str, required=True, choices=all_benchmarks + ["synthetic_my_recall", "synthetic_my_recall_grid"]
    )
    parser.add_argument("--no-save-results", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attention-implementation", type=str, default=None)
    parser.add_argument("--synthetic_my_recall_sample_num_examples", type=int, default=10)
    parser.add_argument(
        "--synthetic_my_recall_niddle_type", type=str, default="numbers", choices=["numbers", "strings", "strings_upper"]
    )
    parser.add_argument("--synthetic_my_recall_hint_first", action="store_true", default=False)

    args = parser.parse_args()

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
    elif args.benchmark == "synthetic_my_recall_grid":
        results = []

        query_words_count_list = [1, 4, 8, 16, 32]
        for query_words_count in tqdm(query_words_count_list, desc="query_words_count"):
            result = evaluate_synthetic_my_recall(
                model,
                tokenizer,
                hint_first=args.synthetic_my_recall_hint_first,
                max_samples=args.max_samples,
                sample_num_examples=args.synthetic_my_recall_sample_num_examples,
                niddle_type=args.synthetic_my_recall_niddle_type,
                save_results=(not args.no_save_results),
                checkpoint_path=args.checkpoint,
                query_words_count=query_words_count,
            )
            print("results", result)
            results.append(result)
        # Save plots for accuracy vs query_words_count
        plt.plot(query_words_count_list, [result["accuracy"] for result in results])
        plt.xlabel("Query words count")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("Synthetic My Recall Accuracy vs Query Words Count")
        plt.savefig(os.path.join(args.checkpoint, "evaluation", "synthetic_my_recall_grid_accuracy_vs_query_words_count.png"))
        plt.close()

        results = []
        value_max_number_exponent_list = [4, 8, 16, 32]
        for value_max_number_exponent in tqdm(value_max_number_exponent_list, desc="value_max_number_exponent"):
            value_max_number = 10**value_max_number_exponent
            result = evaluate_synthetic_my_recall(
                model,
                tokenizer,
                hint_first=args.synthetic_my_recall_hint_first,
                max_samples=args.max_samples,
                sample_num_examples=args.synthetic_my_recall_sample_num_examples,
                niddle_type=args.synthetic_my_recall_niddle_type,
                save_results=(not args.no_save_results),
                checkpoint_path=args.checkpoint,
                value_max_number=value_max_number,
            )
            print("results", result)
            results.append(result)

        # Save plots for accuracy vs value_max_number
        plt.plot(value_max_number_exponent_list, [result["accuracy"] for result in results])
        plt.xlabel("Value max number exponent")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("Synthetic My Recall Accuracy vs Value Max Number Exponent")
        plt.savefig(os.path.join(args.checkpoint, "evaluation", "synthetic_my_recall_grid_accuracy_vs_value_max_number.png"))
        plt.close()

    elif args.benchmark == "synthetic_my_recall":
        results = evaluate_synthetic_my_recall(
            model,
            tokenizer,
            hint_first=args.synthetic_my_recall_hint_first,
            max_samples=args.max_samples,
            sample_num_examples=args.synthetic_my_recall_sample_num_examples,
            niddle_type=args.synthetic_my_recall_niddle_type,
            save_results=(not args.no_save_results),
            checkpoint_path=args.checkpoint,
        )
        print("results", results)
    else:
        if args.no_save_results:
            print("Evaluating without saving results")
            results = evaluate_lighteval_task(model, args.benchmark, max_samples=args.max_samples)
        else:
            results = evaluate_lighteval_task_save_results(model, args.checkpoint, args.benchmark, max_samples=args.max_samples)

    print(results)
