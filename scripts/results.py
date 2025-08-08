import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from sentence_attention.artifacts.experiments import get_all_last_checkpoints
from sentence_attention.evaluation.benchmarks import all_benchmarks
from tabulate import tabulate


def infer_model_family(experiment_name: str) -> str:
    # Example names:
    # sentence_Llama-3.2-3B_ft_full_...  -> Llama
    # sentence_Qwen2.5-3B_ft_lora_...    -> Qwen
    name_lower = experiment_name.lower()
    if "llama" in name_lower:
        return "Llama"
    if "qwen" in name_lower:
        return "Qwen"
    return "Unknown"


def infer_training_type(experiment_name: str) -> str:
    # Normalize to three buckets
    name_lower = experiment_name.lower()
    if "ft_only_eos_embedding" in name_lower:
        return "eos_only"
    if "ft_full" in name_lower:
        return "full finetune"
    if "ft_lora" in name_lower:
        return "lora"
    return "unknown"


def extract_checkpoint_step(checkpoint_dir_name: str) -> int:
    # checkpoint-12345 -> 12345
    try:
        return int(checkpoint_dir_name.split("-")[-1])
    except Exception:
        return -1


def build_rows(records: Iterable[dict]) -> List[dict]:
    rows: List[dict] = []
    for rec in records:
        eos_tokens = rec.get("eos_tokens_num")
        experiment = rec.get("experiment_name", "")
        checkpoint_dir = rec.get("checkpoint", "")
        checkpoint_step = extract_checkpoint_step(checkpoint_dir)
        model_family = infer_model_family(experiment)
        train_type = infer_training_type(experiment)
        rows.append(
            {
                "eos_tokens": str(eos_tokens),
                "family": model_family,
                "training": train_type,
                "experiment": experiment,
                "step": str(checkpoint_step),
                "full_path": rec.get("full_path", ""),
            }
        )
    return rows


def group_rows(rows: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (row["family"], row["training"])  # (Family, Training)
        groups[key].append(row)
    return groups


def read_benchmark_metric(checkpoint_path: str, task_name: str) -> str:
    eval_file = os.path.join(checkpoint_path, "evaluation", f"{task_name}.json")
    if not os.path.exists(eval_file):
        return ""
    try:
        with open(eval_file) as f:
            data = json.load(f)
        results_all = data.get("results", {}).get("all", {})
        preferred_order = ["acc_norm", "ppl"]
        value = None
        for key in preferred_order:
            if key in results_all:
                value = results_all[key]
                break
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)
    except Exception:
        return ""


def prettify_experiment_name(experiment_name: str) -> str:
    return (
        experiment_name.replace("ft_full_", "")
        .replace("_num_eos_tokens_4", "")
        .replace("ft_lora_", "")
        .replace("ft_only_eos_embedding_", "")
        .replace("sentence_", "")[:-9]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print LaTeX tables for last checkpoints grouped by model family and training type."
    )
    parser.add_argument(
        "--sort-experiment-name",
        action="store_true",
        help="Sort rows within each group by experiment name",
    )
    parser.add_argument(
        "--tablefmt",
        default="latex",
        choices=["latex", "latex_raw", "latex_booktabs"],
        help="Tabulate LaTeX format",
    )
    args = parser.parse_args()

    last_checkpoints = get_all_last_checkpoints()
    rows = build_rows(last_checkpoints)
    groups = group_rows(rows)

    training_mapping = {
        "eos_only": "EOS only",
        "full finetune": "Full finetune",
        "lora": "LoRA",
        "unknown": "Unknown",
    }

    # Controlled ordering: families and training types
    family_order = ["Qwen", "Llama", "Unknown"]
    train_type_order = ["eos_only", "full finetune", "lora", "unknown"]

    headers = ["#EOS", "Family", "Training", "Experiment"] + all_benchmarks

    for family in family_order:
        for train in train_type_order:
            key = (family, train)
            if key not in groups:
                continue
            group_rows_list = groups[key]
            if args.sort_experiment_name:
                group_rows_list = sorted(group_rows_list, key=lambda r: r["experiment"])

            # Build table rows with benchmark metrics
            table_rows: List[List[str]] = []
            for row in group_rows_list:
                values = [
                    row["eos_tokens"],
                    row["family"],
                    training_mapping[row["training"]],
                    prettify_experiment_name(row["experiment"]),
                ]
                for task in all_benchmarks:
                    metric = read_benchmark_metric(row["full_path"], task)
                    values.append(metric)
                table_rows.append(values)

            # Print a LaTeX comment as a group header
            print(f"% Group: {family} - {train}")
            print(
                tabulate(
                    table_rows,
                    headers=headers,
                    tablefmt=args.tablefmt,
                    disable_numparse=True,
                )
            )
            print()

    print("\n\n\n")

    table_rows: List[List[str]] = []
    for row in rows:
        values = [
            row["eos_tokens"],
            row["family"],
            training_mapping[row["training"]],
            prettify_experiment_name(row["experiment"]),
        ]
        for task in all_benchmarks:
            metric = read_benchmark_metric(row["full_path"], task)
            values.append(metric)
        table_rows.append(values)

    table_rows = sorted(
        table_rows,
        key=lambda x: (
            x[3],
            x[2],
            x[0],
        ),
    )

    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )


if __name__ == "__main__":
    main()
