import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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
    if "base_model" in name_lower:
        return "base model"

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
    normalized_name = (
        experiment_name.replace("ft_full_", "")
        .replace("_base_model", "")
        .replace("_num_eos_tokens_4", "")
        .replace("ft_lora_", "")
        .replace("ft_only_eos_embedding_", "")
        .replace("sentence_", "")
    )

    if normalized_name[-9] == "_":
        normalized_name = normalized_name[:-9]

    return normalized_name


def row_to_base_values(row: dict, training_mapping: Dict[str, str]) -> List[str]:
    return [
        row["eos_tokens"],
        row["family"],
        training_mapping.get(row["training"], row["training"]),
        prettify_experiment_name(row["experiment"]),
    ]


def build_table(
    rows: List[dict],
    benchmarks: List[str],
    training_mapping: Dict[str, str],
    row_predicate=None,
    model_filter: Optional[str] = None,
    eos_tokens_filter: Optional[int] = None,
) -> List[List[str]]:
    table_rows: List[List[str]] = []
    for row in rows:
        if model_filter is not None and row["family"].lower() != model_filter.lower():
            continue
        if eos_tokens_filter is not None and int(row["eos_tokens"]) != eos_tokens_filter:
            continue
        if row_predicate is not None and not row_predicate(row):
            continue

        values = row_to_base_values(row, training_mapping)
        for task in benchmarks:
            metric = read_benchmark_metric(row["full_path"], task)
            values.append(metric)
        table_rows.append(values)

    table_rows = sorted(
        table_rows,
        key=lambda x: (
            x[3],  # experiment
            x[2],  # training
            x[0],  # #EOS
        ),
    )
    return table_rows


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter results by model name",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Filter results by benchmark name",
    )
    parser.add_argument(
        "--eos-tokens",
        type=int,
        default=None,
        help="Filter results by number of EOS tokens",
    )
    args = parser.parse_args()

    last_checkpoints = get_all_last_checkpoints()
    rows = build_rows(last_checkpoints)

    training_mapping = {
        "eos_only": "EOS only",
        "full finetune": "Full finetune",
        "lora": "LoRA",
        "base model": "Base",
        "unknown": "Unknown",
    }

    short_benchmarks = ["mmlu_cloze", "hellaswag", "arc", "winogrande"]

    if args.benchmarks == "all":
        benchmarks = all_benchmarks
    elif args.benchmarks == "short":
        benchmarks = short_benchmarks
    else:
        raise ValueError(f"Invalid benchmarks: {args.benchmarks}")

    headers = ["#EOS", "Family", "Training", "Experiment"] + benchmarks

    # Full table: all benchmarks (CLI filters apply only here)
    full_table_rows = build_table(
        rows=rows,
        benchmarks=benchmarks,
        training_mapping=training_mapping,
        row_predicate=None,
        model_filter=args.model,
        eos_tokens_filter=args.eos_tokens,
    )

    print(
        tabulate(
            full_table_rows,
            headers=headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )

    # Short results benchmark tables
    short_headers = ["#EOS", "Family", "Training", "Experiment"] + short_benchmarks

    print("\n\nMain results:")
    # 1) Main results: base models (0 EOS) and fully finetuned models
    main_short_rows = build_table(
        rows=rows,
        benchmarks=short_benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: int(r["eos_tokens"]) == 0 or r["training"] == "full finetune",
    )
    print(
        tabulate(
            main_short_rows,
            headers=short_headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )

    print("\n\nLoRA results:")
    # 2) Do we need LoRA? Fully finetuned and LoRA models
    lora_rows = build_table(
        rows=rows,
        benchmarks=short_benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: r["training"] in ("full finetune", "lora"),
    )
    print(
        tabulate(
            lora_rows,
            headers=short_headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )


if __name__ == "__main__":
    main()
