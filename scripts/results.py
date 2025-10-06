import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_attention.artifacts.experiments import get_all_checkpoints, get_all_last_checkpoints
from sentence_attention.evaluation.benchmarks import all_benchmarks, long_benchmarks, short_benchmarks
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
    if "ft_bos_token_full" in name_lower:
        return "full w/bos"
    if "ft_flexible_eos_tokens_full" in name_lower:
        return "full flexible"
    if "ft_full" in name_lower:
        return "full finetune"
    if "ft_lora" in name_lower:
        return "lora"
    if "base_model" in name_lower:
        return "base model"
    if "ft_4k_full" in name_lower:
        return "full finetune 4k"
    if "ft_4k_distill" in name_lower:
        return "full finetune 4k distill"
    if "ft_4k_colddown_full" in name_lower:
        return "full finetune 4k colddown"

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
                "eos_tokens": eos_tokens,
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

    # if task_name == "pg19":
    #     pass

    if task_name in long_benchmarks:
        return read_long_benchmark_metric(checkpoint_path, task_name)
    else:
        return read_short_benchmark_metric(checkpoint_path, task_name)


def get_score_metric_for_helmet_task(task_name: str) -> str:
    return {
        "recall": "ruler_recall",
        "rerank": "NDCG@10",
        "cite": "rougeLsum",
        "longqa": "rougeL_recall",
        "summ": "rougeL_recall",
        "icl": "exact_match",
    }[task_name]


def read_long_benchmark_metric(checkpoint_path: str, task_name: str) -> str:
    eval_file = os.path.join(checkpoint_path, "helmet_eval", task_name, "*.score")

    score_files = glob.glob(eval_file)

    # if 'Llama-3.2-3B' in checkpoint_path:
    #     print("Llama-3.2-3B", checkpoint_path, task_name)
    #     print("score_files", score_files)
    #     breakpoint()

    if len(score_files) == 0:
        return ""

    assert len(score_files) == 1, f"Multiple score files found for {checkpoint_path} {task_name}"
    score_file = score_files[0]

    task_metric = get_score_metric_for_helmet_task(task_name)

    with open(score_file) as f:
        data = json.load(f)
    return f"{data[task_metric]:.2f}"


def read_short_benchmark_metric(checkpoint_path: str, task_name: str) -> str:
    eval_file = os.path.join(checkpoint_path, "evaluation", f"{task_name}.json")
    if not os.path.exists(eval_file):
        return ""
    try:
        with open(eval_file) as f:
            data = json.load(f)

        if task_name == "pg19":
            results_all = data
        else:
            results_all = data.get("results", {}).get("all", {})

        preferred_order = ["acc_norm", "ppl", "overall_ppl", "qem"]
        # if task_name == "pg19" and "Llama-3.2-3B_base_model" in checkpoint_path:
        #     breakpoint()
        value = None
        for key in preferred_order:
            if key in results_all:
                value = results_all[key]
                break
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            if task_name in ["hellaswag", "arc", "winogrande", "mmlu_cloze"]:
                value = value * 100

            return f"{value:.2f}"
        return str(value)
    except Exception:
        return ""


def prettify_experiment_name(experiment_name: str) -> str:
    normalized_name = (
        experiment_name.replace("ft_full_", "")
        .replace("_ft_4k_colddown_full", "")
        .replace("_ft_4k_distill_full", "")
        .replace("_ft_4k_full", "")
        .replace("_base_model", "")
        .replace("_num_eos_tokens_1", "")
        .replace("_num_eos_tokens_2", "")
        .replace("_num_eos_tokens_4", "")
        .replace("_num_eos_tokens_8", "")
        .replace("_num_eos_tokens_16", "")
        .replace("ft_lora_", "")
        .replace("ft_only_eos_embedding_", "")
        .replace("ft_bos_token_full_", "")
        .replace("ft_flexible_eos_tokens_full_", "")
        .replace("sentence_", "")
        .replace("_5V455ZHK", "")
        .replace("_62XMQ139", "")
    )

    if normalized_name[-9] == "_":
        normalized_name = normalized_name[:-9]

    return normalized_name


def row_to_base_values(row: dict, training_mapping: Dict[str, str]) -> List[str]:
    return [
        row["eos_tokens"],
        # row["family"],
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
        # print(row["experiment"])
        if model_filter is not None and model_filter.lower() not in row["experiment"].lower():
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
            x[2],  # experiment
            x[1],  # training
            x[0],  # #EOS
        ),
    )
    return table_rows


def plot_per_checkpoint_short_results():
    # TODO

    matplotlib.style.use("seaborn-v0_8-darkgrid")

    for eos_tokens_num in [1, 4, 8, 16]:
        all_checkpoints = get_all_checkpoints(eos_tokens_num=eos_tokens_num)

        experiment_rows = build_rows(all_checkpoints)
        if len(experiment_rows) == 0:
            continue

        df = pd.DataFrame(experiment_rows)

        for training_type in ["eos_only", "full finetune", "lora"]:
            df_training_type = df[df["training"] == training_type]

            for benchmark in short_benchmarks:

                plt.figure()

                for model in df_training_type["experiment"].unique().tolist():
                    df_model_training_type = df_training_type[df_training_type["experiment"] == model].copy()

                    df_model_training_type["step"] = df_model_training_type["step"].astype(int)
                    df_model_training_type = df_model_training_type.sort_values(by="step")

                    model_steps = []
                    model_metrics = []

                    for _, row in df_model_training_type.iterrows():
                        row_dict = row.to_dict()
                        metric = read_benchmark_metric(row_dict["full_path"], benchmark)
                        if metric == "":
                            continue

                        model_steps.append(row_dict["step"])
                        model_metrics.append(float(metric))

                    model_label = model.replace("sentence_", "").split("_")[0]
                    plt.plot(model_steps, model_metrics, label=model_label)

                plt.legend()
                plt.title(f"{training_type} {benchmark}")

                plot_dir = f"artifacts/plots/eos_{eos_tokens_num}"
                os.makedirs(plot_dir, exist_ok=True)
                plot_file_name = f"{plot_dir}/per_checkpoint_short_results_{training_type.replace(' ', '-')}_{benchmark}.png"
                plt.savefig(plot_file_name)
                print(f"Saved plot to {plot_file_name}")
                plt.tight_layout()
                plt.close()


def plot_helmet_short_heatmap(
    model: str,
) -> None:
    matplotlib.style.use("seaborn-v0_8-darkgrid")

    benchmarks = ["recall", "rerank", "cite", "longqa", "summ", "icl"]
    sequence_lengths = [8192, 16384, 32768]

    expected_count_checkpoints = {
        "recall": {
            8192: 12,
            16384: 12,
            32768: 12,
        },
        "rerank": {
            8192: 3,
            16384: 3,
            32768: 3,
        },
        "cite": {
            8192: 6,
            16384: 6,
            32768: 6,
        },
        "longqa": {
            8192: 9,
            16384: 9,
            32768: 9,
        },
        "summ": {
            8192: 6,
            16384: 6,
            32768: 6,
        },
        "icl": {
            8192: 15,
            16384: 15,
            32768: 15,
        },
    }

    checkpoint_infos = get_all_last_checkpoints(model=model)
    checkpoints_bench_infos = []

    for benchmark in benchmarks:
        for checkpoint_info in checkpoint_infos:
            checkpoint_path = checkpoint_info["full_path"]

            benchmark_dir = os.path.join(checkpoint_path, "helmet_eval_short", benchmark)

            for sequence_length in sequence_lengths:

                all_scores = []
                score_files = glob.glob(os.path.join(benchmark_dir, f"*_in{sequence_length}_*.score"))

                expected_count_checkpoints[benchmark][sequence_length]
                if len(score_files) != expected_count_checkpoints[benchmark][sequence_length]:
                    print(
                        f"Expected {expected_count_checkpoints[benchmark][sequence_length]} score files for {benchmark} {sequence_length}, but got {len(score_files)}"
                    )
                    mean_bench_sequence_length_score = np.nan
                else:
                    for score_file in score_files:
                        with open(score_file) as f:
                            score_data = json.load(f)

                        task_metric = get_score_metric_for_helmet_task(benchmark)
                        all_scores.append(score_data[task_metric])

                    mean_bench_sequence_length_score = np.mean(all_scores)

                checkpoints_bench_infos.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "benchmark": benchmark,
                        "sequence_length": sequence_length,
                        "mean_score": mean_bench_sequence_length_score,
                    }
                )

    # TODO plot benchmarks heatmap with seaborn heatmap for each checkpoint and sequence length


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
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Plot per checkpoint short results",
        default=False,
    )
    parser.add_argument(
        "--poor-mask",
        action="store_true",
        help="Plot per checkpoint short results",
        default=False,
    )
    parser.add_argument(
        "--in_progress",
        action="store_true",
        help="Use in progress experiments",
        default=False,
    )
    args = parser.parse_args()

    # last_checkpoints = get_all_last_checkpoints(poor_mask=args.poor_mask)
    model_filter = args.model.split(",") if args.model else None
    last_checkpoints = get_all_last_checkpoints(in_progress=args.in_progress, model=model_filter)
    # print(f"Found {len(last_checkpoints)} last checkpoints", last_checkpoints)

    rows = build_rows(last_checkpoints)

    training_mapping = {
        "eos_only": "EOS only",
        "full finetune": "Full finetune",
        "lora": "LoRA",
        "base model": "Base",
        "unknown": "Unknown",
    }

    if args.benchmarks == "all":
        benchmarks = all_benchmarks
    elif args.benchmarks == "short":
        benchmarks = short_benchmarks
    elif args.benchmarks == "long":
        benchmarks = long_benchmarks
    else:
        raise ValueError(f"Invalid benchmarks: {args.benchmarks}")

    headers = ["#EOS", "Training", "Experiment"] + benchmarks

    # Full table: all benchmarks (CLI filters apply only here)
    full_table_rows = build_table(
        rows=rows,
        benchmarks=benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: "llama-3-8b" not in r["experiment"],
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
    short_headers = ["#EOS", "Training", "Experiment"] + short_benchmarks

    print("\n\nMain Short results:")
    # 1) Main results: base models (0 EOS) and fully finetuned models
    main_short_rows = build_table(
        rows=rows,
        benchmarks=short_benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: int(
            ("Llama-3.2-3B" in r["experiment"] or "Llama-3.2-1B" in r["experiment"])
            and (r["training"] in ("base model", "full finetune 4k", "full finetune 4k colddown"))
        ),
    )
    print(
        tabulate(
            main_short_rows,
            headers=short_headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )

    # Long results benchmark tables
    long_headers = ["#EOS", "Training", "Experiment"] + long_benchmarks

    print("\n\nMain Long results:")
    # 1) Main results: base models (0 EOS) and fully finetuned models
    # breakpoint()
    main_long_rows = build_table(
        rows=rows,
        benchmarks=long_benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: int(
            ("Llama-3.2-3B" in r["experiment"] or "Llama-3.2-1B" in r["experiment"])
            and (r["training"] in ("base model", "full finetune 4k", "full finetune 4k colddown", "unknown"))
        ),
    )
    print(
        tabulate(
            main_long_rows,
            headers=long_headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )

    print("\nSep cache / Beacon / Sentence Attention:")
    # 1) Main results: base models (0 EOS) and fully finetuned models
    # breakpoint()
    main_long_rows = build_table(
        rows=rows,
        # benchmarks=[ 'recall', 'longqa', 'icl' ],
        benchmarks=long_benchmarks,
        training_mapping=training_mapping,
        row_predicate=lambda r: int(
            #     # ("Llama-3.2-3B" in r["experiment"] or "beacon" in r["experiment"].lower() or "qwen" in r["experiment"].lower())
            r["training"]
            in ("base model", "full finetune 4k colddown")
        ),
    )
    print(
        tabulate(
            main_long_rows,
            headers=long_headers,
            tablefmt=args.tablefmt,
            disable_numparse=True,
        )
    )

    # HELMET Short heatmap
    plot_helmet_short_heatmap(
        model=args.model,
    )

    print_lora = False
    if print_lora:
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

    if args.plots:
        plot_per_checkpoint_short_results()


if __name__ == "__main__":
    main()
