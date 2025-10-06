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
import seaborn as sns
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


def get_score_metric_for_helmet_task(task_name: str, score_file: Optional[str] = None) -> str:
    if score_file is not None and task_name == "recall":
        if "json_kv_eval_" in score_file:
            return "substring_exact_match"

    if score_file is not None and task_name == "cite":
        if "qampari_eval_" in score_file:
            return "qampari_rec_top5"

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
    return f"{data[task_metric]:.1f}"


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
            8192: [
                "ruler_niah_mk_2_eval_validation_8192_in8192_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mk_3_eval_validation_8192_in8192_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mv_eval_validation_8192_in8192_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "json_kv_eval_test_k105_dep6_in8192_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "ruler_niah_mk_2_eval_validation_16384_in16384_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mk_3_eval_validation_16384_in16384_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mv_eval_validation_16384_in16384_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "json_kv_eval_test_k220_dep6_in16384_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "ruler_niah_mk_2_eval_validation_32768_in32768_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mk_3_eval_validation_32768_in32768_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "ruler_niah_mv_eval_validation_32768_in32768_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score",
                "json_kv_eval_test_k440_dep6_in32768_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
        "rerank": {
            8192: [
                "msmarco_rerank_psg_eval_test_reranking_data_k50_dep3_in8192_size100_shots2_sampFalsemax200min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "msmarco_rerank_psg_eval_test_reranking_data_k130_dep3_in16384_size100_shots2_sampFalsemax200min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "msmarco_rerank_psg_eval_test_reranking_data_k285_dep3_in32768_size100_shots2_sampFalsemax200min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
        "cite": {
            8192: [
                "alce_asqa_30_eval_asqa_eval_gtr_top2000_in8192_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
                "alce_qampari_30_eval_qampari_eval_gtr_top2000_in8192_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "alce_asqa_75_eval_asqa_eval_gtr_top2000_in16384_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
                "alce_qampari_75_eval_qampari_eval_gtr_top2000_in16384_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "alce_qampari_165_eval_qampari_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
                "alce_asqa_165_eval_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
        "longqa": {
            8192: [
                "narrativeqa_7892_eval__in8192_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "infbench_qa_eng_7982_eval__in8192_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.score",
                "infbench_choice_eng_7982_eval__in8192_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "narrativeqa_16084_eval__in16384_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "infbench_qa_eng_16174_eval__in16384_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.score",
                "infbench_choice_eng_16174_eval__in16384_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "narrativeqa_32468_eval__in32768_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score",
                "infbench_qa_eng_32558_eval__in32768_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.scorere",
                "infbench_choice_eng_32558_eval__in32768_size100_shots2_sampFalsemax10min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
        "summ": {
            8192: [
                "infbench_sum_eng_6792_eval__in8192_size100_shots2_sampFalsemax1200min0t0.0p1.0_chatFalse_42.json.score",
                "multi_lexsum_7492_eval__in8192_size100_shots2_sampFalsemax400min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "infbench_sum_eng_14984_eval__in16384_size100_shots2_sampFalsemax1200min0t0.0p1.0_chatFalse_42.json.score",
                "multi_lexsum_15684_eval__in16384_size100_shots2_sampFalsemax400min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "infbench_sum_eng_31368_eval__in32768_size100_shots2_sampFalsemax1200min0t0.0p1.0_chatFalse_42.json.score",
                "multi_lexsum_32068_eval__in32768_size100_shots2_sampFalsemax400min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
        "icl": {
            8192: [
                "icl_trec_coarse_400shot_balance_eval__in8192_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
                "icl_trec_fine_400shot_balance_eval__in8192_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
            ],
            16384: [
                "icl_trec_coarse_800shot_balance_eval__in16384_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
                "icl_trec_fine_800shot_balance_eval__in16384_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
            ],
            32768: [
                "icl_trec_coarse_1600shot_balance_eval__in32768_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
                "icl_trec_fine_1600shot_balance_eval__in32768_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score",
            ],
        },
    }

    checkpoint_infos = get_all_last_checkpoints(model=model.split(","))
    checkpoints_bench_infos = []

    for benchmark in benchmarks:
        for checkpoint_info in sorted(checkpoint_infos, key=lambda x: x["full_path"]):
            checkpoint_path = checkpoint_info["full_path"]

            benchmark_dir = os.path.join(checkpoint_path, "helmet_eval_short", benchmark)

            for sequence_length in sequence_lengths:

                all_scores = []
                for score_file in expected_count_checkpoints[benchmark][sequence_length]:
                    score_file = os.path.join(benchmark_dir, score_file)

                    if not os.path.exists(score_file):
                        print(f"Score file {score_file} does not exist:", score_file)
                        all_scores = []
                        break

                    with open(score_file) as f:
                        score_data = json.load(f)

                    task_metric = get_score_metric_for_helmet_task(benchmark, score_file=score_file)
                    all_scores.append(score_data[task_metric])

                # print("all_scores", checkpoint_path, benchmark, sequence_length, all_scores)
                mean_bench_sequence_length_score = np.mean(all_scores)

                checkpoints_bench_infos.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "benchmark": benchmark,
                        "sequence_length": sequence_length,
                        "mean_score": mean_bench_sequence_length_score,
                    }
                )

    # Build a single combined heatmap: rows=checkpoints, columns grouped by task (sequence lengths)
    if len(checkpoints_bench_infos) == 0:
        return

    df = pd.DataFrame(checkpoints_bench_infos)

    # Ensure consistent ordering
    df["benchmark"] = pd.Categorical(df["benchmark"], categories=benchmarks, ordered=True)
    df["sequence_length"] = pd.Categorical(df["sequence_length"], categories=sequence_lengths, ordered=True)

    # Derive human-friendly row labels and sorting keys per checkpoint
    rows_meta = []
    for checkpoint_path in sorted(df["checkpoint_path"].unique()):
        experiment_dir = os.path.basename(os.path.dirname(checkpoint_path))
        pretty_experiment = prettify_experiment_name(experiment_dir)

        import re

        num_eos_tokens = int(re.search(r"eos_(\d+)", checkpoint_path).group(1))

        # step = extract_checkpoint_step(checkpoint_dir_name)
        row_label = f"{pretty_experiment} | Ng={num_eos_tokens}"
        rows_meta.append((checkpoint_path, row_label, pretty_experiment, num_eos_tokens))

    # Sort by experiment then step
    rows_meta = sorted(rows_meta, key=lambda t: (t[2], t[3]))
    ordered_paths = [t[0] for t in rows_meta]
    ordered_labels = [t[1] for t in rows_meta]

    # Pivot to wide with MultiIndex columns (benchmark, sequence_length)
    wide = df.pivot_table(
        index="checkpoint_path",
        columns=["benchmark", "sequence_length"],
        values="mean_score",
        aggfunc="mean",
    )

    # Reindex rows and columns to desired order
    wide = wide.reindex(index=ordered_paths)
    # Ensure full columns order exists
    full_columns = []
    for b in benchmarks:
        for sl in sequence_lengths:
            full_columns.append((b, sl))
    # Align columns order and include any missing ones
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(full_columns, names=["benchmark", "sequence_length"]))

    # Insert separator columns between tasks to visually separate groups
    sep_width = 1  # one separator column between groups

    data = wide.values
    n_rows, n_cols = data.shape
    num_groups = len(benchmarks)
    new_cols = n_cols + sep_width * (num_groups - 1)

    data_with_sep = np.full((n_rows, new_cols), np.nan, dtype=float)
    mask_with_sep = np.ones((n_rows, new_cols), dtype=bool)
    x_labels = []
    group_centers = []

    col_ptr = 0
    for gi, b in enumerate(benchmarks):  # noqa: B007
        # columns for this group in original matrix
        start = gi * len(sequence_lengths)
        end = start + len(sequence_lengths)
        span = end - start

        data_with_sep[:, col_ptr : col_ptr + span] = data[:, start:end]
        mask_with_sep[:, col_ptr : col_ptr + span] = np.isnan(data[:, start:end])

        # labels for this group (sequence lengths only)
        for sl in sequence_lengths:
            x_labels.append(f"{int(sl/1024)}k")

        # group center (for top labels)
        group_centers.append(col_ptr + span / 2)

        col_ptr += span
        if gi < num_groups - 1:
            # leave separator column as NaN (masked)
            x_labels.append("")
            col_ptr += sep_width

    # Figure size scales with matrix size
    n_rows, n_cols_new = data_with_sep.shape
    fig_width = max(12, 1.2 * n_cols_new)
    fig_height = max(4, 0.8 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        data_with_sep,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        vmin=0,
        vmax=np.nanmax(wide.values),
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=ax,
        mask=mask_with_sep,
    )
    ax.set_xticks(np.arange(n_cols_new) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(ordered_labels, rotation=0)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Checkpoint")

    # Add benchmark names on the top centered over each group
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks([c + 0.5 for c in group_centers])
    ax_top.set_xticklabels(benchmarks, rotation=0)
    ax_top.set_xlabel("Benchmark")

    # Draw vertical lines to emphasize group boundaries
    boundary_positions = []
    pos = 0
    for gi in range(num_groups - 1):  # noqa: B007
        pos += len(sequence_lengths)
        boundary_positions.append(pos)
        pos += sep_width
    for bp in boundary_positions:  # noqa: B007
        ax.axvline(bp, color="white", linewidth=2)

    ax.set_title("HELMET short benchmarks heatmap")
    plt.tight_layout()

    # Output
    base_plot_dir = os.path.join("artifacts", "plots", "helmet_short_heatmap")
    os.makedirs(base_plot_dir, exist_ok=True)
    model_slug = "all" if not model else str(model).replace(",", "_")
    out_path = os.path.join(base_plot_dir, f"combined_{model_slug}.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved combined heatmap to {out_path}")
    plt.close(fig)


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
    if False:
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
