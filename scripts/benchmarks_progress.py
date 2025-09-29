import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sentence_attention.evaluation.benchmarks import long_benchmarks


def extract_checkpoint_step(checkpoint_dir_name: str) -> int:
    try:
        return int(os.path.basename(checkpoint_dir_name).split("-")[-1])
    except Exception:
        return -1


def read_score_file_metric(score_file: str, task_name: str) -> str:
    task_metric = {
        "recall": "ruler_recall",
        "rerank": "NDCG@10",
        "cite": "rougeLsum",
        "longqa": "rougeL_recall",
        "summ": "rougeL_recall",
        "icl": "exact_match",
    }[task_name]

    with open(score_file) as f:
        data = json.load(f)
    return round(data[task_metric], 2)


def read_long_benchmark_metric(checkpoint_path: str, task_name: str) -> str:
    eval_file = os.path.join(checkpoint_path, "helmet_eval", task_name, "*.score")

    score_files = glob.glob(eval_file)

    # if 'Llama-3.2-3B' in checkpoint_path:
    #     print("Llama-3.2-3B", checkpoint_path, task_name)
    #     print("score_files", score_files)
    #     breakpoint()

    if len(score_files) == 0:
        return None

    assert len(score_files) == 1, f"Multiple score files found for {checkpoint_path} {task_name}"
    score_file = score_files[0]

    return read_score_file_metric(score_file, task_name)


def parse_bench_length(bench_name: str, score_file: str) -> int:
    # find regexp in(\d+)
    score_file_tmp = re.findall(r"__in(\d+)", score_file)

    return int(score_file_tmp[0])


# bench_name -> seq = [ ... ], values = [ ... ]
def read_long_short_benchmark_metric(checkpoint_path: str, task_name: str) -> str:
    eval_file = os.path.join(checkpoint_path, "helmet_eval_short", task_name, "*.score")

    score_files = glob.glob(eval_file)

    bench_names = [
        # Longqa
        "narrativeqa",
        "infbench_qa_eng",
        "icl_trec_coarse",
    ]

    # assert len(score_files) == 1, f"Multiple score files found for {checkpoint_path} {task_name}"
    # score_file = score_files[0]

    current_results = {}

    for score_file in score_files:
        score_file_basename = os.path.basename(score_file)

        bench_name = None
        for bench_name_i in bench_names:
            if score_file_basename.startswith(bench_name_i):
                bench_name = bench_name_i
                break

        if bench_name is None:
            raise ValueError(f"Unknown bench name for {checkpoint_path}: {score_file}")

        bench_value = read_score_file_metric(score_file, task_name)

        bench_length = parse_bench_length(bench_name, score_file_basename)

        if bench_name not in current_results:
            current_results[bench_name] = {
                "seq_values": [],
            }

        current_results[bench_name]["seq_values"].append((bench_length, bench_value))

    for bench in current_results:
        current_results[bench]["seq_values"].sort(key=lambda x: x[0])

    final_results = {}
    for bench, values in current_results.items():
        final_results[bench] = {
            "seq": [x[0] for x in values["seq_values"]],
            "values": [x[1] for x in values["seq_values"]],
        }

    return final_results


def read_benchmark_metric(checkpoint_path: str, task_name: str, preferred_order: Sequence[str]) -> Optional[float]:

    if task_name in long_benchmarks:
        return read_long_benchmark_metric(checkpoint_path, task_name)

    eval_file = os.path.join(checkpoint_path, "evaluation", f"{task_name}.json")
    if not os.path.exists(eval_file):
        return None
    try:
        with open(eval_file) as f:
            data = json.load(f)
        results_all: Dict = data.get("results", {}).get("all", {})
        value = None
        for key in preferred_order:
            if key in results_all:
                value = results_all[key]
                break
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        # Some tasks may store nested dicts. Try common nested schema like {"value": x}
        if isinstance(value, dict):
            for k in ("value", "mean", "score"):
                if k in value and isinstance(value[k], (int, float)):
                    return float(value[k])
        return None
    except Exception:
        return None


def find_all_checkpoints(experiment_dir: str) -> List[Tuple[int, str]]:
    pattern = os.path.join(experiment_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(pattern)
    steps_and_paths: List[Tuple[int, str]] = []
    for ckpt in checkpoint_dirs:
        step = extract_checkpoint_step(ckpt)
        if step >= 0 and os.path.isdir(os.path.join(ckpt, "evaluation")):
            steps_and_paths.append((step, ckpt))
    steps_and_paths.sort(key=lambda x: x[0])
    return steps_and_paths


def discover_benchmarks(experiment_dir: str, checkpoints: List[Tuple[int, str]]) -> List[str]:
    seen = set()
    for _, ckpt in checkpoints:
        eval_dir = os.path.join(ckpt, "evaluation")
        if not os.path.isdir(eval_dir):
            continue
        for fname in os.listdir(eval_dir):
            if fname.endswith(".json"):
                seen.add(os.path.splitext(fname)[0])
    return sorted(seen)


def plot_benchmark_over_checkpoints(
    experiment_dirs: List[str],
    benchmarks: List[str],
    metric_preference: Sequence[str],
    output_dir: Optional[str] = None,
) -> None:
    matplotlib.style.use("seaborn-v0_8-darkgrid")

    # Collect data from all experiments
    experiment_data = {}
    all_benchmarks = set()

    for experiment_dir in experiment_dirs:
        checkpoints = find_all_checkpoints(experiment_dir)
        if not checkpoints:
            print(f"No checkpoints with evaluations found under: {experiment_dir}")
            continue

        # Discover benchmarks for this experiment
        exp_benchmarks = discover_benchmarks(experiment_dir, checkpoints)
        all_benchmarks.update(exp_benchmarks)

        # Store experiment data
        experiment_data[experiment_dir] = {"checkpoints": checkpoints, "benchmarks": exp_benchmarks}

    if not experiment_data:
        print("No valid experiments found.")
        return

    if not benchmarks:
        benchmarks = sorted(all_benchmarks)

    if not benchmarks:
        print("No benchmark JSON files discovered. Nothing to plot.")
        return

    if output_dir is None:
        if len(experiment_dirs) == 1:
            output_dir = os.path.join(experiment_dirs[0], "evaluation_plots")
        else:
            raise ValueError("Output directory must be specified when plotting multiple experiments")
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        # Preparing heatmap figure

        for experiment_dir, data in experiment_data.items():
            steps: List[int] = []
            values: List[float] = []

            for step, ckpt_path in data["checkpoints"]:
                val = read_benchmark_metric(ckpt_path, benchmark, metric_preference)
                if val is None:
                    continue
                steps.append(step)
                values.append(val)

            if not steps:
                print(f"Skipping '{benchmark}' for experiment '{experiment_dir}': no numeric metrics found across checkpoints.")
                continue

            # Create experiment label from directory name
            exp_label = os.path.basename(experiment_dir.rstrip("/"))
            exp_label = exp_label.split("_")[-1]
            plt.plot(steps, values, marker="o", label=exp_label)

        plt.xlabel("Checkpoint step")
        plt.ylabel("Metric value")
        plt.title(f"{benchmark}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{benchmark}_over_checkpoints.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def heatmap_benchmark_over_checkpoints(
    experiment_dirs: List[str],
    benchmarks: List[str],
    metric_preference: Sequence[str],
    output_dir: Optional[str] = None,
) -> None:
    matplotlib.style.use("seaborn-v0_8-darkgrid")

    # Collect data from all experiments
    experiment_data = {}
    all_benchmarks = set()

    for experiment_dir in experiment_dirs:
        checkpoints = find_all_checkpoints(experiment_dir)
        if not checkpoints:
            print(f"No checkpoints with evaluations found under: {experiment_dir}")
            continue

        # Discover benchmarks for this experiment
        exp_benchmarks = discover_benchmarks(experiment_dir, checkpoints)
        all_benchmarks.update(exp_benchmarks)

        # Store experiment data
        experiment_data[experiment_dir] = {"checkpoints": checkpoints, "benchmarks": exp_benchmarks}

    if not experiment_data:
        print("No valid experiments found.")
        return

    if not benchmarks:
        benchmarks = sorted(all_benchmarks)

    if not benchmarks:
        print("No benchmark JSON files discovered. Nothing to plot.")
        return

    if output_dir is None:
        if len(experiment_dirs) == 1:
            output_dir = os.path.join(experiment_dirs[0], "evaluation_plots")
        else:
            raise ValueError("Output directory must be specified when plotting multiple experiments")
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        plt.figure()

        # bench name -> checkpoint -> seq = [], values = []
        bench_results = {}

        for data in experiment_data.values():
            for step, ckpt_path in data["checkpoints"]:
                # bench_name -> seq = [ ... ], values = [ ... ]
                result = read_long_short_benchmark_metric(ckpt_path, benchmark)

                if len(result) == 0:
                    continue

                for bench_name, bench_result in result.items():
                    if bench_name not in bench_results:
                        bench_results[bench_name] = []

                    bench_results[bench_name].append(
                        {
                            "step": step,
                            "bench_length": bench_result["seq"],
                            "bench_value": bench_result["values"],
                        }
                    )

        if len(bench_results) == 0:
            continue

        # bench_results = {'narrativeqa':
        # [{'step': 250, 'bench_length': [7892, 16084, 32468, 65236], 'bench_value': [32.16, 37.2, 38.88, 15.56]},
        # {'step': 500, 'bench_length': [7892, 16084, 32468, 65236], 'bench_value': [32.82, 37.19, 39.94, 18.25]},
        # {'step': 750, 'bench_length': [7892, 16084, 32468, 65236], 'bench_value': [35.16, 36.01, 41.67, 20.83]},
        # {'step': 1000, 'bench_length': [7892, 16084, 32468, 65236], 'bench_value': [34.7, 39.73, 41.51, 17.79]}],
        # 'infbench_qa_eng':
        # [{'step': 250, 'bench_length': [7982, 16174, 32558], 'bench_value': [23.24, 27.17, 30.04]},
        # {'step': 500, 'bench_length': [7982, 16174, 32558], 'bench_value': [26.51, 28.5, 32.92]},
        # {'step': 750, 'bench_length': [7982, 16174, 32558], 'bench_value': [20.07, 25.77, 29.46]},
        # {'step': 1000, 'bench_length': [7982, 16174, 32558], 'bench_value': [25.58, 25.36, 31.19]}]}

        # Plot heatmaps here
        bench_names = sorted(bench_results.keys())

        # If nothing collected for this benchmark, skip
        if len(bench_names) == 0:
            continue

        # Collect union of steps across all benches
        all_steps = sorted({item["step"] for vals in bench_results.values() for item in vals})
        step_to_idx = {s: i for i, s in enumerate(all_steps)}

        # Create subplots, one per bench name
        nrows = len(bench_names)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(max(6, len(all_steps) * 0.6), 3 * nrows),
            squeeze=False,
        )
        axes = axes.flatten()

        for i, bench_name in enumerate(bench_names):
            entries = bench_results[bench_name]
            # Union of lengths for Y-axis
            lengths = sorted({l for e in entries for l in e["bench_length"]})  # noqa: E741
            length_to_idx = {l: idx for idx, l in enumerate(lengths)}  # noqa: E741
            # Initialize grid with NaNs
            grid = np.full((len(lengths), len(all_steps)), np.nan, dtype=float)

            # Fill grid with values for each (length, step)
            for e in entries:  # noqa: E741
                col = step_to_idx.get(e["step"])
                if col is None:
                    continue
                for l, v in zip(e["bench_length"], e["bench_value"]):  # noqa: E741
                    row = length_to_idx.get(l)
                    if row is not None:
                        grid[row, col] = v

            # Use seaborn heatmap, flip vertically to mimic origin='lower'
            grid_plot = np.flipud(grid)
            ylabels = [str(l) for l in lengths][::-1]  # noqa: E741
            xlabels = [str(s) for s in all_steps]  # noqa: E741

            sns.heatmap(
                grid_plot,
                ax=axes[i],
                cmap="viridis",
                mask=np.isnan(grid_plot),
                cbar=True,
                annot=True,
                fmt=".2f",
                vmin=0,
                # vmax=50,
                annot_kws={"fontsize": 8},
                xticklabels=xlabels,
                yticklabels=ylabels,
            )

            axes[i].set_title(bench_name)
            axes[i].set_ylabel("Seq length")

            if i == nrows - 1:
                axes[i].set_xlabel("Checkpoint step")
                axes[i].tick_params(axis="x", labelrotation=45)
            else:
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", labelbottom=False)

            # Label colorbar if available
            mappable = axes[i].collections[0] if len(axes[i].collections) > 0 else None
            if mappable is not None and getattr(mappable, "colorbar", None) is not None:
                mappable.colorbar.set_label("Score")

        fig.suptitle(f"{benchmark}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(output_dir, f"{benchmark}_short_over_checkpoints.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how benchmark metrics change across checkpoints for given experiment(s). "
            "X-axis is checkpoint step, Y-axis is the metric value. "
            "When multiple experiments are specified, all experiments are plotted together with legends."
        )
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=str,
        help=(
            "Path(s) to experiment directory(ies) containing checkpoint-*/evaluation/*.json.\n"
            "Example: artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR\n"
            "For multiple experiments: exp1 exp2 exp3"
        ),
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help=(
            "Which benchmarks to plot: 'all', 'short', or a comma-separated list of names. "
            "Short set: mmlu_cloze,hellaswag,arc,winogrande"
        ),
    )
    parser.add_argument(
        "--ruler_mode",
        type=str,
        default="none",
        choices=["none", "tiny", "short"],
        help="Use short ruler mode.",
    )
    parser.add_argument(
        "--metric-keys",
        type=str,
        default="acc_norm,ppl",
        help=("Comma-separated preference order of metric keys to read from results['all']. " "Defaults to 'acc_norm,ppl'."),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Required when plotting multiple experiments. Defaults to <experiment_dir>/evaluation_plots for single experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    short_benchmarks = ["mmlu_cloze", "hellaswag", "arc", "winogrande"]
    long_benchmarks = ["recall", "rerank", "cite", "longqa", "summ", "icl"]

    metric_preference = [k.strip() for k in args.metric_keys.split(",") if k.strip()]

    if args.benchmarks == "all":
        benchmarks: List[str] = []
    elif args.benchmarks == "short":
        benchmarks = short_benchmarks
    elif args.benchmarks == "long":
        benchmarks = long_benchmarks
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    if args.ruler_mode != "short":
        plot_benchmark_over_checkpoints(
            experiment_dirs=args.experiment_dirs,
            benchmarks=benchmarks,
            metric_preference=metric_preference,
            output_dir=args.output_dir,
        )
    else:
        heatmap_benchmark_over_checkpoints(
            experiment_dirs=args.experiment_dirs,
            benchmarks=benchmarks,
            metric_preference=metric_preference,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
