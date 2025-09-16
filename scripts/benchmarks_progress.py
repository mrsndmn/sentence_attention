import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt


def extract_checkpoint_step(checkpoint_dir_name: str) -> int:
    try:
        return int(os.path.basename(checkpoint_dir_name).split("-")[-1])
    except Exception:
        return -1


def read_benchmark_metric(checkpoint_path: str, task_name: str, preferred_order: Sequence[str]) -> Optional[float]:
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
    experiment_dir: str,
    benchmarks: List[str],
    metric_preference: Sequence[str],
    output_dir: Optional[str] = None,
) -> None:
    matplotlib.style.use("seaborn-v0_8-darkgrid")

    checkpoints = find_all_checkpoints(experiment_dir)
    if not checkpoints:
        print(f"No checkpoints with evaluations found under: {experiment_dir}")
        return

    if not benchmarks:
        benchmarks = discover_benchmarks(experiment_dir, checkpoints)

    if not benchmarks:
        print("No benchmark JSON files discovered. Nothing to plot.")
        return

    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "evaluation_plots")
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        steps: List[int] = []
        values: List[float] = []

        for step, ckpt_path in checkpoints:
            val = read_benchmark_metric(ckpt_path, benchmark, metric_preference)
            if val is None:
                continue
            steps.append(step)
            values.append(val)

        if not steps:
            print(f"Skipping '{benchmark}': no numeric metrics found across checkpoints.")
            continue

        plt.figure()
        plt.plot(steps, values, marker="o")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Metric value")
        plt.title(f"{benchmark}")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{benchmark}_over_checkpoints.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how benchmark metrics change across checkpoints for a given experiment. "
            "X-axis is checkpoint step, Y-axis is the metric value."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help=(
            "Path to the experiment directory containing checkpoint-*/evaluation/*.json.\n"
            "Example: artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR"
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
        "--metric-keys",
        type=str,
        default="acc_norm,ppl",
        help=("Comma-separated preference order of metric keys to read from results['all']. " "Defaults to 'acc_norm,ppl'."),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory for plots (defaults to <experiment_dir>/evaluation_plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    short_benchmarks = ["mmlu_cloze", "hellaswag", "arc", "winogrande"]

    metric_preference = [k.strip() for k in args.metric_keys.split(",") if k.strip()]

    if args.benchmarks == "all":
        benchmarks: List[str] = []
    elif args.benchmarks == "short":
        benchmarks = short_benchmarks
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    plot_benchmark_over_checkpoints(
        experiment_dir=args.experiment_dir,
        benchmarks=benchmarks,
        metric_preference=metric_preference,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
