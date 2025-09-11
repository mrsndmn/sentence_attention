import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt


def find_pg19_files(root: str) -> List[str]:
    pattern = os.path.join(root, "artifacts", "**", "evaluation", "pg19.json")
    return sorted(glob(pattern, recursive=True))


def load_aggregated_ppl_by_prefix(path: str) -> Tuple[List[int], List[float]]:
    try:
        with open(path) as f:
            data = json.load(f)
        agg: Dict[str, Dict[str, float]] = data.get("aggregated_ppl_by_prefix", {})
        # Keys are prefixes as strings, sort numerically
        xs = sorted(int(k) for k in agg)
        ys = [float(agg[str(x)]["ppl"]) for x in xs]
        return xs, ys
    except Exception:
        return [], []


def short_label_from_path(path: str) -> str:
    # Expect .../experiments/eos_*/<experiment_name>/checkpoint-XXXX/evaluation/pg19.json
    parts = path.split(os.sep)
    try:
        idx_eval = parts.index("evaluation")
        checkpoint_dir = parts[idx_eval - 1]  # checkpoint-XXXX
        experiment_dir = parts[idx_eval - 2]  # experiment name
        eos_dir = parts[idx_eval - 4]  # eos_*
        # Compact label: <eos>/<exp>/<step>
        step = checkpoint_dir.replace("checkpoint-", "")
        exp_short = experiment_dir.replace("sentence_", "")
        return f"{eos_dir}/{exp_short}/{step}"
    except Exception:
        return os.path.basename(os.path.dirname(os.path.dirname(path)))


def plot_pg19(files: List[str], output_path: str, log_scale: bool = False) -> None:
    if not files:
        print("No pg19.json files found. Nothing to plot.")
        return

    matplotlib.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 6))

    for fp in files:
        xs, ys = load_aggregated_ppl_by_prefix(fp)
        if not xs:
            continue
        label = short_label_from_path(fp)
        plt.plot(xs, ys, label=label)

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Prefix length (tokens)")
    plt.ylabel("Aggregated PPL")
    plt.xlim(1024, 32000)
    plt.title("PG19 aggregated PPL by prefix across checkpoints")
    plt.legend(fontsize=7, ncol=1, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PG19 aggregated PPL across checkpoints")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory containing artifacts/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/evaluation/",
        help="Output PNG path",
    )
    args = parser.parse_args()

    files = find_pg19_files(args.root)

    output_file = os.path.join(args.output, "pg19_ppl.png")
    plot_pg19(files, output_file, log_scale=False)

    output_file = os.path.join(args.output, "pg19_ppl_log.png")
    plot_pg19(files, output_file, log_scale=True)


if __name__ == "__main__":
    main()
