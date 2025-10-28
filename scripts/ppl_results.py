import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt


def find_pg19_files(root: str, model=None) -> List[str]:
    pattern = os.path.join(root, "artifacts", "**", "evaluation", "pg19.json")
    files = sorted(glob(pattern, recursive=True))
    if model:
        model_patterns = model.split(",")
        files_filtered = []
        for file in files:
            for model_pattern in model_patterns:
                if model_pattern in file:
                    files_filtered.append(file)
                    break

        files = files_filtered
    return files


def load_aggregated_ppl_by_prefix(path: str, no_eos: bool = False) -> Tuple[List[int], List[float]]:
    try:
        with open(path) as f:
            data = json.load(f)

        data_key = "aggregated_ppl_by_prefix_no_eos" if no_eos else "aggregated_ppl_by_prefix"
        agg: Dict[str, Dict[str, float]] = data.get(data_key, {})
        # Keys are prefixes as strings, sort numerically
        xs = sorted(int(k) for k in agg)
        ys = [float(agg[str(x)]["ppl"]) for x in xs]
        return xs, ys
    except Exception:
        return [], []


def short_label_from_path(path: str) -> str:
    # Expect .../experiments/eos_*/<experiment_name>/checkpoint-XXXX/evaluation/pg19.json
    parts = path.split(os.sep)

    idx_eval = parts.index("evaluation")
    # checkpoint_dir = parts[idx_eval - 1]  # checkpoint-XXXX
    experiment_dir = parts[idx_eval - 2]  # experiment name
    # eos_dir = parts[idx_eval - 4]  # eos_*
    # Compact label: <eos>/<exp>/<step>
    # step = re.sub(r"checkpoint-\\d+", "", checkpoint_dir)
    exp_short = experiment_dir.replace("sentence_", "")
    exp_short = exp_short.removeprefix("experiments/")
    exp_short = exp_short.replace("_base_model", "")

    if "_ft_4k_colddown_full_num_eos_tokens_" in exp_short:
        exp_short = exp_short.replace("_ft_4k_colddown_full_num_eos_tokens_", " Ng=")
        exp_short = "_".join(exp_short.split("_")[:-1])

    print("exp_short", exp_short)
    return exp_short


def plot_pg19(files: List[str], output_path: str, log_scale: bool = False, model: str = None) -> None:
    if not files:
        print("No pg19.json files found. Nothing to plot.")
        return

    matplotlib.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(12, 12))
    fint_size = 25
    matplotlib.rcParams.update({"font.size": fint_size})

    for fp in files:
        xs, ys = load_aggregated_ppl_by_prefix(fp)
        if not xs:
            continue
        label = short_label_from_path(fp)
        plt.plot(xs, ys, label=label, linewidth=5.0)

        xs_no_eos, ys_no_eos = load_aggregated_ppl_by_prefix(fp, no_eos=True)
        if not xs_no_eos:
            continue
        plt.plot(xs_no_eos, ys_no_eos, label=label + " no Gist Tokens", linewidth=5.0)

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Prefix length (tokens)")
    plt.ylabel("Aggregated PPL")
    plt.xlim(0, 32000)
    plt.title("PG19 aggregated PPL")
    plt.legend(fontsize=fint_size, ncol=1, frameon=True)
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name",
    )
    args = parser.parse_args()

    files = find_pg19_files(args.root, model=args.model)

    output_file = os.path.join(args.output, "pg19_ppl.pdf")
    plot_pg19(files, output_file, log_scale=False)
    output_file = os.path.join(args.output, "pg19_ppl.png")
    plot_pg19(files, output_file, log_scale=False)

    output_file = os.path.join(args.output, "pg19_ppl_log.pdf")
    plot_pg19(files, output_file, log_scale=True)
    output_file = os.path.join(args.output, "pg19_ppl_log.png")
    plot_pg19(files, output_file, log_scale=True)


if __name__ == "__main__":
    main()
