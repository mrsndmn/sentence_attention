import glob
import os

all_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    # "mmlu_pro_cloze",
    # "piqa",
    # "siqa",
    # "openbookqa",
    "winogrande",
    "pg19",
    # "gsm8k",
]

short_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    "winogrande",
    # "gsm8k",
    "pg19",  # long benchmark
]

long_benchmarks = [
    "recall",
    # "rag",
    "rerank",
    "cite",
    "longqa",
    "summ",
    "icl",
]


long_short_benchmarks_need_score_files = {
    "recall": 12,
    "rerank": 3,
    "cite": 6,
    "longqa": 9,
    "summ": 6,
    "icl": 15,
}


def checkpoint_evaluation_file(model_checkpoint, task_name, ruler_mode="tiny"):

    if task_name in long_benchmarks:
        need_score_files = 1
        if ruler_mode == "tiny":
            eval_dir = "helmet_eval"
        elif ruler_mode == "short":
            eval_dir = "helmet_eval_short"
            need_score_files = long_short_benchmarks_need_score_files[task_name]
        elif ruler_mode == "long":
            eval_dir = "helmet_eval_long"
            raise ValueError(f"Invalid ruler_mode: {ruler_mode}")
        else:
            raise ValueError(f"Invalid ruler_mode: {ruler_mode}")

        score_glob = os.path.join(model_checkpoint, eval_dir, task_name, "*.score")
        score_files = glob.glob(score_glob)
        if len(score_files) == 0:
            return score_glob

        if len(score_files) >= need_score_files:
            return score_files[0]
        else:
            print("Not enough score files", len(score_files), need_score_files)

        return score_glob

    evaluation_dir = os.path.join(model_checkpoint, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    return os.path.join(evaluation_dir, f"{task_name}.json")
