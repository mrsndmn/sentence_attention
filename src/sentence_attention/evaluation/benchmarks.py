import glob
import os

all_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    "mmlu_pro_cloze",
    "piqa",
    "siqa",
    "openbookqa",
    "winogrande",
    "pg19",
    "gsm8k",
]

short_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    "winogrande",
    "gsm8k",
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


def checkpoint_evaluation_file(model_checkpoint, task_name, ruler_mode="tiny"):

    if task_name in long_benchmarks:
        if ruler_mode == "tiny":
            eval_dir = "helmet_eval"
        elif ruler_mode == "short":
            eval_dir = "helmet_eval_short"
        else:
            raise ValueError(f"Invalid ruler_mode: {ruler_mode}")

        score_glob = os.path.join(model_checkpoint, eval_dir, task_name, "*.score")
        score_files = glob.glob(score_glob)
        if len(score_files) == 0:
            return score_glob

        return score_files[0]

    evaluation_dir = os.path.join(model_checkpoint, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    return os.path.join(evaluation_dir, f"{task_name}.json")
