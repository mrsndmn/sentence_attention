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
]

short_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    "winogrande",
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


def checkpoint_evaluation_file(model_checkpoint, task_name):

    if task_name in long_benchmarks:
        score_glob = os.path.join(model_checkpoint, "helmet_eval", task_name, "*.score")
        score_files = glob.glob(score_glob)
        if len(score_files) == 0:
            return score_glob

        return score_files[0]

    evaluation_dir = os.path.join(model_checkpoint, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    return os.path.join(evaluation_dir, f"{task_name}.json")
