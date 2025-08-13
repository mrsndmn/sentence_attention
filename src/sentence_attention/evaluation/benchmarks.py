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
]

short_benchmarks = [
    "arc",
    "hellaswag",
    "mmlu_cloze",
    "winogrande",
]


def checkpoint_evaluation_file(model_checkpoint, task_name):
    evaluation_dir = os.path.join(model_checkpoint, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    return os.path.join(evaluation_dir, f"{task_name}.json")
