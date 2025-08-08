import os

per_task_default_params = {
    "hellaswag": {
        "override_batch_size": 512,
    },
    "mmlu_cloze": {
        "override_batch_size": 64,
    },
    "mmlu_pro_cloze": {
        "override_batch_size": 64,
    },
    "piqa": {
        "override_batch_size": 128,
    },
    "siqa": {
        "override_batch_size": 512,
    },
    "openbookqa": {
        "override_batch_size": 256,
    },
    "winogrande": {
        "override_batch_size": 512,
    },
    "arc": {
        "override_batch_size": 64,
    },
}

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

assert set(all_benchmarks) == set(
    per_task_default_params.keys()
), "all_benchmarks and per_task_default_params must have the same keys"


def checkpoint_evaluation_file(model_checkpoint, task_name):
    evaluation_dir = os.path.join(model_checkpoint, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    return os.path.join(evaluation_dir, f"{task_name}.json")
