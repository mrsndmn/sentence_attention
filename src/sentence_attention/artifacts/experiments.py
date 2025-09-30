import os
import re

WORKDIR_PREFIX = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"
EXPERIMENTS_DIR = os.path.join(WORKDIR_PREFIX, "artifacts", "experiments")
EXPERIMENTS_IN_PROGRESS_DIR = os.path.join(WORKDIR_PREFIX, "artifacts", "experiments_in_progress")


def sort_checkpoints(checkpoints: list[str]):

    checkpoints_dirs = []

    for checkpoint in checkpoints:
        if checkpoint in ["evaluation_plots", "logs"]:
            continue

        if not checkpoint.startswith("checkpoint-"):
            continue
        checkpoints_dirs.append(checkpoint)

    return sorted(checkpoints_dirs, key=lambda x: int(x.split("-")[-1]), reverse=True)


def extract_eos_tokens_num(eos_tokens_num: str):
    if eos_tokens_num.startswith("eos_"):
        return int(eos_tokens_num.split("_")[1])
    return eos_tokens_num


def extract_eos_tokens_num_from_experiment_dir(experiment_name: str):
    return int(re.findall(r"_num_eos_tokens_(\d+)", experiment_name)[0])


def get_all_checkpoints_in_progress(eos_tokens_num=None, limit_checkpoints_num=None, model: list[str] | None = None):
    all_checkpoints = []
    base_dir = EXPERIMENTS_IN_PROGRESS_DIR

    for experiment_name in os.listdir(base_dir):
        eos_num_parsed = extract_eos_tokens_num_from_experiment_dir(experiment_name)
        if eos_tokens_num is not None and eos_num_parsed != eos_tokens_num:
            continue

        # Optional substring filtering by model name(s)
        if model:
            name_lower = experiment_name.lower()
            if not any(substr.lower() in name_lower for substr in model):
                continue

        experiment_eval_dir = os.listdir(os.path.join(base_dir, experiment_name))

        result_checkpoints = sort_checkpoints(experiment_eval_dir)
        if limit_checkpoints_num is not None:
            result_checkpoints = result_checkpoints[:limit_checkpoints_num]

        for checkpoint in result_checkpoints:
            all_checkpoints.append(
                {
                    "eos_tokens_num": eos_num_parsed,
                    "experiment_name": experiment_name,
                    "checkpoint": checkpoint,
                    "full_path": os.path.join(base_dir, experiment_name, checkpoint),
                }
            )

    return all_checkpoints


def get_all_checkpoints(eos_tokens_num=None, limit_checkpoints_num=None, model: list[str] | None = None):
    all_checkpoints = []
    base_dir = EXPERIMENTS_DIR
    for eos_dir in os.listdir(base_dir):
        if eos_dir == "bad_multi_eos_experiments":
            continue

        eos_dir_parsed = extract_eos_tokens_num(eos_dir)
        if eos_tokens_num is not None and eos_dir_parsed != eos_tokens_num:
            continue

        for experiment_name in os.listdir(os.path.join(base_dir, eos_dir)):
            # Optional substring filtering by model name(s)
            if model:
                name_lower = experiment_name.lower()
                if not any(substr.lower() in name_lower for substr in model):
                    continue

            experiment_eval_dir = os.listdir(os.path.join(base_dir, eos_dir, experiment_name))

            result_checkpoints = sort_checkpoints(experiment_eval_dir)
            if limit_checkpoints_num is not None:
                result_checkpoints = result_checkpoints[:limit_checkpoints_num]

            for checkpoint in result_checkpoints:
                all_checkpoints.append(
                    {
                        "eos_tokens_num": eos_dir_parsed,
                        "experiment_name": experiment_name,
                        "checkpoint": checkpoint,
                        "full_path": os.path.join(base_dir, eos_dir, experiment_name, checkpoint),
                    }
                )

    return all_checkpoints


def get_all_last_checkpoints(eos_tokens_num=None, in_progress: bool = False, model: list[str] | None = None):

    if in_progress:
        return get_all_checkpoints_in_progress(eos_tokens_num, limit_checkpoints_num=1, model=model)
    else:
        return get_all_checkpoints(eos_tokens_num, limit_checkpoints_num=1, model=model)
