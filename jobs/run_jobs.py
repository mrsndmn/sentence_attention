import argparse
import glob
import math
import os
import random
import re
import string
import time
from copy import deepcopy
from typing import Any, Dict, List

import client_lib  # импортируем библиотеку для работы с ML Space
from mls.manager.job.utils import training_job_api_from_profile
from transformers.models.llama.extra_types import AVAILABLE_OPTIMIZED_PARAMS

from sentence_attention.artifacts.experiments import sort_checkpoints
from sentence_attention.integration.job import get_in_progress_jobs

# Defaults and constants
REGION = "SR004"
SEED = 1008
INSTANCE_TYPE = "a100.4gpu"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"

# Workspace root used inside jobs
workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"

# Required interpreter path policy
ENV_BIN = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"


def run_experiments(experiments: List[Dict], job_description: str = "", dry: bool = False) -> None:

    for exp in experiments:
        exp = deepcopy(exp)

        output_dir = exp.pop("output_dir")

        output_dir += f"_{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

        output_dir_full_path = os.path.join(workdir_prefix, "artifacts", "experiments_in_progress", output_dir)

        optimized_params = exp.pop("optimized_params")
        for param in optimized_params.split(","):
            assert param in AVAILABLE_OPTIMIZED_PARAMS, f"{param} is not in {AVAILABLE_OPTIMIZED_PARAMS}"

        warmup_steps = exp.pop("warmup_steps", 2000)
        num_train_epochs = exp.pop("num_train_epochs", 1)
        select_train_dataset_items = exp.pop("select_train_dataset_items", 150000)
        model_type = exp.pop("model_type", "pretrained")  # pretrained_checkpoint
        model_checkpoint = exp.pop("model_checkpoint", '""')
        number_of_eos_tokens = exp.pop("number_of_eos_tokens", 1)

        learning_rate = exp.pop("learning_rate", 1e-4)
        lr_scheduler_type = exp.pop("lr_scheduler_type", "cosine_with_min_lr")
        max_grad_norm = exp.pop("max_grad_norm", 1)
        max_steps = exp.pop("max_steps", -1)
        assert float(max_grad_norm) > 0

        optim = exp.pop("optim", "")
        if optim != "":
            optim = f"--optim {optim}"

        gradient_checkpointing = exp.pop("gradient_checkpointing", "0")

        logging_steps = exp.pop("logging_steps", "")
        if logging_steps == "":
            logging_steps = "1"

        logging_steps = f"--logging_steps {logging_steps}"

        per_device_train_batch_size = exp.pop("per_device_train_batch_size", 32)
        gradient_accumulation_steps = exp.pop("gradient_accumulation_steps", "")
        if gradient_accumulation_steps != "":
            gradient_accumulation_steps = f"--gradient_accumulation_steps {gradient_accumulation_steps}"

        adam_epsilon = exp.pop("adam_epsilon", "1e-8")
        if adam_epsilon != "":
            adam_epsilon = f"--adam_epsilon {adam_epsilon}"

        save_steps = exp.pop("save_steps", 5000)
        save_total_limit = exp.pop("save_total_limit", "")
        if save_total_limit != "":
            save_total_limit = f"--save_total_limit {save_total_limit}"
        torch_compile = exp.pop("torch_compile", 1)

        eval_strategy = exp.pop("eval_strategy", "no")
        weight_decay = exp.pop("weight_decay", "0.01")

        add_end_of_sentence_token = exp.pop("add_end_of_sentence_token", 0)

        eval_steps = exp.pop("eval_steps", 500)

        adam_beta1 = exp.pop("adam_beta1", "0.9")
        adam_beta2 = exp.pop("adam_beta2", "0.95")

        limit_dataset_shards = exp.pop("limit_dataset_shards", 0)
        offset_dataset_shards = exp.pop("offset_dataset_shards", 0)

        bf16 = exp.pop("bf16", 0)
        flexible_eos_tokens = exp.pop("flexible_eos_tokens", "0")
        ft_with_bos_token = exp.pop("ft_with_bos_token", "0")

        instance_type = exp.pop("instance_type", INSTANCE_TYPE)
        num_nodes = exp.pop("num_nodes", 1)
        fsdp = exp.pop("fsdp", False)

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        save_only_model = ""

        # accelerate_config = accelerate_config_by_instance_type(instance_type, workdir_prefix)

        seed = SEED

        # Normalize flags
        gradient_checkpointing_flag = "1" if str(gradient_checkpointing).lower() in {"1", "true", "yes"} else "0"

        # Use required full Python interpreter path and launch accelerate as a module

        if fsdp:
            script_prefix = f"bash {workdir_prefix}/jobs/prepare_multinode_accelerate_fsdp.sh"
        else:
            script_prefix = f"bash {workdir_prefix}/jobs/prepare_multinode_accelerate.sh"

        script_str = (
            f"{script_prefix} "
            f"{workdir_prefix}/scripts/train_sentence_llama.py "
            f"--save_strategy steps "
            f" --save_steps {save_steps} "
            f"--per_device_train_batch_size {per_device_train_batch_size} "
            f"--learning_rate {learning_rate} --max_grad_norm {max_grad_norm} "
            f"--num_train_epochs {num_train_epochs} --seed {seed} "
            f"--model_type {model_type} --model_checkpoint {model_checkpoint} "
            f"--adam_beta1 {adam_beta1} --adam_beta2 {adam_beta2} {adam_epsilon} "
            f"--lr_scheduler_type {lr_scheduler_type} "
            '--lr_scheduler_kwargs {\\"min_lr\\":0.00002} '
            f"--optimized_params {optimized_params} "
            f"--warmup_steps {warmup_steps} "
            f"--output_dir {output_dir_full_path} "
            f"--logging_dir {output_dir_full_path}/logs "
            f"--report_to tensorboard "
            f"--select_train_dataset_items {select_train_dataset_items} "
            f"--weight_decay {weight_decay} --bf16 {bf16} --torch_compile {torch_compile}  "
            f"{gradient_accumulation_steps} "
            f"--eval_strategy {eval_strategy} "
            f" --eval_steps {eval_steps} "
            f"--gradient_checkpointing {gradient_checkpointing_flag} {save_total_limit} {optim} {save_only_model} {logging_steps} "
            f"--add_end_of_sentence_token {add_end_of_sentence_token} --disable_tqdm 1 "
            f"--limit_dataset_shards {limit_dataset_shards} --offset_dataset_shards {offset_dataset_shards} "
            f"--number_of_eos_tokens {number_of_eos_tokens} "
            f"--flexible_eos_tokens {flexible_eos_tokens} "
            f"--ft_with_bos_token {ft_with_bos_token} "
            f"--max_steps {max_steps} "
        )

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type="binary",
            region=REGION,
            instance_type=instance_type,
            n_workers=num_nodes,
            processes_per_worker=1,
            job_desc=f"{job_description} #rnd #multimodality #tarasov #notify_completed @mrsndmn",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin",
                # "CLEARML_CONFIG_FILE": f"{workdir_prefix}/configs/clearml.conf",
                # "CLEARML_PROJECT": "sentence_attention",
                # "CLEARML_LOG_MODEL": "FALSE",
                "WANDB_MODE": "offline",
                # Always make sure PYTHONPATH and HF_HOME are set
                "PYTHONPATH": f"{workdir_prefix}/src:{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(output_dir, job_w_args.submit())

    return


def run_training_experiments(
    number_of_eos_tokens: int = 1,
    weight_decay: str = "0.01",
    num_train_epochs: int = 1,
    adam_epsilon: str = "1e-8",
    optimized_params: str = "full",
    learning_rate: float = 0.08,
    model_type: str = "pretrained",
    limit_dataset_shards: int = 0,
    offset_dataset_shards: int = 0,
    model_checkpoint: str = "unsloth/Meta-Llama-3.1-8B",
    select_train_dataset_items: int = 500000,
    lr_scheduler_type: str = "cosine_with_min_lr",
    instance_type: str = "a100.1gpu",
    num_nodes: int = 1,
    experiment_prefix_base_name: str = "adaptive_llama31_8B",
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    adam_beta1: str = "0.9",
    adam_beta2: str = "0.98",
    save_steps: int = 5000,
    save_total_limit: int = 3,
    per_device_train_batch_size: int = 4,
    optim: str = "adamw_torch_fused",
    torch_compile: str = "1",
    logging_steps: str = "",
    max_grad_norm: float = 1.0,
    warmup_steps: int = 2000,
    max_steps: int = -1,
    bf16: str = "0",
    add_end_of_sentence_token: str = "1",
    job_description=None,
    flexible_eos_tokens: str = "0",
    ft_with_bos_token: str = "0",
    fsdp: bool = False,
    **kwargs: Dict,
) -> None:

    assert job_description is not None, "job_description is required"

    common_params = {
        # Model
        "model_type": model_type,
        "model_checkpoint": model_checkpoint,
        "limit_dataset_shards": limit_dataset_shards,
        "offset_dataset_shards": offset_dataset_shards,
        "number_of_eos_tokens": number_of_eos_tokens,
        "optimized_params": optimized_params,
        "adam_epsilon": adam_epsilon,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "num_train_epochs": num_train_epochs,
        "gradient_checkpointing": gradient_checkpointing,
        "warmup_steps": warmup_steps,
        "add_end_of_sentence_token": add_end_of_sentence_token,
        # Data
        "select_train_dataset_items": select_train_dataset_items,
        "per_device_train_batch_size": per_device_train_batch_size,
        "logging_steps": logging_steps,
        "max_steps": max_steps,
        "bf16": bf16,
        "optim": optim,
        "torch_compile": torch_compile,
        # Training
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "max_grad_norm": max_grad_norm,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "instance_type": instance_type,
        "num_nodes": num_nodes,
        "flexible_eos_tokens": flexible_eos_tokens,
        "ft_with_bos_token": ft_with_bos_token,
        "fsdp": fsdp,
    }

    experiments = []

    exp_config = {
        **common_params,
    }

    exp_config["output_dir"] = experiment_prefix_base_name

    experiments.append(exp_config)

    run_experiments(experiments, job_description=job_description, **kwargs)

    return


def _models_for_eos_only() -> List[str]:
    return [
        {
            "model_checkpoint": "unsloth/Llama-3.2-1B",
        },
        {
            "model_checkpoint": "unsloth/Llama-3.2-3B",
        },
        {
            "model_checkpoint": "Qwen/Qwen2.5-1.5B",
        },
        {"model_checkpoint": "Qwen/Qwen2.5-3B", "per_device_train_batch_size": 1},
        {
            # "model_checkpoint": "unsloth/Meta-Llama-3.1-8B",
            "model_checkpoint": "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments_in_progress/sentence_Meta-Llama-3.1-8B_ft_only_eos_embedding_num_eos_tokens_4_CZO3YUYH/checkpoint-100/",
            "gradient_checkpointing": "0",
            "torch_compile": "1",
            "per_device_train_batch_size": 1,
        },
    ]


def _eos_tuned_checkpoints() -> List[Dict[str, Any]]:
    """Collect latest checkpoints from all EOS-tuned experiments.

    Scans `artifacts/experiments/eos_{1,4}` and for each experiment directory
    returns a dict with the latest checkpoint path and metadata. This does not
    filter by training success; callers may implement additional checks.
    """
    all_experiments: List[Dict[str, Any]] = []

    for number_of_eos_tokens in [1, 2, 4]:
        eos_dir = f"{workdir_prefix}/artifacts/experiments/eos_{number_of_eos_tokens}"

        if not os.path.exists(eos_dir):
            continue

        for experiment in os.listdir(eos_dir):
            if "_ft_only_eos_embedding_" not in experiment:
                continue
            if not experiment.startswith("sentence_"):
                continue

            experiment_path = f"{eos_dir}/{experiment}"

            last_checkpoint = sort_checkpoints(os.listdir(experiment_path))[0]

            model_slug = experiment.replace("sentence_", "")
            model_slug = re.sub(r"_ft_.*", "", model_slug)

            current_checkpoint = os.path.join(eos_dir, experiment, last_checkpoint)

            all_experiments.append(
                {
                    "model_checkpoint": current_checkpoint,
                    "model_slug": model_slug,
                    "number_of_eos_tokens": number_of_eos_tokens,
                    "per_device_train_batch_size": 4,
                }
            )

    return all_experiments


def _full_tuned_checkpoints() -> List[Dict[str, Any]]:
    """Collect latest checkpoints from all EOS-tuned experiments.

    Scans `artifacts/experiments/eos_{1,4}` and for each experiment directory
    returns a dict with the latest checkpoint path and metadata. This does not
    filter by training success; callers may implement additional checks.
    """
    all_experiments: List[Dict[str, Any]] = []

    for number_of_eos_tokens in [1, 2, 4]:
        eos_dir = f"{workdir_prefix}/artifacts/experiments/eos_{number_of_eos_tokens}"

        if not os.path.exists(eos_dir):
            continue

        for experiment in os.listdir(eos_dir):
            if "_ft_full_num_eos_tokens_" not in experiment:
                continue

            experiment_path = f"{eos_dir}/{experiment}"

            last_checkpoint = sort_checkpoints(os.listdir(experiment_path))[0]

            model_slug = experiment.replace("sentence_", "")
            model_slug = re.sub(r"_ft_.*", "", model_slug)

            all_experiments.append(
                {
                    "model_checkpoint": os.path.join(eos_dir, experiment, last_checkpoint),
                    "model_slug": model_slug,
                    "number_of_eos_tokens": number_of_eos_tokens,
                    "per_device_train_batch_size": 4,
                }
            )

    return all_experiments


def check_checkpoint_model_exists(experiment_prefix_base_name: str, number_of_eos_tokens: int) -> bool:
    """Check if there is exactly one matching experiment directory for the given base name.

    Returns True iff exactly one directory exists that starts with the base name.
    Returns False if none exist.
    Raises ValueError if multiple matches are found or if a name collision is detected.
    """
    experiment_prefix_base_name_full = (
        f"{workdir_prefix}/artifacts/experiments/eos_{number_of_eos_tokens}/{experiment_prefix_base_name}"
    )

    matches = glob.glob(experiment_prefix_base_name_full + "*")

    if len(matches) == 1:
        if matches[0].startswith(experiment_prefix_base_name_full):
            return True
        raise ValueError(f"Experiments names collision? Found for {experiment_prefix_base_name_full}")
    elif len(matches) == 0:
        return False
    else:
        raise ValueError(f"Multiple experiments found for {experiment_prefix_base_name_full}: {matches}")


def check_experiment_in_progress(experiment_prefix_base_name: str, in_progress_jobs: List[Dict]) -> bool:

    experiment_in_progress = False
    for job in in_progress_jobs:
        if experiment_prefix_base_name in job["job_desc"]:
            experiment_in_progress = True
            break

    return experiment_in_progress


def run_group_eos_only(*, dry: bool, num_eos_tokens: List[int], in_progress_jobs: List[Dict], model: str) -> None:
    ngpus = 8
    n_nodes = 6
    num_train_epochs = 1
    per_device_train_batch_size = 4
    save_steps = 100
    optimized_params = "only_eos_embedding"
    limit_dataset_shards = 2
    # lr_scheduler_type = "constant"

    max_steps = -1

    for number_of_eos_tokens in num_eos_tokens:

        for exp_config in _models_for_eos_only():
            model_checkpoint = exp_config["model_checkpoint"]
            gradient_checkpointing = exp_config.get("gradient_checkpointing", None)
            torch_compile = exp_config.get("torch_compile", None)
            local_per_device_train_batch_size = exp_config.get("per_device_train_batch_size", per_device_train_batch_size)

            extra_kwargs = {}
            if gradient_checkpointing is not None:
                extra_kwargs["gradient_checkpointing"] = gradient_checkpointing

            if torch_compile is not None:
                extra_kwargs["torch_compile"] = torch_compile

            if model is not None and model.lower() not in model_checkpoint.lower():
                continue

            # TODO check sucessful experiment has already been processed

            model_checkpoint_slug = model_checkpoint.split("/")[-1]
            gradient_accumulation_steps = math.ceil(1024 / ngpus / n_nodes / local_per_device_train_batch_size)

            model_dir_prefix = f"sentence_{model_checkpoint_slug}_ft_{optimized_params}"

            if check_checkpoint_model_exists(model_dir_prefix, number_of_eos_tokens):
                print(f"Experiment eos_{number_of_eos_tokens} / {model_dir_prefix} already exists")
                continue

            experiment_prefix_base_name = f"{model_dir_prefix}_num_eos_tokens_{number_of_eos_tokens}"
            job_description = f"ST: {experiment_prefix_base_name}"

            if check_experiment_in_progress(experiment_prefix_base_name, in_progress_jobs):
                print(f"Experiment {experiment_prefix_base_name} is already in progress")
                continue

            run_training_experiments(
                learning_rate=0.0001,
                model_type="sentence_pretrained_checkpoint",
                limit_dataset_shards=limit_dataset_shards,
                number_of_eos_tokens=number_of_eos_tokens,
                optimized_params=optimized_params,
                weight_decay="0.01",
                per_device_train_batch_size=local_per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                adam_beta1="0.9",
                adam_beta2="0.95",
                optim="adamw_torch_fused",
                num_train_epochs=num_train_epochs,
                max_grad_norm="1.0",
                save_total_limit=100,
                save_steps=save_steps,
                max_steps=max_steps,
                instance_type=f"a100.{ngpus}gpu",
                num_nodes=n_nodes,
                model_checkpoint=model_checkpoint,
                select_train_dataset_items=0,
                adam_epsilon="1e-8",
                warmup_steps=10,
                dry=dry,
                bf16="0",
                add_end_of_sentence_token=1,
                experiment_prefix_base_name=experiment_prefix_base_name,
                job_description=job_description,
                **extra_kwargs,
            )


def run_group_full_4k(
    *,
    dry: bool,
    num_eos_tokens: List[int],
    in_progress_jobs: List[Dict],
    model: str,
    flexible_eos_tokens: bool = False,
    ft_with_bos_token: bool = False,
) -> None:
    ngpus = 4
    num_nodes = 14

    num_train_epochs = 1
    save_steps = 200
    optimized_params = "full"
    max_grad_norm = "2.0"

    default_limit_shards = 2

    for exp_config in _eos_tuned_checkpoints():
        # TODO check sucessful experiment has already been processed
        model_checkpoint = exp_config["model_checkpoint"]
        model_slug = exp_config["model_slug"]
        per_device_train_batch_size = exp_config["per_device_train_batch_size"]
        if per_device_train_batch_size != 1:
            print("Force per_device_train_batch_size to 1")
            per_device_train_batch_size = 1

        number_of_eos_tokens = exp_config["number_of_eos_tokens"]

        local_limit_shards = exp_config.get("limit_dataset_shards", default_limit_shards)

        extra_kwargs = {}
        fsdp = None
        local_torch_compile = None
        if "Llama-3.1-8B" in model_checkpoint:
            fsdp = "1"
            local_torch_compile = "0"

        if fsdp is not None:
            extra_kwargs["fsdp"] = fsdp

        if local_torch_compile is not None:
            extra_kwargs["torch_compile"] = local_torch_compile

        local_save_steps = exp_config.get("save_steps", save_steps)

        if model is not None and model.lower() not in model_checkpoint.lower():
            continue

        if int(number_of_eos_tokens) not in num_eos_tokens:
            continue

        model_dir_prefix_mid = "_ft_4k_"
        if flexible_eos_tokens:
            model_dir_prefix_mid = f"{model_dir_prefix_mid}flexible_eos_tokens_"

        if ft_with_bos_token:
            model_dir_prefix_mid = f"{model_dir_prefix_mid}bos_token_"

        model_dir_prefix = f"sentence_{model_slug}{model_dir_prefix_mid}{optimized_params}"

        if check_checkpoint_model_exists(model_dir_prefix, number_of_eos_tokens):
            print(f"Experiment eos_{number_of_eos_tokens} / {model_dir_prefix} already exists")
            continue

        # gradient_accumulation_steps = math.ceil(1024 / ngpus / num_nodes / per_device_train_batch_size)
        gradient_accumulation_steps = math.ceil(1024 / ngpus / num_nodes / per_device_train_batch_size)

        experiment_prefix_base_name = f"{model_dir_prefix}_num_eos_tokens_{number_of_eos_tokens}"
        job_description = f"ST: {experiment_prefix_base_name}"

        if check_experiment_in_progress(experiment_prefix_base_name, in_progress_jobs):
            print(f"Experiment {experiment_prefix_base_name} is already in progress")
            continue

        run_training_experiments(
            learning_rate=0.00005,
            model_type="sentence_pretrained_checkpoint",
            # Rertain on EOSo data
            limit_dataset_shards=local_limit_shards,
            offset_dataset_shards=4,
            number_of_eos_tokens=number_of_eos_tokens,
            optimized_params=optimized_params,
            weight_decay="0.01",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            adam_beta1="0.9",
            adam_beta2="0.95",
            optim="adamw_torch_fused",
            num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm,
            save_total_limit=100,
            save_steps=local_save_steps,
            instance_type=f"a100.{ngpus}gpu",
            num_nodes=num_nodes,
            model_checkpoint=model_checkpoint,
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=200,
            dry=dry,
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=experiment_prefix_base_name,
            job_description=job_description,
            flexible_eos_tokens="1" if flexible_eos_tokens else "0",
            ft_with_bos_token="1" if ft_with_bos_token else "0",
            **extra_kwargs,
        )


def run_group_full_4k_distill_from_4eos_tokens(
    *,
    dry: bool,
    num_eos_tokens: List[int],
    in_progress_jobs: List[Dict],
    model: str,
    flexible_eos_tokens: bool = False,
    ft_with_bos_token: bool = False,
) -> None:

    ngpus = 8
    num_nodes = 4
    num_train_epochs = 1
    save_steps = 1000
    optimized_params = "full"
    max_grad_norm = "2.0"

    all_experiments = []
    all_experiments.extend(
        [
            {
                "model_checkpoint": "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-10794/",
                "model_slug": "Llama-3.2-3B",
                "number_of_eos_tokens": 2,
                "per_device_train_batch_size": 4,
            },
            {
                "model_checkpoint": "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_62XMQ139/checkpoint-10794/",
                "model_slug": "Llama-3.2-3B",
                "number_of_eos_tokens": 1,
                "per_device_train_batch_size": 4,
            },
        ]
    )

    for exp_config in all_experiments:
        # TODO check sucessful experiment has already been processed
        model_checkpoint = exp_config["model_checkpoint"]
        model_slug = exp_config["model_slug"]
        per_device_train_batch_size = exp_config["per_device_train_batch_size"]
        if per_device_train_batch_size != 1:
            print("Force per_device_train_batch_size to 1")
            per_device_train_batch_size = 1

        number_of_eos_tokens = exp_config["number_of_eos_tokens"]

        if model is not None and model.lower() not in model_checkpoint.lower():
            continue

        if int(number_of_eos_tokens) not in num_eos_tokens:
            continue

        model_dir_prefix_mid = "_ft_4k_distill_"
        if flexible_eos_tokens:
            model_dir_prefix_mid = f"{model_dir_prefix_mid}flexible_eos_tokens_"

        if ft_with_bos_token:
            model_dir_prefix_mid = f"{model_dir_prefix_mid}bos_token_"

        model_dir_prefix = f"sentence_{model_slug}{model_dir_prefix_mid}{optimized_params}"

        if check_checkpoint_model_exists(model_dir_prefix, number_of_eos_tokens):
            print(f"Experiment eos_{number_of_eos_tokens} / {model_dir_prefix} already exists")
            continue

        # gradient_accumulation_steps = math.ceil(4096 / ngpus / num_nodes / per_device_train_batch_size)
        gradient_accumulation_steps = math.ceil(512 / ngpus / num_nodes / per_device_train_batch_size)

        experiment_prefix_base_name = f"{model_dir_prefix}_num_eos_tokens_{number_of_eos_tokens}"
        job_description = f"ST: {experiment_prefix_base_name}"

        if check_experiment_in_progress(experiment_prefix_base_name, in_progress_jobs):
            print(f"Experiment {experiment_prefix_base_name} is already in progress")
            continue

        run_training_experiments(
            learning_rate=0.00005,
            model_type="sentence_pretrained_checkpoint",
            # Rertain on EOSo data
            limit_dataset_shards=10,
            offset_dataset_shards=0,
            number_of_eos_tokens=number_of_eos_tokens,
            optimized_params=optimized_params,
            weight_decay="0.01",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            adam_beta1="0.9",
            adam_beta2="0.95",
            optim="adamw_torch_fused",
            num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm,
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{ngpus}gpu",
            num_nodes=num_nodes,
            model_checkpoint=model_checkpoint,
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=1000,
            dry=dry,
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=experiment_prefix_base_name,
            job_description=job_description,
            flexible_eos_tokens="1" if flexible_eos_tokens else "0",
            ft_with_bos_token="1" if ft_with_bos_token else "0",
        )


def run_group_lora(*, dry: bool, num_eos_tokens: List[int], in_progress_jobs: List[Dict], model: str) -> None:
    ngpus = 8
    num_train_epochs = 1
    save_steps = 250
    optimized_params = "lora"

    for exp_config in _eos_tuned_checkpoints():
        # TODO check sucessful experiment has already been processed
        if "llama-3-8b" not in exp_config["model_checkpoint"]:
            continue

        model_checkpoint = exp_config["model_checkpoint"]
        model_slug = exp_config["model_slug"]
        per_device_train_batch_size = exp_config["per_device_train_batch_size"]
        number_of_eos_tokens = exp_config["number_of_eos_tokens"]

        if model is not None and model.lower() not in model_checkpoint.lower():
            continue

        if int(number_of_eos_tokens) not in num_eos_tokens:
            continue

        model_dir_prefix = f"sentence_{model_slug}_ft_{optimized_params}"
        if check_checkpoint_model_exists(model_dir_prefix, number_of_eos_tokens):
            print(f"Experiment eos_{number_of_eos_tokens} / {model_dir_prefix} already exists")
            continue

        per_device_train_batch_size = max(per_device_train_batch_size, 8)

        gradient_accumulation_steps = math.ceil(4096 / ngpus / per_device_train_batch_size)

        experiment_prefix_base_name = f"{model_dir_prefix}_num_eos_tokens_{number_of_eos_tokens}"
        job_description = f"ST: {experiment_prefix_base_name}"

        if check_experiment_in_progress(experiment_prefix_base_name, in_progress_jobs):
            print(f"Experiment {experiment_prefix_base_name} is already in progress")
            continue

        run_training_experiments(
            learning_rate=0.0001,
            model_type="sentence_pretrained_checkpoint",
            limit_dataset_shards=8,
            offset_dataset_shards=4,
            number_of_eos_tokens=number_of_eos_tokens,
            optimized_params=optimized_params,
            weight_decay="0.01",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            adam_beta1="0.9",
            adam_beta2="0.95",
            optim="adamw_torch_fused",
            num_train_epochs=num_train_epochs,
            max_grad_norm="1.0",
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{ngpus}gpu",
            model_checkpoint=model_checkpoint,
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=100,
            dry=dry,
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=experiment_prefix_base_name,
            job_description=job_description,
        )


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit sentence-attention training jobs")

    parser.add_argument(
        "--group",
        choices=[
            "eos-only",
            "full",
            "full2",
            "full_4k",
            "full_4k_distill_from_4eos_tokens",
            "lora",
            "full-flexible-eos-tokens",
            "ft-with-bos-token",
        ],
        required=True,
        help="Which experiment group to run",
    )
    parser.add_argument(
        "--num_eos_tokens", type=int, default=None, help="Number of EOS tokens to use, default is [1, 4, 8, 16]"
    )
    parser.add_argument("--dry", action="store_true")

    parser.add_argument("--wait", type=str, help="Job ID to wait for")
    parser.add_argument("--model", type=str, help="Model checkpoint filter")

    return parser


def main() -> None:
    parser = _cli()
    args = parser.parse_args()

    num_eos_tokens = [1, 2, 4] if args.num_eos_tokens is None else [args.num_eos_tokens]

    if args.wait is not None:
        job_id = args.wait
        print("Waiting for jobs to finish", job_id)
        while True:
            job = None
            try:
                client, _ = training_job_api_from_profile("default")
                job = client.get_job_status(job_id)
                print("Job info", job)
                print("Job status", job["status"])  # type: ignore[index]
            except Exception as e:
                print("Error", e)
                time.sleep(10)

            if job is not None and "status" in job and job["status"].lower() in ["completed", "failed", "stopped"]:  # type: ignore[index]
                break
            print("Waiting for jobs to finish", job_id)
            time.sleep(10)
        print("Job finished", job_id)

    client, _ = training_job_api_from_profile("default")
    in_progress_jobs = get_in_progress_jobs(client)

    if args.group == "eos-only":
        run_group_eos_only(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
        )
    elif args.group == "full_4k":
        run_group_full_4k(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
        )
    elif args.group == "full_4k_distill_from_4eos_tokens":
        run_group_full_4k_distill_from_4eos_tokens(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
        )
    elif args.group == "lora":
        run_group_lora(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
        )
    elif args.group == "full-flexible-eos-tokens":
        run_group_full_4k(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
            flexible_eos_tokens=True,
        )
    elif args.group == "ft-with-bos-token":
        run_group_full_4k(
            dry=args.dry,
            num_eos_tokens=num_eos_tokens,
            in_progress_jobs=in_progress_jobs,
            model=args.model,
            ft_with_bos_token=True,
        )

    return


if __name__ == "__main__":
    main()
