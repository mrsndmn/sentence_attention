import math
import time
import string
import random
import client_lib  # импортируем библиотеку для работы с ML Space

from rich.console import Console

import os

from copy import deepcopy

from transformers.models.llama.extra_types import AVAILABLE_OPTIMIZED_PARAMS

from sentence_attention.integration.job import accelerate_config_by_instance_type

REGION = "SR004"

SEED = 1008

INSTANCE_TYPE = "a100.4gpu"
N_WORKERS = 1
# BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"


def run_experiments(experiments, job_description_prefix="", dry=False):

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
        lr_scheduler_type = exp.pop("lr_scheduler_type", "cosine")
        max_grad_norm = exp.pop("max_grad_norm", 1)

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

        instance_type = exp.pop("instance_type", INSTANCE_TYPE)

        eval_strategy = exp.pop("eval_strategy", "steps")
        weight_decay = exp.pop("weight_decay", "0.01")

        add_end_of_sentence_token = exp.pop("add_end_of_sentence_token", 0)

        eval_steps = exp.pop("eval_steps", 500)

        adam_beta1 = exp.pop("adam_beta1", "0.9")
        adam_beta2 = exp.pop("adam_beta2", "0.95")

        limit_dataset_shards = exp.pop("limit_dataset_shards", 0)
        offset_dataset_shards = exp.pop("offset_dataset_shards", 0)

        bf16 = exp.pop("bf16", 0)

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        save_only_model = ""

        accelerate_config = accelerate_config_by_instance_type(instance_type, workdir_prefix)

        seed = SEED

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file {accelerate_config} {workdir_prefix}/scripts/train_sentence_llama.py --save_strategy steps --save_steps {save_steps} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --max_grad_norm {max_grad_norm} --num_train_epochs {num_train_epochs} --seed {seed} --model_type {model_type} --model_checkpoint {model_checkpoint} --adam_beta1 {adam_beta1} --adam_beta2 {adam_beta2} {adam_epsilon} --lr_scheduler_type {lr_scheduler_type} --optimized_params {optimized_params} --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --select_train_dataset_items {select_train_dataset_items} --weight_decay {weight_decay} --bf16 {bf16} --torch_compile {torch_compile}  {gradient_accumulation_steps}  --eval_strategy {eval_strategy} --eval_steps {eval_steps} --gradient_checkpointing {gradient_checkpointing} {save_total_limit} {optim} {save_only_model} {logging_steps} --add_end_of_sentence_token {add_end_of_sentence_token} --disable_tqdm 1 --limit_dataset_shards {limit_dataset_shards} --offset_dataset_shards {offset_dataset_shards} --number_of_eos_tokens {number_of_eos_tokens}"

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type="binary",
            region=REGION,
            instance_type=instance_type,
            n_workers=N_WORKERS,
            processes_per_worker=1,
            job_desc=f"{job_description_prefix}{output_dir} #rnd #multimodality #tarasov @mrsndmn",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin",
                "CLEARML_CONFIG_FILE": f"{workdir_prefix}/configs/clearml.conf",
                "CLEARML_PROJECT": "sentence_attention",
                "CLEARML_LOG_MODEL": "FALSE",
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
    number_of_eos_tokens=1,
    weight_decay="0.01",
    num_train_epochs=1,
    adam_epsilon="1e-8",
    optimized_params="full",
    learning_rate=0.08,
    model_type="pretrained",
    limit_dataset_shards=0,
    offset_dataset_shards=0,
    model_checkpoint="unsloth/Meta-Llama-3.1-8B",
    select_train_dataset_items=500000,
    lr_scheduler_type="constant_with_warmup",
    instance_type="a100.1gpu",
    experiment_prefix_base_name="adaptive_llama31_8B",
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    adam_beta1="0.9",
    adam_beta2="0.98",
    save_steps=5000,
    save_total_limit=3,
    per_device_train_batch_size=4,
    optim="adamw_torch_fused",
    torch_compile="1",
    logging_steps="",
    max_grad_norm=1.0,
    warmup_steps=2000,
    bf16="0",
    dataset=None,
    add_end_of_sentence_token="1",
    **kwargs,
):

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
        "dataset": dataset,
        "num_train_epochs": num_train_epochs,
        "gradient_checkpointing": gradient_checkpointing,
        "warmup_steps": warmup_steps,
        "add_end_of_sentence_token": add_end_of_sentence_token,
        # Data
        "select_train_dataset_items": select_train_dataset_items,
        "per_device_train_batch_size": per_device_train_batch_size,
        "logging_steps": logging_steps,
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
    }

    experiments = []

    exp_config = {
        **common_params,
    }

    exp_config["output_dir"] = f"{experiment_prefix_base_name}"

    experiments.append(exp_config)

    run_experiments(experiments, job_description_prefix="ST: ", **kwargs)

    return


if __name__ == "__main__":

    import sys

    console = Console()
    console.print(client_lib.get_instance_types(regions="SR004"))

    dry = len(sys.argv) > 1 and sys.argv[1] == "dry"
    print("dry", dry)

    if len(sys.argv) > 1 and sys.argv[1] == "wait":
        from mls.manager.job.utils import training_job_api_from_profile

        assert len(sys.argv) == 3, "Usage: python run_jobs.py wait <job_id>"
        job_id = sys.argv[2]
        print("Waiting for jobs to finish", job_id)

        while True:
            job = None
            try:
                client, extra_options = training_job_api_from_profile("default")
                job = client.get_job_status(job_id)
                print("Job info", job)
            except Exception as e:
                print("Error", e)
                time.sleep(60)

            print("Job status", job["status"])
            if job is not None and job["status"].lower() == "completed":
                break

            print("Waiting for jobs to finish", job_id)
            time.sleep(10)

        print("Job finished", job_id)

    # if not dry:
    #     tests_run = subprocess.run(["pytest", "src/transformers/models/llama/tests/"])
    #     if tests_run.returncode != 0:
    #         print("Tests failed")
    #         exit(1)

    # Train only one embedding param
    NGPUS = 4
    num_train_epochs = 1
    per_device_train_batch_size = 8
    gradient_accumulation_steps = math.ceil(128 / NGPUS)
    save_steps = 500

    # if True:
    if False:
        run_training_experiments(
            learning_rate=0.0001,
            model_type="sentence_pretrained_checkpoint",
            # optimized_params='full',
            optimized_params="only_eos_embedding",
            weight_decay="0.0",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{NGPUS}gpu",
            # model_checkpoint=f'{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_token_w_0.100_l__S9M6BLOE/checkpoint-400/',
            # model_checkpoint=f'{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_token_w_0.100_l__5RHWG2LO/checkpoint-1000/',
            model_checkpoint="HuggingFaceTB/SmolLM2-1.7B",
            dataset="smollm-corpus",
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=1000,
            dry=dry,
            lr_scheduler_type="cosine",
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name="sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd",
        )

    # Train full params
    NGPUS = 4
    num_train_epochs = 1
    per_device_train_batch_size = 4
    save_steps = 250

    # models_checkpoints = [ 'unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B', 'HuggingFaceTB/SmolLM2-1.7B', 'unsloth/Llama-3.2-3B', 'Qwen/Qwen2.5-3B',  ]
    # models_checkpoints = [ 'unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B' ]
    models_checkpoints = ["unsloth/Llama-3.2-1B"]

    # 3B
    # models_checkpoints = [ 'unsloth/Llama-3.2-3B', 'Qwen/Qwen2.5-3B' ]
    # models_checkpoints = [ 'unsloth/Llama-3.2-3B' ]

    # No eos only experiments tuning
    models_checkpoints = []

    number_of_eos_tokens = 4

    # model_checkpoint = 'unsloth/Llama-3.2-1B'
    # model_checkpoint = 'HuggingFaceTB/SmolLM2-1.7B'
    # model_checkpoint = 'Qwen/Qwen2.5-1.5B'

    # Train full params
    for model_checkpoint in models_checkpoints:
        model_checkpoint_slug = model_checkpoint.split("/")[-1]
        gradient_accumulation_steps = math.ceil(4096 / NGPUS / per_device_train_batch_size)

        optimized_params = "only_eos_embedding"  # 'full' 'only_eos_embedding'

        run_training_experiments(
            learning_rate=0.0001,
            model_type="sentence_pretrained_checkpoint",
            limit_dataset_shards=4,
            number_of_eos_tokens=number_of_eos_tokens,
            optimized_params=optimized_params,
            weight_decay="0.01",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            adam_beta1="0.9",
            adam_beta2="0.95",
            optim="adamw_torch_fused",
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            max_grad_norm="1.0",
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{NGPUS}gpu",
            model_checkpoint=model_checkpoint,
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd_4IQFRDRG/checkpoint-500/",
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_VYE9JVA0/checkpoint-6500/",
            dataset="smollm-corpus",
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=500,
            dry=dry,
            lr_scheduler_type="cosine",
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=f"sentence_{model_checkpoint_slug}_ft_{optimized_params}_num_eos_tokens_{number_of_eos_tokens}",
        )

    model_checkpoints_eos_tuned = [
        # (f"{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/sentence_Llama-3.2-1B_ft_only_eos_embedding_70ODXUT4/checkpoint-2698", "Llama-3.2-1B", 16, 1),
        # (f"{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/sentence_Qwen2.5-1.5B_ft_only_eos_embedding_25L1K5XT/checkpoint-2698/", "Qwen2.5-1.5B", 4, 1),
        # (f"{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/sentence_Qwen2.5-3B_ft_only_eos_embedding_UAJKCWG0/checkpoint-674/", "Qwen2.5-3B", 4, 1),
        # (f"{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/sentence_Llama-3.2-3B_ft_only_eos_embedding_U4IS2OTK/checkpoint-674/", "Llama-3.2-3B", 4, 1),
        (
            f"{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/sentence_Llama-3.2-1B_ft_only_eos_embedding_num_eos_tokens_4_LIATRTYH/checkpoint-674/",
            "Llama-3.2-1B",
            16,
            4,
        ),
    ]
    # model_checkpoints_eos_tined = []

    NGPUS = 4
    num_train_epochs = 1

    model_checkpoints_eos_tined_full = model_checkpoints_eos_tuned
    # model_checkpoints_eos_tined_full = []

    for (
        model_checkpoint,
        model_checkpoint_slug,
        per_device_train_batch_size,
        number_of_eos_tokens,
    ) in model_checkpoints_eos_tined_full:
        gradient_accumulation_steps = math.ceil(4096 / NGPUS / per_device_train_batch_size)
        optimized_params = "full"  # 'full' 'only_eos_embedding'

        run_training_experiments(
            learning_rate=0.00005,
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
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            max_grad_norm="1.0",
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{NGPUS}gpu",
            model_checkpoint=model_checkpoint,
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd_4IQFRDRG/checkpoint-500/",
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_VYE9JVA0/checkpoint-6500/",
            dataset="smollm-corpus",
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=100,
            dry=dry,
            lr_scheduler_type="cosine",
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=f"sentence_{model_checkpoint_slug}_ft_{optimized_params}_num_eos_tokens_{number_of_eos_tokens}",
        )

    # sys.exit()

    # LoRa training
    NGPUS = 4
    num_train_epochs = 1

    model_checkpoints_eos_tined_lora = model_checkpoints_eos_tuned
    for (
        model_checkpoint,
        model_checkpoint_slug,
        per_device_train_batch_size,
        number_of_eos_tokens,
    ) in model_checkpoints_eos_tined_lora:

        per_device_train_batch_size = 16
        gradient_accumulation_steps = math.ceil(4096 / NGPUS / per_device_train_batch_size)
        optimized_params = "lora"  # 'full' 'only_eos_embedding'

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
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            max_grad_norm="1.0",
            save_total_limit=100,
            save_steps=save_steps,
            instance_type=f"a100.{NGPUS}gpu",
            model_checkpoint=model_checkpoint,
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd_4IQFRDRG/checkpoint-500/",
            # model_checkpoint=f"{workdir_prefix}/sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_VYE9JVA0/checkpoint-6500/",
            dataset="smollm-corpus",
            select_train_dataset_items=0,
            adam_epsilon="1e-8",
            warmup_steps=100,
            dry=dry,
            lr_scheduler_type="cosine",
            bf16="0",
            add_end_of_sentence_token=1,
            experiment_prefix_base_name=f"sentence_{model_checkpoint_slug}_ft_{optimized_params}_num_eos_tokens_{number_of_eos_tokens}",
        )
