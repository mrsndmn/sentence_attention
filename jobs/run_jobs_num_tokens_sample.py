import argparse
from typing import Dict

import client_lib  # импортируем библиотеку для работы с ML Space
import string

import random

# Defaults and constants
REGION = "SR004"
SEED = 1008
INSTANCE_TYPE = "a100.4gpu"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"

# Workspace root used inside jobs
workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"

# Required interpreter path policy
ENV_BIN = "/workspace-SR004.nfs2/d.tarasov/envs/sentence_attention/bin"


def run_experiments(
    experiment: Dict,
    job_description: str = "Num Gist Sample",
    dry: bool = False,
    n_nodes=1,
) -> None:

    n_gpus = 1
    per_device_train_batch_size = 1
    instance_type = f"a100.{n_gpus}gpu"

    grpo_num_generate_samples = experiment.get("grpo_num_generate_samples", 4)
    global_batch_size = experiment["global_batch_size"]
    learning_rate = experiment.get("learning_rate", "0.0001")
    max_steps = experiment.get("max_steps", "-1")
    grpo_loss_alpha = experiment.get("grpo_loss_alpha", "1")

    random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    gradient_accumulation_steps = int(
        global_batch_size // (n_nodes * n_gpus * per_device_train_batch_size * grpo_num_generate_samples)
    )

    all_args = (
        "--save_strategy steps  --save_steps 5000 --max_grad_norm 1.0 --num_train_epochs 1 --seed 1008 --model_type sentence_pretrained_checkpoint "
        '--adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --lr_scheduler_type cosine_with_min_lr   --lr_scheduler_kwargs {\\"min_lr\\":0.00005}  --optimized_params full --warmup_steps 50 '
        "--report_to tensorboard --select_train_dataset_items 0 --weight_decay 0.01 --bf16 0 --torch_compile 0  --eval_strategy no  --eval_steps 0 --gradient_checkpointing 0 --save_total_limit 100 "
        "--optim adamw_torch_fused  --logging_steps 1 --add_end_of_sentence_token 1 --disable_tqdm 0 --dataset fineweb_edu --flexible_eos_tokens 0 "
        "--ft_with_bos_token 0 --save_safetensors 1 --dataloader_num_workers 0  --use_liger_kernel 1 --sentence_attention_implementation sentence_attention_flex "
        " --limit_dataset_shards 1 --offset_dataset_shards 8 "
        f"--max_steps {max_steps} "
        f"--learning_rate {learning_rate} "
        f"--per_device_train_batch_size {per_device_train_batch_size} "
        f"--gradient_accumulation_steps {gradient_accumulation_steps} "
        "--model_checkpoint artifacts/experiments/eos_8/sentence_Llama-3.2-1B_boundaries_prediction_full_boundary_predictor__num_eos_tokens_8_manual_base_weights_KH000000/checkpoint-1000/ "
        f"--output_dir /workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments_in_progress/test_test_sentence_Llama-3.2-1B_ft_full_4k_grpo_{random_suffix} "
        f"--logging_dir /workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments_in_progress/test_test_sentence_Llama-3.2-1B_ft_full_4k_grpo_{random_suffix}/logs "
        "--number_of_eos_tokens 8 --gist_placement boundary_predictor "
        "--boundary_predictor_model_checkpoint artifacts/experiments/bp/unsloth__Llama-3.2-1B_drop_punkt_0.0_bs_4_train_samples_10000/model.pt "
        "--trainer_type grpo "
        f"--grpo_num_generate_samples {grpo_num_generate_samples} "
        f"--grpo_loss_alpha {grpo_loss_alpha} "
    )

    script_str = f"bash {workdir_prefix}/jobs/prepare_multinode_accelerate.sh scripts/train_sentence_llama.py {all_args}"

    print(f"\n\n{script_str}\n\n")

    job_w_args = client_lib.Job(
        base_image=BASE_IMAGE,
        script=script_str,
        type="binary",
        region=REGION,
        instance_type=instance_type,
        n_workers=n_nodes,
        processes_per_worker=1,
        job_desc=f"{job_description} {random_suffix} #rnd #multimodality #tarasov #notify_completed @mrsndmn",
        # stop_timer=600, # в минутах, = 10 часов
        env_variables={
            "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/sentence_attention/bin:/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin",
            "PYTHONPATH": f"{workdir_prefix}/src:{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/src:{workdir_prefix}/../accelerate/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
    )

    if dry:
        print("JOB WAS NOT LAUNCHED")
    else:
        print(job_w_args.submit())

    return


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit boundary prediction training jobs")

    parser.add_argument("--grpo_num_generate_samples", type=int, default=4)
    parser.add_argument("--grpo_loss_alpha", type=float, default=1.0)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default="0.0001")
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--dry", action="store_true")

    return parser


def main() -> None:
    parser = _cli()
    args = parser.parse_args()

    run_experiments(
        {
            "grpo_num_generate_samples": args.grpo_num_generate_samples,
            "grpo_loss_alpha": args.grpo_loss_alpha,
            "global_batch_size": args.global_batch_size,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
        },
        n_nodes=args.n_nodes,
        dry=args.dry,
    )

    return


if __name__ == "__main__":
    main()
