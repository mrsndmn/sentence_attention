import argparse
from typing import Dict

import client_lib  # импортируем библиотеку для работы с ML Space

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
    experiment: Dict, job_description: str = "Boundaries Predictor", test: bool = False, dry: bool = False
) -> None:

    instance_type = "a100.1gpu"

    proba_drop_punkt = experiment.get("proba_drop_punkt", 0.1)
    model_checkpoint = experiment.get("model_checkpoint", "unsloth/Llama-3.2-1B")
    batch_size = experiment.get("batch_size", 4)
    max_train_samples = experiment.get("max_train_samples", 10000)

    all_args = (
        f"--model_checkpoint {model_checkpoint} "
        f"--proba_drop_punkt {proba_drop_punkt} "
        f"--batch_size {batch_size} "
        f"--max_train_samples {max_train_samples} "
        "--compile "
        "--num_workers 0 "
    )

    script_str = f"bash -c 'date && cd {workdir_prefix} && {ENV_BIN}/python scripts/pretrain_boundaries_model.py {all_args} '"

    print(f"\n\n{script_str}\n\n")

    job_w_args = client_lib.Job(
        base_image=BASE_IMAGE,
        script=script_str,
        type="binary",
        region=REGION,
        instance_type=instance_type,
        n_workers=1,
        processes_per_worker=1,
        job_desc=f"{job_description} #rnd #multimodality #tarasov #notify_completed @mrsndmn",
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

    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--proba_drop_punkt", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    # parser.add_argument("--ngpus", type=int, default=None, help="Override GPUs per node (instance type a100.<ngpus>gpu)")
    # parser.add_argument("--num_nodes", type=int, default=None, help="Override number of nodes/workers")

    return parser


def main() -> None:
    parser = _cli()
    args = parser.parse_args()

    run_experiments(
        {
            "model_checkpoint": args.model_checkpoint,
            "proba_drop_punkt": args.proba_drop_punkt,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples,
        }
    )

    return


if __name__ == "__main__":
    main()
