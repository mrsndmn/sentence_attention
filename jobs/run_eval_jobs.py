import copy
import glob
import json
import os
import time

import client_lib  # импортируем библиотеку для работы с ML Space
from mls.manager.job.utils import training_job_api_from_profile

from sentence_attention.artifacts.experiments import sort_checkpoints
from sentence_attention.evaluation.benchmarks import (
    all_benchmarks,
    checkpoint_evaluation_file,
    long_benchmarks,
    short_benchmarks,
)
from sentence_attention.integration.job import REGION, get_in_progress_jobs

SEED = 1008

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"
# BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

JOB_DESCRIPTION_SUFFIX = "#rnd #multimodality #tarasov @mrsndmn"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"

experiments_dir = os.path.join(workdir_prefix, "artifacts", "experiments")


def run_helmet_eval_experiments(experiment, job_description="Eval", dry=False, local=False):

    experiment = copy.deepcopy(experiment)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    pretrained_model = experiment.pop("pretrained_model")
    benchmark = experiment.pop("benchmark")

    if len(experiment.keys()) > 0:
        raise ValueError("Invalid exp values!")

    helmet_workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/HELMET"

    output_dir = os.path.join(pretrained_model, "helmet_eval", benchmark)
    os.makedirs(output_dir, exist_ok=True)

    assert benchmark in long_benchmarks, f"Invalid benchmark: {benchmark}"

    script_str = f"bash -c 'date && cd {helmet_workdir_prefix} && {env_bin_path}/python eval.py --config configs/{benchmark}_tiny.yaml --model_name_or_path {pretrained_model} --use_chat_template False --no_torch_compile --output_dir {output_dir} '"

    print(f"\n\n{script_str}\n")

    job_w_args = client_lib.Job(
        base_image=BASE_IMAGE,
        script=script_str,
        type="binary",  # =='binary' allows to run bash scripts
        region=REGION,
        instance_type=INSTANCE_TYPE,
        n_workers=N_WORKERS,
        # conda_env="test_client_lib",
        processes_per_worker=1,
        job_desc=job_description,
        # stop_timer=600, # в минутах, = 10 часов
        env_variables={
            "PATH": f"{env_bin_path}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
            "PYTHONPATH": f"{helmet_workdir_prefix}/src:{helmet_workdir_prefix}/../sentence_attention/src:{helmet_workdir_prefix}/../transformers_adaptive_fan_in_fan_out/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
    )

    if dry:
        print("JOB WAS NOT LAUNCHED")
    else:
        if local:
            print("Running local process")
            import subprocess

            # Example: Running a simple command
            result = subprocess.run(script_str, shell=True)
            if result.returncode != 0:
                print("Failed to run job:", job_description)

        else:
            print(job_description, "\n", job_w_args.submit())

    return


def run_lighteval_eval_experiments(experiment, job_description="Eval", dry=False, local=False):

    experiment = copy.deepcopy(experiment)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    pretrained_model = experiment.pop("pretrained_model")
    benchmark = experiment.pop("benchmark")

    if len(experiment.keys()) > 0:
        raise ValueError("Invalid exp values!")

    script_str = f"bash -c 'date && cd {workdir_prefix} && {env_bin_path}/python scripts/evaluate.py --checkpoint {pretrained_model} --benchmark {benchmark}'"

    print(f"\n\n{script_str}\n")

    job_w_args = client_lib.Job(
        base_image=BASE_IMAGE,
        script=script_str,
        type="binary",  # =='binary' allows to run bash scripts
        region=REGION,
        instance_type=INSTANCE_TYPE,
        n_workers=N_WORKERS,
        # conda_env="test_client_lib",
        processes_per_worker=1,
        job_desc=job_description,
        # stop_timer=600, # в минутах, = 10 часов
        env_variables={
            "PATH": f"{env_bin_path}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
            "PYTHONPATH": f"{workdir_prefix}/src:{workdir_prefix}/../transformers_adaptive_fan_in_fan_out/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
    )

    if dry:
        print("JOB WAS NOT LAUNCHED")
    else:
        if local:
            print("Running local process")
            import subprocess

            # Example: Running a simple command
            result = subprocess.run(script_str, shell=True)
            if result.returncode != 0:
                print("Failed to run job:", job_description)

        else:
            print(job_description, "\n", job_w_args.submit())

    return


def run_eval_experiments(experiment, job_description="Eval", dry=False, local=False):

    if experiment["benchmark"] in ["recall", "rag", "rerank", "cite", "longqa", "summ", "icl"]:
        return run_helmet_eval_experiments(experiment, job_description, dry, local)

    return run_lighteval_eval_experiments(experiment, job_description, dry, local)


def run_extract_metrics(checkpoints: list[str], tasks=None):

    if tasks is None:
        tasks = [
            "custom|arc:_average|0",
            "custom|piqa|0",
            "custom|mmlu_cloze:_average|0",
            "custom|mmlu_pro_cloze|0",
            "custom|wikitext_103|0",
        ]
    else:
        tasks_mapping = {
            "custom|arc|0|1": "custom|arc:_average|0",
            "custom|openbookqa|0|1": "custom|openbookqa|0",
            "custom|mmlu_cloze|0|1": "custom|mmlu_cloze:_average|0",
            "custom|mmlu_cloze|5|1": "custom|mmlu_cloze:_average|5",
            "custom|mmlu_pro_cloze|0|1": "custom|mmlu_pro_cloze|0",
            "custom|wikitext_103|0|1": "custom|wikitext_103|0",
            "custom|siqa|0|1": "custom|siqa|0",
            "custom|piqa|0|1": "custom|piqa|0",
            "custom|hellaswag|0|1": "custom|hellaswag|0",
            "custom|winogrande|0|1": "custom|winogrande|0",
            "custom|tiny_stories|0|1": "custom|tiny_stories|0",
            "custom|tiny_stories_with_end_of_sentence|0|1": "custom|tiny_stories_with_end_of_sentence|0",
            "custom|gsm8k|4|1": "custom|gsm8k|4",
        }
        tasks = list(map(lambda x: tasks_mapping[x], tasks))

    print(" & ".join(["checkpoint"] + tasks), " \\\\")

    for checkpoint in checkpoints:

        if "unsloth/" in checkpoint or "Qwen/" in checkpoint or "HuggingFaceTB/" in checkpoint:
            # exps_evaluation/results/unsloth/Meta-Llama-3.1-8B/
            checkpoint_norm = checkpoint.split("/")
        else:
            checkpoint_norm = [checkpoint.replace("/", "_")]
        metrics_mask = os.path.join("exps_evaluation", "results", *checkpoint_norm, "*.json")
        metrics_paths = glob.glob(metrics_mask)
        # print("metrics_paths", metrics_paths)
        assert len(metrics_paths) >= 1, f"No metrics files found for {checkpoint}"
        # Sort by creation time and take the most recent one
        metrics_paths = sorted(metrics_paths, key=lambda x: os.path.getctime(x))[-9:]
        # print("")
        # print("checkpoint", checkpoint)
        # print("metrics_paths", metrics_paths)

        checkpoint_metrics = []

        metrics_dict = {}

        for metrics_path in metrics_paths:

            with open(metrics_path) as f:
                json_data = json.load(f)

            for key in tasks:

                metric_dict = json_data["results"].get(key, {})

                metric = 0

                if "acc_norm" in metric_dict:
                    metric = metric_dict["acc_norm"]
                elif "qem" in metric_dict:
                    metric = metric_dict["qem"]
                elif "ppl" in metric_dict:
                    metric = metric_dict["ppl"] / 100
                elif len(metric_dict.keys()) > 0:
                    raise ValueError("unknown metrics:", metric_dict)

                if metric != 0:
                    metrics_dict[key] = f"{metric*100:.2f}"

        for key in tasks:
            checkpoint_metrics.append(metrics_dict.get(key, ""))

        print(
            " & ".join(
                [
                    checkpoint,
                ]
                + checkpoint_metrics
            ),
            " \\\\",
        )


def get_in_progress_jobs_descriptions(client):

    in_progress_jobs_descriptions = set()

    for job in get_in_progress_jobs(client):
        in_progress_jobs_descriptions.add(job["job_desc"])

    return in_progress_jobs_descriptions


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_checkpoints", type=int, default=1)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--benchmark", type=str, default="all")
    parser.add_argument(
        "--eos_num", type=str, default="all", choices=["all", "eos_0", "eos_1", "eos_2", "eos_4", "eos_8", "eos_16"]
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit_jobs", type=int, default=None)
    parser.add_argument("--max_jobs_queue_size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    num_checkpoints = args.num_checkpoints

    client, _ = training_job_api_from_profile("default")

    in_progress_jobs_descriptions = get_in_progress_jobs_descriptions(client)

    if args.limit_jobs is not None:
        assert args.limit_jobs > 0

    if args.benchmark == "all":
        benchmarks = copy.deepcopy(all_benchmarks)
    elif args.benchmark == "long":
        benchmarks = copy.deepcopy(long_benchmarks)
    elif args.benchmark == "short":
        benchmarks = copy.deepcopy(short_benchmarks)
    else:
        benchmarks = args.benchmark.split(",")

    for benchmark in benchmarks:
        assert (
            benchmark in all_benchmarks + long_benchmarks
        ), f"Benchmark {benchmark} not in {all_benchmarks + long_benchmarks + short_benchmarks}"

    print(f"Evaluating {num_checkpoints} checkpoints for {benchmarks} benchmarks")

    check_queue_processed_models = -1
    processed_models = 0

    stop = False

    for eos_num in os.listdir(experiments_dir):
        if stop:
            break

        if args.eos_num != "all":
            if eos_num != args.eos_num:
                continue

        experiments_dirs = os.listdir(os.path.join(experiments_dir, eos_num))

        for benchmark in benchmarks:
            if benchmark == "gsm8k":
                continue

            for experiment_dir in experiments_dirs:
                if stop:
                    break

                if args.model is not None and args.model.lower() not in experiment_dir.lower():
                    print(f"Skipping {experiment_dir} because it does not contain {args.model}")
                    continue

                # if experiment_dir not in ['sentence_Llama-3.2-3B_ft_bos_token_full_num_eos_tokens_4_8H2VTT04', 'sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR']:
                #     continue

                experiment_eval_dir = os.listdir(os.path.join(experiments_dir, eos_num, experiment_dir))
                checkpoints = sort_checkpoints(experiment_eval_dir)[:num_checkpoints]

                for checkpoint in checkpoints:

                    if args.max_jobs_queue_size is not None and check_queue_processed_models < processed_models:
                        while True:
                            in_queue_jobs = get_in_progress_jobs(client, statuses=["Pending"])
                            queue_size = len(in_queue_jobs)

                            if queue_size >= args.max_jobs_queue_size:
                                sleep_time = 10
                                print(
                                    f"Max jobs queue size {queue_size} / {args.max_jobs_queue_size} reached, waiting for {sleep_time} seconds"
                                )
                                time.sleep(sleep_time)
                            else:
                                # update in_progress_jobs_descriptions
                                in_progress_jobs_descriptions = get_in_progress_jobs_descriptions(client)
                                break

                            check_queue_processed_models = processed_models

                    full_experiment_dir = os.path.join(experiments_dir, eos_num, experiment_dir, checkpoint)

                    evaluation_file = checkpoint_evaluation_file(full_experiment_dir, benchmark)

                    if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size > 0:
                        if args.force:
                            if not args.dry:
                                os.remove(evaluation_file)
                            print(f"Force remove metrics {evaluation_file}")

                        else:
                            print(f"Evaluation file {evaluation_file} already exists")
                            continue

                    experiment = {
                        "pretrained_model": full_experiment_dir,
                        "benchmark": benchmark,
                    }

                    job_description = f"Eval {benchmark}: {eos_num}/{experiment_dir}/{checkpoint} {JOB_DESCRIPTION_SUFFIX}"

                    if job_description in in_progress_jobs_descriptions:
                        print(f"Job {job_description} already in progress, skipping")
                        continue

                    run_eval_experiments(
                        experiment,
                        dry=args.dry,
                        job_description=job_description,
                        local=args.local,
                    )

                    processed_models += 1
                    if not args.dry:
                        pass
                        # time.sleep(5)  # to avoid race conditions and brusting max queue size

                    if args.limit_jobs is not None and processed_models >= args.limit_jobs:
                        print(f"Processed {processed_models} models, stopping")
                        stop = True
                        break

    print(f"Runned {processed_models} jobs")
