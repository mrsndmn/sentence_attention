import copy
import glob
import json
import os

import client_lib  # импортируем библиотеку для работы с ML Space
from mls.manager.job.utils import training_job_api_from_profile
from rich.console import Console
from sentence_attention.artifacts.experiments import sort_checkpoints
from sentence_attention.evaluation.benchmarks import all_benchmarks, checkpoint_evaluation_file

REGION = "SR004"

SEED = 1008

console = Console()
console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36"
# BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

JOB_DESCRIPTION_SUFFIX = "#rnd #multimodality @mrsndmn"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"

experiments_dir = os.path.join(workdir_prefix, "artifacts", "experiments")


def run_eval_experiments(experiment, job_description="Eval", dry=False):

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
        print(job_description, "\n", job_w_args.submit())

    return


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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_checkpoints", type=int, default=1)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", *all_benchmarks])
    parser.add_argument("--eos_num", type=str, default="all", choices=["all", "eos_0", "eos_1", "eos_4"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit_jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    num_checkpoints = args.num_checkpoints

    client, _ = training_job_api_from_profile("default")

    in_progress_jobs_descriptions = set()

    for non_final_status in ["Completing", "Running", "Pending"]:
        non_final_jobs = client.get_list_jobs(
            region=REGION, allocation_name="alloc-officecds-multimodal-2-sr004", status=non_final_status, limit=1000, offset=0
        )
        for job in non_final_jobs["jobs"]:
            in_progress_jobs_descriptions.add(job["job_desc"])

    print("In progress jobs descriptions", len(in_progress_jobs_descriptions), in_progress_jobs_descriptions)

    if args.limit_jobs is not None:
        assert args.limit_jobs > 0

    benchmarks = all_benchmarks if args.benchmark == "all" else [args.benchmark]

    print(f"Evaluating {num_checkpoints} checkpoints for {benchmarks} benchmarks")

    processed_models = 0

    stop = False

    for eos_num in os.listdir(experiments_dir):
        if stop:
            break

        if args.eos_num != "all":
            if eos_num != args.eos_num:
                continue

        for experiment_dir in os.listdir(os.path.join(experiments_dir, eos_num)):
            if stop:
                break

            if args.model is not None and args.model not in experiment_dir.lower():
                # print(f"Skipping {experiment_dir} because it does not contain {args.model}")
                continue

            experiment_eval_dir = os.listdir(os.path.join(experiments_dir, eos_num, experiment_dir))
            checkpoints = sort_checkpoints(experiment_eval_dir)[:num_checkpoints]

            for checkpoint in checkpoints:
                for benchmark in benchmarks:
                    full_experiment_dir = os.path.join(experiments_dir, eos_num, experiment_dir, checkpoint)

                    evaluation_file = checkpoint_evaluation_file(full_experiment_dir, benchmark)

                    if os.path.exists(evaluation_file):
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

                    job_description = f"Eval {eos_num}/{experiment_dir}/{checkpoint} {benchmark} {JOB_DESCRIPTION_SUFFIX}"

                    if job_description in in_progress_jobs_descriptions:
                        print(f"Job {job_description} already in progress, skipping")
                        continue

                    run_eval_experiments(
                        experiment,
                        dry=args.dry,
                        job_description=job_description,
                    )

                    processed_models += 1

                    if args.limit_jobs is not None and processed_models >= args.limit_jobs:
                        print(f"Processed {processed_models} models, stopping")
                        stop = True
                        break

    print(f"Runned {processed_models} jobs")
