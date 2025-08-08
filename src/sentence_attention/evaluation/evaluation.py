import json
import os

import torch
from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder, EvaluationTracker
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from sentence_attention.evaluation.benchmarks import checkpoint_evaluation_file

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"

task_to_default_batch_size = {
    "arc": 128,
    "hellaswag": 64,
    "mmlu_cloze": 16,
    "mmlu_pro_cloze": 16,
    "piqa": 128,
    "siqa": 512,
    "openbookqa": 256,
    "winogrande": 512,
}


def evaluate_lighteval_task_save_results(model, model_checkpoint, task_name, override_batch_size=None, num_fewshot_seeds=None):

    results = evaluate_lighteval_task(model, task_name, override_batch_size, num_fewshot_seeds)

    results_file = checkpoint_evaluation_file(model_checkpoint, task_name)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, cls=EnhancedJSONEncoder)

    print(f"Results saved to {results_file}")

    return results


def evaluate_lighteval_task(model, task_name, override_batch_size=None, num_fewshot_seeds=None):
    evaluation_output_dir = os.path.join(workdir_prefix, "artifacts", "evaluation")
    os.makedirs(evaluation_output_dir, exist_ok=True)

    evaluation_tracker = EvaluationTracker(output_dir=evaluation_output_dir, save_details=True)

    if num_fewshot_seeds is None:
        num_fewshot_seeds = 0

    if override_batch_size is None:
        override_batch_size = task_to_default_batch_size.get(task_name, 8)

    print(f"Evaluating {task_name} with override_batch_size={override_batch_size} and num_fewshot_seeds={num_fewshot_seeds}")

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(
            cache_dir="/workspace-SR004.nfs2/.cache/huggingface",
        ),
        custom_tasks_directory=os.path.join(workdir_prefix, "src", "sentence_attention", "evaluation", "lighteval_tasks.py"),
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        use_chat_template=False,
        system_prompt=None,
        load_responses_from_details_date_id=None,
    )

    tasks = f"custom|{task_name}|{num_fewshot_seeds}|1"

    with torch.no_grad():
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=model,
        )
        pipeline.evaluate()

        pipeline.show_results()
        results = pipeline.get_results()

    return results


def evaluate_ppl_wikitext_103(model):

    results = evaluate_lighteval_task(
        model,
        "wikitext_103",
        override_batch_size=2,
        num_fewshot_seeds=0,
    )

    print('results[results]["custom:wikitext_103:0"]', results["results"]["custom:wikitext_103:0"])

    return results


def evaluate_acc_hellaswag(model):

    results = evaluate_lighteval_task(
        model,
        "hellaswag",
        override_batch_size=32,
        num_fewshot_seeds=0,
    )

    return results


def evaluate_acc_mmlu_0_shot(model):

    results = evaluate_lighteval_task(
        model,
        "mmlu_cloze",
        override_batch_size=16,
        num_fewshot_seeds=0,
    )

    return results


def evaluate_acc_mmlu_5_shot(model):

    results = evaluate_lighteval_task(
        model,
        "mmlu_cloze",
        override_batch_size=16,
        num_fewshot_seeds=5,
    )

    return results


def evaluate_acc_winogrande(model):
    results = evaluate_lighteval_task(
        model,
        "winogrande",
        override_batch_size=512,
        num_fewshot_seeds=0,
    )

    return results


def evaluate_acc_piqa(model):
    results = evaluate_lighteval_task(
        model,
        "piqa",
        override_batch_size=128,
        num_fewshot_seeds=0,
    )

    return results


def evaluate_acc_siqa(model):
    results = evaluate_lighteval_task(
        model,
        "siqa",
        override_batch_size=512,
        num_fewshot_seeds=0,
    )

    return results


def evaluate_acc_openbookqa(model):
    results = evaluate_lighteval_task(
        model,
        "openbookqa",
        override_batch_size=256,
        num_fewshot_seeds=0,
    )

    return results
