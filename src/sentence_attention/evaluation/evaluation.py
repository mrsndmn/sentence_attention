import json
import os

import torch
from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder, EvaluationTracker
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from sentence_attention.artifacts.experiments import WORKDIR_PREFIX
from sentence_attention.evaluation.benchmarks import checkpoint_evaluation_file

workdir_prefix = WORKDIR_PREFIX

task_to_default_batch_size = {
    "arc": 64,
    "hellaswag": 64,
    "mmlu_cloze": 16,
    "mmlu_pro_cloze": 16,
    "piqa": 64,
    "siqa": 256,
    "openbookqa": 256,
    "winogrande": 512,
}


def evaluate_lighteval_task_save_results(
    model, model_checkpoint, task_name, override_batch_size=None, num_fewshot_seeds=None, max_samples=None
):

    results = evaluate_lighteval_task(model, task_name, override_batch_size, num_fewshot_seeds, max_samples=max_samples)

    results_file = checkpoint_evaluation_file(model_checkpoint, task_name)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, cls=EnhancedJSONEncoder)

    print(f"Results saved to {results_file}")

    return results


def build_evaluation_pipeline(model, task_name, override_batch_size=None, num_fewshot_seeds=None, max_samples=None):

    evaluation_output_dir = os.path.join(workdir_prefix, "artifacts", "evaluation")
    os.makedirs(evaluation_output_dir, exist_ok=True)

    evaluation_tracker = EvaluationTracker(output_dir=evaluation_output_dir, save_details=True)

    if num_fewshot_seeds is None:
        num_fewshot_seeds = 0

    if override_batch_size is None:
        override_batch_size = task_to_default_batch_size.get(task_name, 1)

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
        max_samples=max_samples,
    )

    tasks = f"custom|{task_name}|{num_fewshot_seeds}|1"

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
    )

    return pipeline


def evaluate_lighteval_task(model, task_name, override_batch_size=None, num_fewshot_seeds=None, max_samples=None):

    with torch.no_grad():
        pipeline = build_evaluation_pipeline(model, task_name, override_batch_size, num_fewshot_seeds, max_samples=max_samples)
        pipeline.evaluate()

        pipeline.show_results()
        results = pipeline.get_results()

    return results
