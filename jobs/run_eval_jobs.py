import glob
import client_lib # импортируем библиотеку для работы с ML Space
import json
import copy
from rich.console import Console

import os

REGION = "SR004"

SEED = 1008

console = Console()
console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_eval_experiments(experiments, job_description_prefix="eval", dry=False, tasks=None):

    experiments = copy.deepcopy(experiments)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        pretrained_model = exp.pop('pretrained_model')
        output_dir = exp.pop('output_dir', './exps_evaluation')

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values!")

        default_tasks = "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|wikitext_103|0|1,custom|winogrande|0|1,custom|hellaswag|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|piqa|0|1"

        if tasks is None:
            tasks = default_tasks

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "{tasks}"\''
        # script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 4 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1,custom|wikitext_103|0|1"\''

        # TODO gsm8k, math - 8--shot

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} {pretrained_model} {tasks} #rnd #multimodality @mrsndmn",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": f"{env_bin_path}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
                "PYTHONPATH": f"{workdir_prefix}/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(output_dir, job_w_args.submit())

    return

def run_extract_metrics(checkpoints: list[str], tasks=None):

    if tasks is None:
        tasks = [
            'custom|arc:_average|0',
            'custom|piqa|0',
            'custom|mmlu_cloze:_average|0',
            'custom|mmlu_pro_cloze|0',
            'custom|wikitext_103|0',
        ]
    else:
        tasks_mapping = {
            'custom|arc|0|1': 'custom|arc:_average|0',
            'custom|openbookqa|0|1': 'custom|openbookqa|0',
            'custom|mmlu_cloze|0|1': 'custom|mmlu_cloze:_average|0',
            'custom|mmlu_cloze|5|1': 'custom|mmlu_cloze:_average|5',
            'custom|mmlu_pro_cloze|0|1': 'custom|mmlu_pro_cloze|0',
            'custom|wikitext_103|0|1': 'custom|wikitext_103|0',
            'custom|siqa|0|1': 'custom|siqa|0',
            'custom|piqa|0|1': 'custom|piqa|0',
            'custom|hellaswag|0|1': 'custom|hellaswag|0',
            'custom|winogrande|0|1': 'custom|winogrande|0',
            'custom|tiny_stories|0|1': 'custom|tiny_stories|0',
            'custom|tiny_stories_with_end_of_sentence|0|1': 'custom|tiny_stories_with_end_of_sentence|0',
            'custom|gsm8k|4|1': 'custom|gsm8k|4',
        }
        tasks = list(map(lambda x: tasks_mapping[x], tasks))

    print(" & ".join([ 'checkpoint' ] + tasks), " \\\\")

    for checkpoint in checkpoints:

        if 'unsloth/' in checkpoint or 'Qwen/' in checkpoint or 'HuggingFaceTB/' in checkpoint:
            # exps_evaluation/results/unsloth/Meta-Llama-3.1-8B/
            checkpoint_norm = checkpoint.split('/')
        else:
            checkpoint_norm = [ checkpoint.replace('/', '_') ]
        metrics_mask = os.path.join('exps_evaluation', 'results', *checkpoint_norm, '*.json')
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

            with open(metrics_path, "r") as f:
                json_data = json.load(f)

            for key in tasks:

                metric_dict = json_data['results'].get(key, {})

                metric = 0

                if 'acc_norm' in metric_dict:
                    metric = metric_dict['acc_norm']
                elif 'qem' in metric_dict:
                    metric = metric_dict['qem']
                elif 'ppl' in metric_dict:
                    metric = metric_dict['ppl'] / 100
                elif len(metric_dict.keys()) > 0:
                    raise ValueError("unknown metrics:", metric_dict)

                if metric != 0:
                    metrics_dict[key] = f"{metric*100:.2f}"

        for key in tasks:
            checkpoint_metrics.append(metrics_dict.get(key, ""))

        print(" & ".join([ checkpoint, ] + checkpoint_metrics), " \\\\")



def eval_hcg_adaptive_pretrain_all_tasks_parallel(**kwargs):

    checkpoints = [
        # a lot of jobs ...
        # /mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation/results/unsloth/Meta-Llama-3.1-8B/
        # results_2025-06-16T17-17-36.190191.json
        # results_2025-06-16T17-20-07.497466.json
        # results_2025-06-16T17-20-20.107840.json
        # results_2025-06-16T17-20-33.343409.json
        # results_2025-06-16T17-20-49.794687.json
        # results_2025-06-16T17-21-40.487502.json
        # results_2025-06-16T17-21-50.828092.json
        # results_2025-06-16T17-26-02.538177.json
        # results_2025-06-16T17-43-04.025151.json
        #
        # "unsloth/Meta-Llama-3.1-8B",
        "Qwen/Qwen2.5-7B",

        # "./adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/checkpoint-5000",
        # "./adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/calibr60_checkpoint-5000_left_padding/",
        # "./adaptive_hcg_llama31_8B_l22-26_analytical_pruning",

        # "./adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/checkpoint-5000",
        # "./adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/calib40_checkpoint-5000",
        # "./adaptive_hcg_qwen25_7B_l12-17_analytical_pruning",

        # TODO fan out projection + llm layers finetuned
        # TODO fan out projection checkpoint

        # Analytical Pruning
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.25/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.5/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.75/",
        "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q1/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.25/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.5/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.75/",
        "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q1/",
    ]

    tasks = "custom|arc|0|1,custom|openbookqa|0|1,custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|wikitext_103|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|winogrande|0|1".split(",")

    checkpoints = [
        "./paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib40",
        "./paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib40",

        './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib90',
        './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib90',
    ]

    # checkpoints for fan out projection tuned models
    checkpoints = [
        './adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_GHLL4UA3/checkpoint-60000/',
        './adaptive_hcg_qwen25_7B_fan_out_projection_w_1.000_l_9-20_8VWXTSOA/checkpoint-60000/',
    ]

    # temporary checkpoints for finetuned models
    # checkpoints = [
    #     './paper_checkpoints/ft_thrshold0.6/adaptive_hcg_llama31_8B_finetune_w_1.000_l_18-26_GSGM5UVW/checkpoint-12498/',
    #     './paper_checkpoints/ft_thrshold0.6/adaptive_hcg_qwen25_7B_finetune_w_1.000_l_9-20_ZGRS21P4/checkpoint-12498/',
    # ]

    # Lora and Full FineTune
    # checkpoints = [
    #     "./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL/checkpoint-12420/",
    # ]
    # tasks = "custom|arc|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|tiny_stories_with_end_of_sentence|0|1".split(",")

    # checkpoints = [
    #     # Small LLM
    #     "./adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0/checkpoint-12420/",
    #     "./vanilla_slm2_135M_pretrain_w_0.000_l_-_DFAC2NSB/checkpoint-12420",
    # ]
    # tasks = "custom|arc|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|tiny_stories|0|1".split(",")

    # checkpoints = [
    #     "./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL/checkpoint-12420/",
    #     "./adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0/checkpoint-12420/",
    #     "./vanilla_slm2_135M_pretrain_w_0.000_l_-_DFAC2NSB/checkpoint-12420",
    # ]
    # tasks = "custom|arc|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|tiny_stories|0|1".split(",")


    # Large LLM
    # checkpoints = [
    #     "./adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_R2DI580B/checkpoint-7500/",
    #     "./adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_R2DI580B/checkpoint-10999/",
    #     # "./adaptive_hcg_llama31_8B_lora_finetune_w_1.000_l_18-26_4CL0OO5V/checkpoint-2500/",
    #     "./adaptive_hcg_llama31_8B_finetune_w_1.000_l_18-26_8TOJUP14/checkpoint-7500/"
    # ]

    # Checkpoints with hcg in eval mode during training
    checkpoints = [
        # "./adaptive_hcg_llama31_8B_fan_out_projection_trimmed_embeddings_w_1.000_l_18-26_653SHN4I/checkpoint-2750/",
        # "./paper_checkpoints/foproj_thrshold0.6/adaptive_hcg_llama31_8B_fan_out_projection_trimmed_embeddings_w_1.000_l_18-26_653SHN4I/checkpoint-5500/"
        # "./paper_checkpoints/foproj_thrshold0.6/adaptive_hcg_llama31_8B_fan_out_projection_trimmed_embeddings_w_1.000_l_18-26_653SHN4I/checkpoint-10999/",
        # "unsloth/Meta-Llama-3.1-8B",
        "./adaptive_hcg_llama31_8B_fan_out_projection_trimmed_embeddings_w_1.000_l_18-26_WRT210U7/checkpoint-10999",
    ]

    # SLM small batch
    checkpoints = [
        './adaptive_slm2_135M_pretrain_w_0.100_l_10-20_951OTTFF/checkpoint-66241/',
        './adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_CN17H8GB/checkpoint-66241_eos_tokenizer/',
        # "./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_8WUWMNL1/checkpoint-66241/",
        "./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_10-20_YLMXOZZP/checkpoint-66241/",
        './vanilla_slm2_135M_pretrain_w_0.000_l_-_IWQ3X999/checkpoint-66241/',
        "./vanilla_slm2_127M_20L_pretrain_w_0.000_l_-_STYB64DM/checkpoint-60000/",

        # With Large Batch Size
        # "./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_O8T0D1OA/checkpoint-4140/",
    ]
    tasks = "custom|arc|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|tiny_stories|0|1".split(",")


    checkpoints = [
        "./vanilla_slm2_1.7B_pretrain_w_0.000_l_-_BM2DG7DH/checkpoint-19000_for_4shot_mmlu/",
        "./vanilla_slm2_1.7B_pretrain_16L_w_0.000_l_-_X3NTEOHS/checkpoint-19000_for_4shot_mmlu/",
        "./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_K6WC5X8F/checkpoint-19000_for_4shot_mmlu/",
    ]
    tasks = "custom|mmlu_cloze|4|1".split(",")

    # TODO

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    print('len hcg_experiments:', len(hcg_experiments))

    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints, tasks=tasks)
    else:
        for task in tasks:
            run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", tasks=task, **kwargs)


    return


def slm2_mmlu_cloze_5_shot(**kwargs):

    checkpoints = [
        "HuggingFaceTB/SmolLM2-1.7B",
        "sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_IOVO1EQ9/checkpoint-26000/",
    ]

    tasks = "custom|mmlu_cloze|5|1".split(",")

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    print('len hcg_experiments:', len(hcg_experiments))


    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints, tasks=tasks)
    else:
        for task in tasks:
            run_eval_experiments(hcg_experiments, job_description_prefix="Eval: ", tasks=task, **kwargs)

    return



def eval_base_models(**kwargs):

    checkpoints = [

        # EOS Only Embedding
        # "./sentence_Llama-3.2-1B_ft_only_eos_embedding_3DTAMXFX/checkpoint-674/",
        # "./sentence_Llama-3.2-1B_ft_only_eos_embedding_70ODXUT4/checkpoint-2698",

        # 'unsloth/Llama-3.2-1B',
        # "./sentence_Llama-3.2-1B_pretrain_with_end_of_sentence_full_BTLCR6IG/checkpoint-2698/",


        # 'Qwen/Qwen2.5-1.5B',
        # "./sentence_Qwen2.5-1.5B_pretrain_with_end_of_sentence_full_271TTUXM/checkpoint-2698/",

        # 'HuggingFaceTB/SmolLM2-1.7B',
        # "./sentence_SmolLM2-1.7B_pretrain_with_end_of_sentence_full_2V0V8WU1/checkpoint-2698/",

        # # 3B models
        # 'unsloth/Llama-3.2-3B',
        # "./sentence_Llama-3.2-3B_pretrain_with_end_of_sentence_full_KVSAH64V/checkpoint-2698/",

        # # 'Qwen/Qwen2.5-3B',
    ]

    # ("hellaswag", evaluate_acc_hellaswag),
    # ("winogrande", evaluate_acc_winogrande),
    # ("piqa", evaluate_acc_piqa),
    # ("siqa", evaluate_acc_siqa),
    # ("openbookqa", evaluate_acc_openbookqa),
    # ("mmlu_0_shot", evaluate_acc_mmlu_0_shot),

    # tasks = ["custom|mmlu_cloze|5|1"]
    # tasks = "custom|mmlu_cloze|5|1,custom|mmlu_cloze|0|1,custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1".split(",")
    tasks = "custom|mmlu_cloze|0|1,custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1".split(",")

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    print('len hcg_experiments:', len(hcg_experiments))


    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints, tasks=tasks)
    else:
        for task in tasks:
            run_eval_experiments(hcg_experiments, job_description_prefix="Eval: ", tasks=task, **kwargs)

    return



if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    extract_metrics = len(sys.argv) > 1 and sys.argv[1] == 'extract_metrics'

    print("dry", dry, 'extract_metrics', extract_metrics)

    # eval_hcg_adaptive_pretrain_all_tasks_parallel(dry=dry, extract_metrics=extract_metrics)
    # slm2_mmlu_cloze_5_shot(dry=dry, extract_metrics=extract_metrics)
    eval_base_models(dry=dry, extract_metrics=extract_metrics)
