import os
import sys

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    dry = len(sys.argv) > 1 and sys.argv[1] == "dry"

    client, extra_options = training_job_api_from_profile("default")

    workdir = os.getcwd()

    author_name = "d.tarasov"

    # for num_eos_tokens in [ 1, 4, 8, 16 ]:
    for num_eos_tokens in [8, 16]:

        for pretrained_model_name in ["unsloth/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B"]:

            script = f"bash -c 'cd {workdir} && /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python scripts/tokenize_fineweb_edu.py --pretrained_model_name {pretrained_model_name} --with_eos_token --num_eos_tokens {num_eos_tokens}'"
            print("\n\n", script)
            if dry:
                print("Skip running job")
                continue

            result = client.run_job(
                payload={
                    "script": script,
                    "job_desc": f"Tokenize fineweb EOS model={pretrained_model_name} num_eos_tokens={num_eos_tokens} #{author_name} #rnd #multimodal @mrsndmn",
                    "instance_type": "a100.1gpu",
                    "region": extra_options["region"],
                    "env_variables": {
                        "PYTHONPATH": "./src",
                        "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
                    },
                    "type": "binary_exp",
                    "shm_size_class": "medium",
                    "base_image": "cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36",
                    "n_workers": 1,  # Количество воркеров.
                    "processes_per_worker": 1,  # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
                }
            )

            print(pretrained_model_name, "eos", num_eos_tokens, pretrained_model_name, ":\t", result)
