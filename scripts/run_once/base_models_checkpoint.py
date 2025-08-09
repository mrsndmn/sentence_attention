import os

from transformers import AutoModelForCausalLM, AutoTokenizer

target_path = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments/eos_0/"

if __name__ == "__main__":

    models_names = [
        "unsloth/Llama-3.2-1B",
        "unsloth/Llama-3.2-3B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-1.5B",
    ]

    for model_name in models_names:

        model_slug = model_name.split("/")[-1]

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        target_path_for_model = os.path.join(target_path, model_slug, "checkpoint-0")
        print(f"Saving {model_name} to {target_path_for_model}")

        model.save_pretrained(target_path_for_model)
        tokenizer.save_pretrained(target_path_for_model)
