import argparse
import os

import torch
from accelerate.utils import merge_fsdp_weights
from safetensors.torch import safe_open
from transformers import AutoTokenizer

from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_directory", type=str, help="The checkpoint directory to merge")
    parser.add_argument(
        "--initial_model_checkpoint", type=str, help="EOSO checkpoint that was used as initialisation for full pretraining"
    )

    args = parser.parse_args()
    checkpoint_directory = args.checkpoint_directory
    initial_model_checkpoint = args.initial_model_checkpoint

    checkpoint_directory = checkpoint_directory.removesuffix("/")

    assert checkpoint_directory.endswith("/pytorch_model_fsdp_0"), "Checkpoint directory must end with /pytorch_model_fsdp_0"

    # out dir is renamed from pytorch_model_fsdp_0 to pytorch_model_merged
    # so we need to rename the out dir to pytorch_model_fsdp_0
    out_dir = checkpoint_directory.replace("pytorch_model_fsdp_0", "hf_model_merged_weihts_only")

    if not os.path.exists(out_dir):
        merge_fsdp_weights(checkpoint_directory, out_dir, safe_serialization=True)

    # copy model config and tokenizer from initial model checkpoint

    initial_model = SentenceLlamaForCausalLM.from_pretrained(
        initial_model_checkpoint, device_map="cpu", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(initial_model_checkpoint)

    hf_out_dir = checkpoint_directory.removesuffix("/pytorch_model_fsdp_0")
    print("Model will be saved to", hf_out_dir)

    state_dict = {}
    with safe_open(os.path.join(out_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            target_k = key.removeprefix("_orig_mod.")
            state_dict[target_k] = f.get_tensor(key)

    initial_model.load_state_dict(state_dict)

    initial_model.save_pretrained(hf_out_dir)
    tokenizer.save_pretrained(hf_out_dir)
