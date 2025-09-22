import os

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

CHECKPOINT_DIR = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments_in_progress/sentence_Llama-3.1-8B_ft_4k_full_num_eos_tokens_4_MXD7H0E8/checkpoint-2000/pytorch_model_fsdp_0/"
TORCH_SAVE_CHECKPOINT_DIR = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments_in_progress/sentence_Llama-3.1-8B_ft_4k_full_num_eos_tokens_4_MXD7H0E8/checkpoint-torch-2000/"

os.makedirs(TORCH_SAVE_CHECKPOINT_DIR, exist_ok=True)

print(f"Converting DCP model to torch.save model in {TORCH_SAVE_CHECKPOINT_DIR}")

# convert dcp model to torch.save (assumes checkpoint was generated as above)
dcp_to_torch_save(CHECKPOINT_DIR, os.path.join(TORCH_SAVE_CHECKPOINT_DIR, "pytorch_model.pth"))
