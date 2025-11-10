import argparse
import contextlib
import os
import re
import shutil
from tqdm.auto import tqdm
import time
import glob
import torch
import errno
from accelerate.utils import merge_fsdp_weights
from safetensors import safe_open
from transformers import AutoTokenizer

from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_directory", type=str, help="The checkpoint directory to merge")
    parser.add_argument("--checkpoint_glob", type=str, help="The glob pattern to match the checkpoint directories to merge")
    parser.add_argument(
        "--watch", action="store_true", help="Watch the checkpoint directories and merge them when they are created"
    )
    parser.add_argument("--keep_only_last_optimizer", action="store_true", help="Keep only the last optimizer checkpoint")

    parser.add_argument(
        "--initial_model_checkpoint", type=str, help="EOSO checkpoint that was used as initialisation for full pretraining"
    )

    args = parser.parse_args()
    checkpoint_directories = args.checkpoint_glob
    initial_model_checkpoint = args.initial_model_checkpoint

    checkpoint_directories = checkpoint_directories.removesuffix("/")

    # assert checkpoint_directories.endswith("/pytorch_model_fsdp_0"), "Checkpoint directory must end with /pytorch_model_fsdp_0"

    while True:

        def _extract_checkpoint_step(path: str) -> int:
            match = re.search(r"checkpoint-(\d+)", path)
            return int(match.group(1)) if match else -1

        sorted_checkpoint_directories = sorted(glob.glob(checkpoint_directories), key=_extract_checkpoint_step)
        print("sorted_checkpoint_directories", sorted_checkpoint_directories)
        for checkpoint_directory in tqdm(sorted_checkpoint_directories):
            # out dir is renamed from pytorch_model_fsdp_0 to pytorch_model_merged
            # so we need to rename the out dir to pytorch_model_fsdp_0
            checkpoint_directory_root = checkpoint_directory
            checkpoint_directory = os.path.join(checkpoint_directory, "pytorch_model_fsdp_0")

            out_dir = checkpoint_directory.replace("pytorch_model_fsdp_0", "hf_model_merged_weihts_only")
            print("Checking", out_dir)

            if os.path.exists(checkpoint_directory) and os.stat(checkpoint_directory).st_ctime > time.time() - 10 * 60:
                # Time for all fsdp checkpoints to be saved
                sleep_time = 10 * 60 - (time.time() - os.stat(checkpoint_directory).st_ctime)
                time.sleep(sleep_time)
                print("Sleeping for", sleep_time, "seconds")

            if not os.path.exists(out_dir) and not os.path.exists(os.path.join(checkpoint_directory_root, "config.json")):
                print("Merging", checkpoint_directory, "to", out_dir)
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

            def rmtree_with_retry(path: str, max_retries: int = 1, base_sleep: float = 0.5) -> None:
                """Remove a directory tree robustly on NFS.

                Retries when encountering ENOTEMPTY (often due to NFS .nfs* placeholders from open file handles).
                Also attempts to remove any .nfs* files that may appear during deletion.
                """
                if not os.path.exists(path):
                    return

                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(path)
                        return
                    except OSError as e:
                        # ENOTEMPTY can be 39 on Linux; handle generically via errno.ENOTEMPTY as well
                        if e.errno in {errno.ENOTEMPTY, 39}:
                            # Attempt to remove transient .nfs* files, then backoff and retry
                            with contextlib.suppress(Exception):
                                for root, _dirs, files in os.walk(path):
                                    for fname in files:
                                        if fname.startswith(".nfs"):
                                            fpath = os.path.join(root, fname)
                                            with contextlib.suppress(Exception):
                                                os.unlink(fpath)

                            time.sleep(base_sleep * (attempt + 1))
                            continue
                        elif e.errno == errno.ENOENT:
                            return
                        else:
                            pass
                            print("Failed to remove directory", path, "Will retry later")
                # Best-effort final attempt without raising
                with contextlib.suppress(Exception):
                    shutil.rmtree(path, ignore_errors=True)

            rmtree_with_retry(checkpoint_directory)  # /hf_model_merged_weihts_only
            rmtree_with_retry(out_dir)  # /pytorch_model_fsdp_0

        if args.keep_only_last_optimizer:
            for checkpoint_rm_optimizer_state in sorted_checkpoint_directories[:-1]:
                optimizer_state_path = os.path.join(checkpoint_rm_optimizer_state, "optimizer_0")
                if os.path.exists(optimizer_state_path):
                    shutil.rmtree(optimizer_state_path)
                    print("Removed optimizer state from", optimizer_state_path)

        if not args.watch:
            break
        else:
            time.sleep(60)
