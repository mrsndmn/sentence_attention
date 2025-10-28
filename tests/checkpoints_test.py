import os
import random

from sentence_attention.artifacts.experiments import WORKDIR_PREFIX
from tqdm import tqdm
from transformers import AutoTokenizer


def test_eos_tokens_count():

    for eos_num in os.listdir(os.path.join(WORKDIR_PREFIX, "artifacts/experiments")):
        if eos_num in ["bad_multi_eos_experiments", "frankensteins"]:
            continue

        num_eos_tokens = int(eos_num.split("_")[1])
        if num_eos_tokens != 4:
            continue

        for experiment_dir in tqdm(
            os.listdir(os.path.join(WORKDIR_PREFIX, "artifacts/experiments", eos_num)),
            desc=f"Checking EOS tokens count: {eos_num}",
        ):
            checkpoints_dirs = os.listdir(os.path.join(WORKDIR_PREFIX, "artifacts/experiments", eos_num, experiment_dir))
            if len(checkpoints_dirs) == 0:
                continue

            checkpoints = random.sample(checkpoints_dirs, min(3, len(checkpoints_dirs)))
            for checkpoint in tqdm(checkpoints, desc=f"Checking checkpoints: {experiment_dir}"):
                if not checkpoint.startswith("checkpoint"):
                    continue

                full_checkpoint_path = os.path.join(
                    WORKDIR_PREFIX, "artifacts/experiments", eos_num, experiment_dir, checkpoint
                )

                if os.path.exists(os.path.join(full_checkpoint_path, "pytorch_model_fsdp_0")):
                    continue

                tokenizer = AutoTokenizer.from_pretrained(full_checkpoint_path)
                for i in range(4):
                    eos_token = f"<end_of_sentence_{i}>"
                    assert eos_token in tokenizer.get_vocab(), f"Tokenizer {checkpoint} does not have {eos_token} token"

                assert "<end_of_sentence>" not in tokenizer.get_vocab(), f"Tokenizer {checkpoint} has <end_of_sentence> token"


# check on same name checkpoints in differen eos tokens
def test_same_name_checkpoints_in_different_eos_tokens():

    eos_nums_sets = []

    for eos_num in os.listdir(os.path.join(WORKDIR_PREFIX, "artifacts/experiments")):
        if eos_num == "bad_multi_eos_experiments":
            continue

        experiments = set(os.listdir(os.path.join(WORKDIR_PREFIX, "artifacts/experiments", eos_num)))

        for prev_sets in eos_nums_sets:
            assert (
                len(experiments.intersection(prev_sets)) == 0
            ), f"Experiments {experiments.intersection(prev_sets)} are in different eos tokens directories"

        eos_nums_sets.append(experiments)


if __name__ == "__main__":
    test_same_name_checkpoints_in_different_eos_tokens()
