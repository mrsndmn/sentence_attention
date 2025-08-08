import os

from transformers import AutoTokenizer

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention"


def test_eos_tokens_count():

    for eos_num in os.listdir(os.path.join(workdir_prefix, "artifacts/experiments")):
        for experiment_dir in os.listdir(os.path.join(workdir_prefix, "artifacts/experiments", eos_num)):
            for checkpoint in os.listdir(os.path.join(workdir_prefix, "artifacts/experiments", eos_num, experiment_dir)):
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(workdir_prefix, "artifacts/experiments", eos_num, experiment_dir, checkpoint)
                )
                if eos_num == "eos_4":
                    for i in range(4):
                        eos_token = f"<end_of_sentence_{i}>"
                        assert eos_token in tokenizer.get_vocab(), f"Tokenizer {checkpoint} does not have {eos_token} token"

                    assert (
                        "<end_of_sentence>" not in tokenizer.get_vocab()
                    ), f"Tokenizer {checkpoint} has <end_of_sentence> token"
                else:
                    for i in range(4):
                        eos_token = f"<end_of_sentence_{i}>"
                        assert eos_token not in tokenizer.get_vocab(), f"Tokenizer {checkpoint} does not have {eos_token} token"

                    assert (
                        "<end_of_sentence>" in tokenizer.get_vocab()
                    ), f"Tokenizer {checkpoint} does not have <end_of_sentence> token"


# check on same name checkpoints in differen eos tokens
def test_same_name_checkpoints_in_different_eos_tokens():

    eos_nums_sets = []

    for eos_num in os.listdir(os.path.join(workdir_prefix, "artifacts/experiments")):

        experiments = set(os.listdir(os.path.join(workdir_prefix, "artifacts/experiments", eos_num)))

        for prev_sets in eos_nums_sets:
            assert (
                len(experiments.intersection(prev_sets)) == 0
            ), f"Experiments {experiments.intersection(prev_sets)} are in different eos tokens directories"

        eos_nums_sets.append(experiments)


if __name__ == "__main__":
    test_same_name_checkpoints_in_different_eos_tokens()
