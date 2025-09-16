import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm

dataset_path = "/home/ubuntu/workspace/sentence_attention/artifacts/data/fineweb_edu_tokenized_Llama-3.2-1B_max_length_1024_with_eos_token_num_1_merged"


if __name__ == "__main__":

    print("Loading fineweb edu tokenized with eos tokenizer")
    datasets_path_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/data"

    max_length_dataset_suffix = "_max_length_4096"
    dataset_suffix = "_num_4"
    dataset_path = f"{datasets_path_prefix}/fineweb_edu_tokenized_Llama-3.2-1B{max_length_dataset_suffix}_with_eos_token{dataset_suffix}_merged"

    fineweb_dataset = Dataset.load_from_disk(dataset_path)

    fineweb_dataset = fineweb_dataset.shuffle(seed=42)

    input_ids_lengths = []

    for item in tqdm(fineweb_dataset.select(range(1000))):
        input_ids_lengths.append(len([1 for x in item["input_ids"] if x != 128001]))

    plt.hist(input_ids_lengths, bins=100)
    plt.show()
    plt.savefig("/tmp/input_ids_lengths.png")
    print("saved to /tmp/input_ids_lengths.png")
    breakpoint()
