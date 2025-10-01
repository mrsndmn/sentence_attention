import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":

    print("Loading dclm with eos tokenizer")

    dataset = Dataset.load_from_disk(
        "./artifacts/data/dclm_tokenized_Llama-3.2-1B_max_length_16384_with_eos_token_num_4_merged"
    )
    print(f"Dataset length: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

    input_ids_lengths = []

    for item in tqdm(dataset.shuffle(seed=42).select(range(1000))):
        input_ids_lengths.append(len([1 for x in item["input_ids"] if x != 128001]))

    plt.hist(input_ids_lengths, bins=100)
    plt.show()
    plt.savefig("/tmp/input_ids_lengths_dclm.png")
    print("saved to /tmp/input_ids_lengths_dclm.png")
    breakpoint()
