import random

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":

    print("Loading fineweb edu tokenized with eos tokenizer")
    dataset_name = "mlfoundations/dclm-baseline-1.0"
    dataset_files = []
    for j in range(1, 11):
        for i in range(50):
            dataset_files.append(f"global-shard_{j:02}_of_10/local-shard_0_of_10/shard_{i:08d}_processed.jsonl.zst")

    dataset = load_dataset(dataset_name, num_proc=16, split="train", data_files=dataset_files)

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

    total_tokens = 0

    input_ids_lengths = []

    samples_count = 10000
    bins_counts = [0] * 100
    total_dataset_size = 2500
    max_bin_size = total_dataset_size / len(bins_counts)

    dclm_only_texts = []

    pbar = tqdm(total=total_dataset_size)

    for item in dataset.shuffle(seed=42):
        # for item in tqdm(dataset.select(range(samples_count))):
        tokenized = tokenizer(item["text"])
        cur_len = len(tokenized["input_ids"])

        if cur_len > 16384:
            continue

        if pbar.n > total_dataset_size:
            break

        current_bin = -1
        for i in range(len(bins_counts)):
            max_value = 16384 / len(bins_counts) * (i + 1)
            min_value = 16384 / len(bins_counts) * i
            if cur_len <= max_value and cur_len >= min_value:
                current_bin = i
                break

        assert current_bin != -1
        if bins_counts[current_bin] > max_bin_size:
            if cur_len > (8192 * (0.5 + random.random())):
                pass
            else:
                continue

        bins_counts[current_bin] += 1

        total_tokens += cur_len
        input_ids_lengths.append(cur_len)
        dclm_only_texts.append(item["text"])

        pbar.update(1)

    print("total_tokens", total_tokens)
    # print("approx total tokens", total_tokens / samples_count * len(dataset))

    plt.hist(input_ids_lengths, bins=len(bins_counts))
    plt.show()
    plt.savefig("/tmp/input_ids_lengths_dclm.png")
    print("saved to /tmp/input_ids_lengths_dclm.png")
    breakpoint()

    while True:
        if input("Save to disk? (y/n)") == "y":
            ds = Dataset.from_dict({"text": dclm_only_texts})
            ds = ds.shuffle(seed=42)
            ds.save_to_disk("artifacts/data/dclm_texts")
            print("saved to artifacts/data/dclm_texts")
            break
