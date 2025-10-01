import argparse
import os

from datasets import Dataset, concatenate_datasets

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    all_shards = [
        Dataset.load_from_disk(
            "artifacts/data/dclm_tokenized_Llama-3.2-1B_max_length_16384_with_eos_token_num_4_merged_long_balance/"
        )
    ]

    for shard_index in range(args.num_shards):
        shard_dir = args.target_dir + f"_shard_{shard_index}_of_{args.num_shards}"
        if not os.path.exists(shard_dir):
            print(f"Shard {shard_dir} does not exist")
            continue

        shard = Dataset.load_from_disk(shard_dir)
        print(f"Shard {shard_index} length: {len(shard)}")
        all_shards.append(shard)

    print(f"Merging {len(all_shards)} shards")

    dataset = concatenate_datasets(all_shards)
    print(f"Dataset length: {len(dataset)}")
    print(f"Saving dataset to {args.target_dir}")
    input("Press Enter to continue")

    dataset.save_to_disk(args.target_dir)
