
import os
import datasets
from datasets import Dataset
from tqdm import tqdm
import shutil
from pathlib import Path

def merge_sharded_datasets():
    """
    Merges sharded datasets found in 'artifacts/data', saves the merged dataset,
    and moves the original sharded directories to 'artifacts/data/old'.
    """
    base_data_path = Path("artifacts/data")
    old_data_path = base_data_path / "old"

    # Create the 'old' directory if it doesn't exist
    old_data_path.mkdir(exist_ok=True)
    print(f"Created directory for old datasets: {old_data_path}")

    # Process each directory in the base data path
    for dataset_dir in base_data_path.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name != "old":
            dataset_path = str(dataset_dir)
            print(f"\nProcessing dataset: {dataset_path}")

            try:
                # Get the list of shard directories
                output_dir = sorted(os.listdir(dataset_path))
                
                if not output_dir:
                    print(f"No shards found in {dataset_path}. Skipping.")
                    continue

                print(f"Found {len(output_dir)} shards in {dataset_path}")

                all_datasets = []
                for data_file in tqdm(output_dir, desc=f'Loading shards for {dataset_dir.name}'):
                    shard_path = os.path.join(dataset_path, data_file)
                    if os.path.isdir(shard_path):
                        dataset = Dataset.load_from_disk(shard_path)
                        all_datasets.append(dataset)
                    else:
                        print(f"Skipping non-directory file: {data_file}")
                
                if not all_datasets:
                    print(f"No valid datasets loaded from shards in {dataset_path}. Skipping.")
                    continue

                # Concatenate all datasets
                print("Concatenating datasets...")
                merged_dataset = datasets.concatenate_datasets(all_datasets)
                print("Concatenation complete.")

                # Save the merged dataset
                new_dataset_name = f"{dataset_dir.name}_merged"
                new_dataset_path = base_data_path / new_dataset_name
                print(f"Saving merged dataset to: {new_dataset_path}")
                merged_dataset.save_to_disk(str(new_dataset_path))
                print("Merged dataset saved.")

                # Move the old dataset directory
                destination_path = old_data_path / dataset_dir.name
                print(f"Moving old dataset directory '{dataset_dir}' to '{destination_path}'")
                shutil.move(str(dataset_dir), str(destination_path))
                print("Old dataset directory moved.")

            except Exception as e:
                print(f"An error occurred while processing {dataset_path}: {e}")

if __name__ == "__main__":
    merge_sharded_datasets()
