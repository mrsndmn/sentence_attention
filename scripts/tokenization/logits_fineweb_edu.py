import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_model_name = args.pretrained_model_name  # HuggingFaceTB/SmolLM2-1.7B / unsloth/Llama-3.2-1B
    max_length = args.max_length
    num_proc = args.num_proc
    pretrained_model_name_short = pretrained_model_name.split("/")[-1]

    print(f"pretrained_model_name_short: {pretrained_model_name_short}")

    suffix = ""
    if max_length != 1024:
        suffix = f"_max_length_{max_length}"

    target_dir = f"./artifacts/data/fineweb_edu_tokenized_ligits_{pretrained_model_name_short}{suffix}"
    if args.num_shards > 1:
        target_dir = f"{target_dir}_shard_{args.shard_index}_of_{args.num_shards}"

    print("Will be saved to target_dir:", target_dir)

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset = load_dataset(dataset_name, "sample-10BT", num_proc=16, split="train")

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    columns_to_keep = ["input_ids", "attention_mask"]
    columns_to_remove = list(set(dataset.column_names) - set(columns_to_keep))

    # Only half of data!
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)
    # dataset = dataset.remove_columns(columns_to_remove)

    print("shard len", len(dataset))

    # dataset = dataset.map(process_dataset_item, num_proc=num_proc, remove_columns=columns_to_remove)

    per_token_logits = []
    seq_length = []

    with torch.inference_mode():
        for item in tqdm(dataset):
            tokenized_inputs = tokenizer(
                item["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            tokenized_inputs = tokenized_inputs.to(device)
            logits = model(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                use_cache=False,
            ).logits

            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
            target_ids = tokenized_inputs.input_ids[:, 1:].unsqueeze(-1)  # [1, T-1, 1]
            gathered = torch.gather(log_probs, dim=-1, index=target_ids).squeeze(-1)  # [1, T-1]
            token_logprobs = gathered[0].detach().to(torch.float16).cpu().numpy()

            seq_length.append(tokenized_inputs.attention_mask.sum().item())
            per_token_logits.append(token_logprobs)

    ds = Dataset.from_dict({"token_logprobs": per_token_logits, "seq_length": seq_length})

    ds.save_to_disk(target_dir)
    print("Saved to", target_dir)

    print(ds[0])
