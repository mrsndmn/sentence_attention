import time

from tqdm.auto import tqdm

import torch
import torch.profiler
import transformers
from datasets import load_dataset
from sentence_attention.trainer.arguments import SentenceTrainingArguments
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow

from torch.utils.data import DataLoader

# print("Setting inductor config for flex sentence attention")
# inductor_config.max_autotune = True
# inductor_config.max_autotune_gemm_backends = "ATEN,CUBLAS,TRITON"

from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS


if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(SentenceTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    num_eos_tokens = 8
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained("unsloth/Llama-3.2-3B", num_eos_tokens=num_eos_tokens)

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset = load_dataset(dataset_name, "sample-10BT", num_proc=16, split="train")
    dataset = dataset.shard(num_shards=500, index=0)

    special_token_ids = tokenizer.end_of_sentence_token_ids

    def crutch_collator(examples):
        tokenized_text = tokenizer(
            [e["text"] for e in examples], truncation=True, max_length=4096, padding="max_length", return_tensors="pt"
        )

        if len(tokenized_text["attention_mask"].shape) == 3:
            tokenized_text["attention_mask"] = tokenized_text["attention_mask"].squeeze(1)

        input_ids = tokenized_text.input_ids

        special_embeddings_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for special_token_id in special_token_ids:
            special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

        return {
            **tokenized_text,
            "clothest_end_of_sentence_token_idx": clothest_end_of_sentence_token_idx,
        }

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=crutch_collator)

    all_durations = []

    for nw in [8, 16]:
        dataloader = DataLoader(dataset, prefetch_factor=8, batch_size=1, collate_fn=crutch_collator, num_workers=nw)

        start_time = time.perf_counter()
        for _ in tqdm(dataloader, desc=f"nw={nw}"):
            pass
        all_durations.append(time.perf_counter() - start_time)

    print("all_durations", all_durations)
