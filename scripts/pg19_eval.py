import os

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    SentenceLlamaForCausalLM,
    special_token_mask_to_clothest_token_idx_slow,
)
from sentence_attention.models.sentence_llama.scrooge_prefill import scrooge_prefill
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":

    checkpoint_dir = "./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_full_num_eos_tokens_4_IMK8VHPR/checkpoint-1349/"

    max_samples = 1
    # max_tokens_length = 60000
    max_tokens_length = 500

    model_class = SentenceLlamaForCausalLM
    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda")
    model.config._attn_implementation = "sentence_attention"
    # model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    special_token_ids = tokenizer.end_of_sentence_token_ids

    dataset = datasets.Dataset.load_from_disk("../transformers_adaptive_fan_in_fan_out/pg19_test")
    if max_samples != -1:
        dataset = dataset.select(range(max_samples))

    tokens_log_probas = []
    samples_ppls = []

    per_sample_ppls = []
    all_samples_ppls = []
    all_input_ids = []
    all_kv_lengths = []

    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True
    ) as prof:

        for item in tqdm(dataset):

            current_tokens_log_probas = []

            input_ids = tokenizer.encode(item["text"], return_tensors="pt", max_length=max_tokens_length, truncation=True)
            input_ids = input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids).to("cuda")

            special_embeddings_mask = torch.zeros_like(input_ids)
            for special_token_id in special_token_ids:
                special_embeddings_mask = special_embeddings_mask | (input_ids == special_token_id)

            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                special_embeddings_mask,
                num_special_tokens=len(special_token_ids),
            )

            def outputs_hook(input_ids, outputs, prev_sentence_i, sentence_i):
                outputs_logits_normed = F.log_softmax(outputs.logits.float(), dim=-1)

                if sentence_i == input_ids.shape[1]:
                    labels = input_ids[:, prev_sentence_i + 1 : sentence_i]
                    labels = labels.unsqueeze(-1)
                    log_probas = torch.gather(outputs_logits_normed[:, :-1, :], dim=-1, index=labels)
                else:
                    labels = input_ids[:, prev_sentence_i + 1 : sentence_i + 1]
                    labels = labels.unsqueeze(-1)
                    log_probas = torch.gather(outputs_logits_normed, dim=-1, index=labels)

                # print("log_probas", log_probas.shape)
                current_tokens_log_probas.extend(log_probas[0, :, 0].cpu().numpy().tolist())  # noqa: B023

            outputs = scrooge_prefill(
                model,
                input_ids,
                attention_mask=attention_mask,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                outputs_hook=outputs_hook,
            )

            ppl = np.exp(-np.mean(current_tokens_log_probas))
            samples_ppls.append(ppl)
            per_sample_ppls.append(current_tokens_log_probas)
            print("Sample PPL", ppl)
            print("input_ids", input_ids.shape)
            print("kv_length", outputs["past_key_values"].get_seq_length())
            print("compression ratio", input_ids.shape[1] / outputs["past_key_values"].get_seq_length())

            tokens_log_probas.extend(current_tokens_log_probas)
            all_samples_ppls.append(ppl)
            all_input_ids.append(input_ids.shape[1])
            all_kv_lengths.append(outputs["past_key_values"].get_seq_length())

        ppl = np.exp(-np.mean(tokens_log_probas))
        print("Full PPL", ppl)
        print("Samples PPLs", np.mean(samples_ppls), "std", np.std(samples_ppls))

    prof.export_chrome_trace("./pg19_samples_ppls_sentence_attention_trace.json")
    print(
        "Saved trace",
        "./pg19_samples_ppls_sentence_attention_trace.json",
        "file size",
        os.path.getsize("./pg19_samples_ppls_sentence_attention_trace.json") / 1024 / 1024,
        "MB",
    )

    # df = pd.DataFrame(samples_ppls)
    # df.to_csv("./src/transformers/models/llama/benchmarks/data/pg19_samples_ppls.csv", index=False)

    # with open("./src/transformers/models/llama/benchmarks/data/pg19_per_sample_ppls.pkl", "wb") as f:
    #     pickle.dump(per_sample_ppls, f)

    # df = pd.DataFrame({
    #     "input_ids": all_input_ids,
    #     "kv_length": all_kv_lengths,
    #     "ppl": all_samples_ppls,
    # })
    # df.to_csv("./src/transformers/models/llama/benchmarks/data/pg19_samples_ppls_sentence_attention.csv", index=False)

    # plt.hist(tokens_counts, bins=100)
    # plt.savefig("./src/transformers/models/llama/benchmarks/plots/pg19_tokens_counts.png")
