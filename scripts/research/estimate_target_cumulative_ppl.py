from datasets import Dataset
import numpy as np

if __name__ == "__main__":

    num_eos_tokens = 8
    max_log_sum = 50

    ds = Dataset.load_from_disk("./artifacts/data/fineweb_edu_tokenized_ligits_Llama-3.2-1B_shard_0_of_100000")

    print("len ds", len(ds))

    num_intervals = []

    for item in ds:
        token_logprobs = item["token_logprobs"]
        seq_length = item["seq_length"]

        sum_nlogits = 0
        last_interval_start = 0
        intervals = []
        for i, logit in enumerate(token_logprobs[-(seq_length - 1) :]):
            sum_nlogits -= logit
            if sum_nlogits > max_log_sum:
                intervals.append(i)
                sum_nlogits = 0

        if sum_nlogits != 0:
            intervals.append(seq_length)

        num_intervals.append(len(intervals))

    num_intervals_np = np.array(num_intervals)
    lengths = np.array(item["seq_length"])

    mean_tokens_per_interval = lengths / num_intervals_np

    print("num_intervals_np", f"{num_intervals_np.mean():.2f}", f"{num_intervals_np.std():.2f}")
    print("mean_tokens_per_interval", f"{mean_tokens_per_interval.mean():.2f}", f"{mean_tokens_per_interval.std():.2f}")
