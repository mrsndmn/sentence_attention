import argparse
import json

import yaml
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from sentence_attention.evaluation.benchmarks import long_benchmarks, short_benchmarks
from sentence_attention.evaluation.evaluation import build_evaluation_pipeline
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM

artifacts_prefix = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts/experiments"

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--type", type=str, default="short", choices=["short", "long"])
    args = argparse.parse_args()

    checkpoint_path = artifacts_prefix + "/eos_1/sentence_Llama-3.2-1B_ft_full_L1DB3Z21/checkpoint-1349"
    eos_1_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # task -> { total_tokens: int, num_eos_tokens: int, compressed_ratio: float }

    if args.type == "short":
        short_bench_tokens_stats = {}
        eos_1_model = SentenceLlamaForCausalLM.from_pretrained(checkpoint_path)

        for benchmark in short_benchmarks:
            # for benchmark in ['winogrande']:
            if benchmark == "pg19":
                continue

            print(benchmark)
            pipeline = build_evaluation_pipeline(eos_1_model, benchmark)
            pipeline_model = pipeline.model

            total_tokens = 0
            num_eos_tokens = 0

            for request_type, requests in pipeline.requests.items():
                for request in tqdm(requests, desc=f"Processing {benchmark} {request_type}"):
                    if request.context == "":
                        request.tokenized_context = [pipeline_model.tokenizer.eos_token_id]
                        request.tokenized_continuation = pipeline_model.tok_encode(request.choice)
                    else:
                        # The following line is mandatory for compatibility with the harness
                        request.tokenized_context, request.tokenized_continuation = pipeline_model.tok_encode_pair(
                            request.context, request.choice, pairwise=pipeline_model.pairwise_tokenization
                        )

                    num_eos_tokens += sum(
                        1 for x in request.tokenized_context if x == pipeline_model.tokenizer.end_of_sentence_token_ids[0]
                    )
                    total_tokens += len(request.tokenized_context)

            compressed_ratio = num_eos_tokens / total_tokens
            print(f"benchmark: {benchmark}, compressed_ratio: {compressed_ratio}")

            short_bench_tokens_stats[benchmark] = {
                "total_tokens": total_tokens,
                "num_eos_tokens": num_eos_tokens,
                "compressed_ratio": compressed_ratio,
            }

        short_benchmark_stats_path = "artifacts/evaluation/short_bench_tokens_stats.json"
        with open(short_benchmark_stats_path, "w") as f:
            json.dump(short_bench_tokens_stats, f, indent=4)
        print("Saved to", short_benchmark_stats_path)

    elif args.type == "long":
        long_bench_tokens_stats = {}

        # HELMET imports
        from dataclasses import dataclass

        from data import TestItemDataset, load_data
        from model_utils import HFModel

        @dataclass
        class LongConfig:
            input_max_length: int
            datasets: str
            generation_max_length: int
            test_files: str
            demo_files: str
            use_chat_template: bool
            max_test_samples: int
            shots: int
            stop_new_line: bool
            model_name_or_path: str
            seed: int = 42

        hf_model_wrapper = HFModel(checkpoint_path)

        for benchmark in long_benchmarks:
            print(benchmark)

            config_file = f"../HELMET/configs/{benchmark}_tiny.yaml"
            with open(config_file) as f:
                config = yaml.safe_load(f)

            lconf = LongConfig(**config)

            datasets = lconf.datasets.split(",")
            test_files = lconf.test_files.split(",")
            demo_files = lconf.demo_files.split(",")

            num_eos_tokens = 0
            total_tokens = 0

            for dataset, test_file, demo_file in zip(datasets, test_files, demo_files):

                if test_file != "":
                    test_file = "../HELMET/" + test_file

                if demo_file != "":
                    demo_file = "../HELMET/" + demo_file

                data = load_data(lconf, dataset, test_file, demo_file)
                print(f"loaded {len(data['data'])} samples from {dataset}")

                test_item_dataset = TestItemDataset(data, hf_model_wrapper, eos_1_tokenizer)

                for data_item in test_item_dataset:
                    input_ids = data_item[0]["input_ids"][0].numpy().tolist()

                    num_eos_tokens += sum(1 for x in input_ids if x == eos_1_tokenizer.end_of_sentence_token_ids[0])
                    total_tokens += len(input_ids)

            long_bench_tokens_stats[benchmark] = {
                "total_tokens": total_tokens,
                "num_eos_tokens": num_eos_tokens,
                "compressed_ratio": num_eos_tokens / total_tokens,
            }

        long_benchmark_stats_path = "artifacts/evaluation/long_bench_tokens_stats.json"
        with open(long_benchmark_stats_path, "w") as f:
            json.dump(long_bench_tokens_stats, f, indent=4)
        print("Saved to", long_benchmark_stats_path)

    else:
        raise ValueError(f"Invalid type: {args.type}")
