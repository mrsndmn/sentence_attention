#!/usr/bin/env python3
"""
Script to analyze compression rates from benchmark token statistics.

This script reads compression data from:
- artifacts/evaluation/long_bench_tokens_stats.json
- artifacts/evaluation/short_bench_tokens_stats.json

And generates tables showing compression rates for different numbers of gist tokens (1, 2, 4).
"""

import argparse
import json
import os
from typing import Dict, List

from tabulate import tabulate


def load_benchmark_data(file_path: str) -> Dict[str, Dict[str, float]]:
    """Load benchmark data from JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return {}

    with open(file_path) as f:
        return json.load(f)


def calculate_compression_rates(data: Dict[str, Dict[str, float]], gist_tokens: int) -> Dict[str, float]:
    """
    Calculate compression rates for a given number of gist tokens.

    The compression rate is calculated as: num_eos_tokens / total_tokens
    This represents how much of the original content is compressed into gist tokens.
    """
    compression_rates = {}

    for benchmark, stats in data.items():
        total_tokens = stats.get("total_tokens", 0)
        num_eos_tokens = stats.get("num_eos_tokens", 0)

        # assert total_tokens > 0, "total_tokens cant be 0"
        if num_eos_tokens == 0:
            compression_rate = "-"
        else:
            compression_rate = total_tokens / (num_eos_tokens * gist_tokens)
        compression_rates[benchmark] = compression_rate

    return compression_rates


def generate_compression_table(short_data: Dict, long_data: Dict, gist_tokens: int) -> List[List[str]]:
    """Generate a table row for a specific number of gist tokens."""
    short_rates = calculate_compression_rates(short_data, gist_tokens)
    long_rates = calculate_compression_rates(long_data, gist_tokens)

    # Get all unique benchmarks
    all_benchmarks = sorted(set(short_rates.keys()) | set(long_rates.keys()))

    # Create table row
    row = [f"{gist_tokens} gist tokens"]

    for benchmark in all_benchmarks:
        if benchmark in short_rates:
            rate = short_rates[benchmark]
            if isinstance(rate, (float, int)):
                row.append(f"{rate:.2f}")
            else:
                row.append(rate)
        elif benchmark in long_rates:
            rate = long_rates[benchmark]
            if isinstance(rate, (float, int)):
                row.append(f"{rate:.2f}")
            else:
                row.append(rate)
        else:
            row.append("N/A")

    return row


def main():
    """Main function to generate compression rate tables."""
    parser = argparse.ArgumentParser(description="Generate compression rate tables from benchmark token statistics.")
    parser.add_argument(
        "--tablefmt",
        default="latex",
        choices=["grid", "latex", "latex_raw", "latex_booktabs"],
        help="Tabulate format for output tables",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        choices=["all", "short", "long"],
        help="Which benchmarks to include in the output",
    )
    args = parser.parse_args()

    # File paths
    short_file = "artifacts/evaluation/short_bench_tokens_stats.json"
    long_file = "artifacts/evaluation/long_bench_tokens_stats.json"

    # Load data
    print("Loading benchmark data...")
    short_data = load_benchmark_data(short_file)
    long_data = load_benchmark_data(long_file)

    if not short_data and not long_data:
        print("No benchmark data found!")
        return

    # Get all unique benchmarks
    all_benchmarks = sorted(set(short_data.keys()) | set(long_data.keys()))

    # Generate tables for different gist token counts
    gist_token_counts = [1, 2, 4]

    # Determine which benchmarks to show based on args
    if args.benchmarks == "short":
        benchmarks_to_show = sorted(short_data.keys())
        data_to_use = short_data
    elif args.benchmarks == "long":
        benchmarks_to_show = sorted(long_data.keys())
        data_to_use = long_data
    else:  # all
        benchmarks_to_show = all_benchmarks
        data_to_use = {**short_data, **long_data}

    # Generate table
    headers = ["Gist Tokens"] + benchmarks_to_show
    table_data = []

    for gist_tokens in gist_token_counts:
        if args.benchmarks == "all":
            row = generate_compression_table(short_data, long_data, gist_tokens)
        else:
            rates = calculate_compression_rates(data_to_use, gist_tokens)
            row = [f"{gist_tokens} gist tokens"]
            for benchmark in benchmarks_to_show:
                rate = rates.get(benchmark, 0.0)
                row.append(f"{rate:.2f}")
        table_data.append(row)

    # Print table
    print(f"\nCompression Rates by Benchmark and Gist Token Count ({args.benchmarks.upper()}):")
    print(tabulate(table_data, headers=headers, tablefmt=args.tablefmt, floatfmt=".2f"))

    # Print separate tables for short and long benchmarks if showing all
    if args.benchmarks == "all":
        print("\n" + "=" * 60)
        print("SHORT BENCHMARKS")
        print("=" * 60)

        short_benchmarks = sorted(short_data.keys())
        short_headers = ["Gist Tokens"] + short_benchmarks
        short_table_data = []

        for gist_tokens in gist_token_counts:
            short_rates = calculate_compression_rates(short_data, gist_tokens)
            row = [f"{gist_tokens} gist tokens"]
            for benchmark in short_benchmarks:
                rate = short_rates.get(benchmark, 0.0)
                if isinstance(rate, (float, int)):
                    row.append(f"{rate:.2f}")
                else:
                    row.append(rate)
            short_table_data.append(row)

        print(tabulate(short_table_data, headers=short_headers, tablefmt=args.tablefmt, floatfmt=".2f"))

        print("\n" + "=" * 60)
        print("LONG BENCHMARKS")
        print("=" * 60)

        long_benchmarks = sorted(long_data.keys())
        long_headers = ["Gist Tokens"] + long_benchmarks
        long_table_data = []

        for gist_tokens in gist_token_counts:
            long_rates = calculate_compression_rates(long_data, gist_tokens)
            row = [f"{gist_tokens} gist tokens"]
            for benchmark in long_benchmarks:
                rate = long_rates.get(benchmark, 0.0)
                row.append(f"{rate:.2f}")
            long_table_data.append(row)

        print(tabulate(long_table_data, headers=long_headers, tablefmt=args.tablefmt, floatfmt=".2f"))

    # Print summary statistics
    # print("\n" + "="*60)
    # print("SUMMARY STATISTICS")
    # print("="*60)

    # for gist_tokens in gist_token_counts:
    #     short_rates = calculate_compression_rates(short_data, gist_tokens)
    #     long_rates = calculate_compression_rates(long_data, gist_tokens)

    #     all_rates = list(short_rates.values()) + list(long_rates.values())
    #     if all_rates:
    #         avg_rate = sum(all_rates) / len(all_rates)
    #         min_rate = min(all_rates)
    #         max_rate = max(all_rates)

    #         print(f"\n{gist_tokens} gist tokens:")
    #         print(f"  Average compression rate: {avg_rate:.2f}")
    #         print(f"  Min compression rate: {min_rate:.2f}")
    #         print(f"  Max compression rate: {max_rate:.2f}")


if __name__ == "__main__":
    main()
