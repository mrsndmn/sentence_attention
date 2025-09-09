import argparse

from sentence_attention.evaluation.pg19 import evaluate_pg19_ppl, save_pg19_results_json


def parse_args():
    parser = argparse.ArgumentParser(description="PG19 PPL evaluation (single JSON output)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or HF hub id of the model checkpoint")
    parser.add_argument("--model-type", type=str, choices=["sentence", "vanilla"], default="sentence")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to PG19 dataset saved_to_disk")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output-dir", type=str, default="artifacts/ppl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = evaluate_pg19_ppl(
        checkpoint_dir=args.checkpoint,
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
    )

    out_path = save_pg19_results_json(args.output_dir, results)
    print(f"Saved PG19 results to {out_path}")
