import argparse
import copy

from sentence_attention.evaluation.my_recall import evaluate_synthetic_my_recall
from sentence_attention.models.checkpoint import load_model_from_checkpoint
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM


def patch_tokenizer_prune_gist_tokens(tokenizer, gist_token_ids: list[int]):

    tokenizer = copy.deepcopy(tokenizer)

    initial_tokens = tokenizer.end_of_sentence_token_ids
    pruned_tokens = [token for token in initial_tokens if token not in gist_token_ids]
    tokenizer.end_of_sentence_token_ids = pruned_tokens
    tokenizer.end_of_sentence_tokens_list = [tokenizer.decode([token]) for token in pruned_tokens]

    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/tmp/gist_tokens_pruning.json")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_pruned_tokens", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_model_from_checkpoint(args.model_name)  # type: ignore[assignment]
    model: SentenceLlamaForCausalLM

    model.config._attn_implementation = "sentence_attention"

    all_eos_tokens = tokenizer.end_of_sentence_token_ids
    print("all_eos_tokens", all_eos_tokens)

    all_results = []

    for start_prune_idx in range(len(all_eos_tokens) - args.num_pruned_tokens + 1):
        end_prune_idx = start_prune_idx + args.num_pruned_tokens
        tokens_to_prune = all_eos_tokens[start_prune_idx:end_prune_idx]
        print("tokens_to_prune", tokens_to_prune)
        pruned_tokenizer = patch_tokenizer_prune_gist_tokens(tokenizer, copy.deepcopy(tokens_to_prune))
        print("pruned_tokenizer", pruned_tokenizer.encode("Hello, world!"))

        result = evaluate_synthetic_my_recall(model, pruned_tokenizer, max_samples=args.max_samples)
        all_results.append(result)

    for result_idx, result in enumerate(all_results):
        print(result_idx, result)

    breakpoint()


if __name__ == "__main__":
    main()
