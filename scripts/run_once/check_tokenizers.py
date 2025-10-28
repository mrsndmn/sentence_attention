import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def compute_sha256(text: str) -> str:
    """Return short sha256 digest for readability."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_backend_tokenizer_json(tokenizer: PreTrainedTokenizer) -> str | None:
    """Return backend tokenizer JSON string if available (fast tokenizers)."""
    try:
        if isinstance(tokenizer, PreTrainedTokenizerFast) and hasattr(tokenizer, "_tokenizer"):
            # tokenizers>=0.13 supports to_str(); fall back to to_json()
            backend = tokenizer._tokenizer
            if hasattr(backend, "to_str"):
                return backend.to_str()
            if hasattr(backend, "to_json"):
                return backend.to_json()
    except Exception:
        pass
    return None


def summarize_vocab_diff(a_vocab: Dict[str, int], b_vocab: Dict[str, int]) -> Tuple[int, int, int, List[Tuple[str, int, int]]]:
    """Summarize differences between two token->id mappings.

    Returns: (only_in_a, only_in_b, changed, examples)
    """
    a_keys = set(a_vocab.keys())
    b_keys = set(b_vocab.keys())
    only_in_a = a_keys - b_keys
    only_in_b = b_keys - a_keys
    shared = a_keys & b_keys
    changed = []
    for token in shared:
        ida = a_vocab[token]
        idb = b_vocab[token]
        if ida != idb:
            changed.append((token, ida, idb))
            if len(changed) >= 10:
                break
    return len(only_in_a), len(only_in_b), len(changed), changed


def compare_tokenizers(a: PreTrainedTokenizer, b: PreTrainedTokenizer) -> Dict[str, Any]:
    """Compare key aspects of two HF tokenizers."""
    result: Dict[str, Any] = {"equal": True, "diffs": []}

    # Class and fast/slow
    a_cls = type(a).__name__
    b_cls = type(b).__name__
    if a_cls != b_cls:
        result["equal"] = False
        result["diffs"].append({"type": "class", "a": a_cls, "b": b_cls})

    a_fast = isinstance(a, PreTrainedTokenizerFast)
    b_fast = isinstance(b, PreTrainedTokenizerFast)
    if a_fast != b_fast:
        result["equal"] = False
        result["diffs"].append({"type": "fast_mode", "a": a_fast, "b": b_fast})

    # Vocab and added vocab
    a_vocab = a.get_vocab()
    b_vocab = b.get_vocab()
    if a_vocab != b_vocab:
        result["equal"] = False
        only_a, only_b, changed_count, changed_examples = summarize_vocab_diff(a_vocab, b_vocab)
        result["diffs"].append(
            {
                "type": "vocab",
                "a_size": len(a_vocab),
                "b_size": len(b_vocab),
                "only_in_a": only_a,
                "only_in_b": only_b,
                "changed_examples": changed_examples,
            }
        )

    a_added = a.get_added_vocab()
    b_added = b.get_added_vocab()
    if a_added != b_added:
        result["equal"] = False
        only_a, only_b, changed_count, changed_examples = summarize_vocab_diff(a_added, b_added)
        result["diffs"].append(
            {
                "type": "added_vocab",
                "a_size": len(a_added),
                "b_size": len(b_added),
                "only_in_a": only_a,
                "only_in_b": only_b,
                "changed_examples": changed_examples,
            }
        )

    # Special tokens map
    a_special = a.special_tokens_map
    b_special = b.special_tokens_map
    if a_special != b_special:
        result["equal"] = False
        result["diffs"].append({"type": "special_tokens_map", "a": a_special, "b": b_special})

    # Chat template (if present)
    a_chat = getattr(a, "chat_template", None)
    b_chat = getattr(b, "chat_template", None)
    if a_chat != b_chat:
        result["equal"] = False
        a_digest = compute_sha256(a_chat) if isinstance(a_chat, str) else None
        b_digest = compute_sha256(b_chat) if isinstance(b_chat, str) else None
        result["diffs"].append({"type": "chat_template", "a": a_digest, "b": b_digest})

    # Backend (fast) tokenizer JSON
    a_json = get_backend_tokenizer_json(a)
    b_json = get_backend_tokenizer_json(b)
    if (a_json is None) != (b_json is None):
        result["equal"] = False
        result["diffs"].append({"type": "backend_json_presence", "a": a_json is not None, "b": b_json is not None})
    elif a_json is not None and b_json is not None:
        a_digest = compute_sha256(a_json)
        b_digest = compute_sha256(b_json)
        if a_digest != b_digest:
            result["equal"] = False
            # Provide small top-level summary to aid debugging
            try:
                a_top = json.loads(a_json)
                b_top = json.loads(b_json)
                a_model = a_top.get("model", {}) if isinstance(a_top, dict) else {}
                b_model = b_top.get("model", {}) if isinstance(b_top, dict) else {}
                summary = {
                    "a_model_type": a_model.get("type"),
                    "b_model_type": b_model.get("type"),
                }
            except Exception:
                summary = {}
            result["diffs"].append({"type": "backend_json_digest", "a": a_digest, "b": b_digest, "summary": summary})

    # Encoding tests
    test_texts: List[str] = [
        "Hello world!",
        "The brown fox. Jumps over dog.",
        "\n\nA	string with\twhitespace and unicode – café.",
        "<s> [INST] Test chat template [/INST] </s>",
    ]
    encoding_mismatches: List[Tuple[str, List[int], List[int]]] = []
    for text in test_texts:
        try:
            a_ids = a(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
            b_ids = b(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
            if a_ids != b_ids:
                encoding_mismatches.append((text, a_ids[:32], b_ids[:32]))
        except Exception as e:
            encoding_mismatches.append((f"{text} (error: {e})", [], []))
    if encoding_mismatches:
        result["equal"] = False
        result["diffs"].append({"type": "encoding", "mismatches": encoding_mismatches})

    return result


def print_report(model_a: str, model_b: str, cmp: Dict[str, Any]) -> None:
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print("")
    print(f"Overall equal: {'YES' if cmp['equal'] else 'NO'}")
    if not cmp["equal"]:
        print("\nDifferences:")
        for diff in cmp["diffs"]:
            dtype = diff.get("type")
            if dtype == "class":
                print(f"- Tokenizer class differs: {diff['a']} vs {diff['b']}")
            elif dtype == "fast_mode":
                print(f"- Fast/slow mode differs: {diff['a']} vs {diff['b']}")
            elif dtype == "vocab":
                print(
                    f"- Vocab differs: sizes {diff['a_size']} vs {diff['b_size']}; "
                    f"only_in_a={diff['only_in_a']} only_in_b={diff['only_in_b']}; "
                    f"changed_examples={diff['changed_examples']}"
                )
            elif dtype == "added_vocab":
                print(
                    f"- Added vocab differs: sizes {diff['a_size']} vs {diff['b_size']}; "
                    f"only_in_a={diff['only_in_a']} only_in_b={diff['only_in_b']}; "
                    f"changed_examples={diff['changed_examples']}"
                )
            elif dtype == "special_tokens_map":
                print(f"- Special tokens map differs:\n  A: {diff['a']}\n  B: {diff['b']}")
            elif dtype == "chat_template":
                print(f"- Chat template differs (sha256-16): A={diff['a']} B={diff['b']}")
            elif dtype == "backend_json_presence":
                print(f"- Backend tokenizer JSON presence differs: A={diff['a']} B={diff['b']}")
            elif dtype == "backend_json_digest":
                extra = diff.get("summary", {})
                print(f"- Backend tokenizer JSON differs (sha256-16): A={diff['a']} B={diff['b']} summary={extra}")
            elif dtype == "encoding":
                print("- Encoded ids differ for sample texts (first 32 ids shown):")
                for text, a_ids, b_ids in diff["mismatches"]:
                    print(f"  Text: {text!r}")
                    print(f"    A: {a_ids}")
                    print(f"    B: {b_ids}")
            else:
                print(f"- {dtype}: {diff}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check equality of two Hugging Face tokenizers")
    parser.add_argument("--model-a", type=str, default="unsloth/llama-3-8b")
    parser.add_argument("--model-b", type=str, default="unsloth/Llama-3.2-3B")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading")
    parser.add_argument("--use-slow", action="store_true", help="Force slow tokenizer if available")
    args = parser.parse_args()

    # Respect HF cache dir if provided via environment; project convention sets HF_HOME.
    cache_dir = os.environ.get("HF_HOME")

    tok_a = AutoTokenizer.from_pretrained(
        args.model_a,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.use_slow,
        cache_dir=cache_dir,
    )
    tok_b = AutoTokenizer.from_pretrained(
        args.model_b,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.use_slow,
        cache_dir=cache_dir,
    )

    cmp = compare_tokenizers(tok_a, tok_b)
    print_report(args.model_a, args.model_b, cmp)


if __name__ == "__main__":
    main()
