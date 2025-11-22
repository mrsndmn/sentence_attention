import argparse
import math
import os
from typing import List, Optional, Tuple

import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _read_text_arg(text: Optional[str], file_path: Optional[str]) -> str:
    if text is not None and len(text.strip()) > 0:
        logger.debug("Using text from --text argument (%d chars).", len(text))
        return text
    if file_path is not None:
        logger.info("Reading text from file: %s", file_path)
        with open(file_path) as f:
            return f.read()
    # Fallback to stdin
    try:
        import sys

        logger.info("Reading text from stdin.")
        return sys.stdin.read()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("No text provided. Use --text, --file, or pipe stdin.") from exc


def _compute_arrays(
    token_logprobs: List[float],
    cumulative: bool,
    reset_on_max_value: Optional[float] = None,
) -> np.ndarray:
    if len(token_logprobs) == 0:
        return np.array([])
    logprobs_arr = np.array(token_logprobs, dtype=np.float64)  # log p(token | context)

    # Per-token perplexity = exp(-log p_token)
    # per_token_ppl = np.exp(-logprobs_arr)
    per_token_ppl = -logprobs_arr

    if not cumulative:
        return per_token_ppl

    # Cumulative perplexity up to position t:
    # ppl_t = exp( (sum_{i<=t} -log p_i) / t )
    cum_sum = np.cumsum(per_token_ppl)

    if reset_on_max_value is None:
        return cum_sum

    # With resets based on a per-token ppl threshold
    y = np.zeros_like(per_token_ppl)
    segment_sum = 0.0

    print("cum_sum", cum_sum)
    for i, cs_ppl in enumerate(per_token_ppl):
        # accumulate within current segment
        segment_sum += cs_ppl
        y[i] = segment_sum
        # if token-wise ppl exceeds threshold, start a new segment next step
        if cs_ppl > reset_on_max_value:
            segment_sum = 0.0

    print("y", y)

    return y


def _plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    tokens_text: Optional[List[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    reset_positions: Optional[List[int]] = None,
    sentence_boundaries: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    # Use a non-interactive backend by default; switch to show if requested.
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, label="perplexity", color="tab:blue")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Draw token strings on x-axis
    if tokens_text is not None and len(tokens_text) == len(x):
        n = len(tokens_text)
        max_labels = 128
        if n <= max_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens_text, rotation=90, fontsize=7)
        else:
            step = int(math.ceil(n / max_labels))
            tick_positions = x[::step]
            tick_labels = tokens_text[::step]
            # ensure last token is included
            if tick_positions[-1] != x[-1]:
                tick_positions = np.append(tick_positions, x[-1])
                tick_labels = tick_labels + [tokens_text[-1]]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        # Extra padding for rotated labels
        fig.subplots_adjust(bottom=0.25)

    if reset_positions:
        for pos in reset_positions:
            ax.axvline(pos, color="tab:red", linestyle="--", alpha=0.4, linewidth=1.0)

    if sentence_boundaries:
        for pos in sentence_boundaries:
            ax.axvline(pos, color="tab:green", linestyle=":", alpha=0.3, linewidth=1.0)

    ax.legend(loc="upper right")

    if output_path is None and not show:
        output_path = "perplexity_plot.png"

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        logger.info("Saved plot to: %s", output_path)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize per-token/cumulative perplexity over a text.")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path for ppl compute.")
    parser.add_argument(
        "--text",
        type=str,
        default="Moscow is the capital of Russia, a multinational city on the Moskva River in the western part of the country. The medieval Kremlin fortress, the residence of the Russian president, is located in its historical center.",
        help="Text to analyze.",
    )
    parser.add_argument("--file", type=str, default="", help="Read text from file.")
    parser.add_argument("--cumulative", dest="cumulative", action="store_true", help="Draw cumulative ppl.")
    parser.add_argument("--cummulative", dest="cumulative", action="store_true", help="Alias of --cumulative.")
    parser.add_argument("--reset-on-sentences", action="store_true", help="Reset context at sentence boundaries.")
    parser.add_argument(
        "--reset-on-max-value",
        type=float,
        default=None,
        help="Reset context after a token exceeds this per-token perplexity value.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for models.")
    parser.add_argument("--output", type=str, default=None, help="If set, save plot to this path.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Limit y-axis (perplexity) to the given range.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger.debug("Parsed arguments: %s", vars(args))

    text = _read_text_arg(args.text, args.file)
    if len(text.strip()) == 0:
        raise ValueError("Empty text after reading input.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s (CUDA available=%s)", device, torch.cuda.is_available())

    logger.info("Loading tokenizer and model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else None, trust_remote_code=args.trust_remote_code
    )
    model.eval()
    model.to(device)
    logger.info("Model loaded and moved to device.")

    with torch.no_grad():
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        outputs = model(input_ids)
        logits = outputs.logits  # [1, T, V]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
        target_ids = input_ids[:, 1:].unsqueeze(-1)  # [1, T-1, 1]
        gathered = torch.gather(log_probs, dim=-1, index=target_ids).squeeze(-1)  # [1, T-1]
        token_logprobs = gathered[0].detach().cpu().numpy().astype(np.float64)  # log p for tokens 1..T-1

    # Token strings aligned with targets (skip the first, which has no preceding context)
    tokens_text = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[1:])
    tokens_text = [t.replace("Ġ", "_") for t in tokens_text]

    y = _compute_arrays(
        token_logprobs,
        cumulative=args.cumulative,
        reset_on_max_value=args.reset_on_max_value,
        # reset_on_sentences=args.reset_on_sentences,
    )
    x = np.arange(1, len(y) + 1, dtype=np.int64)

    title_mode = "Cumulative" if args.cumulative else "Token-wise"
    title_parts = [title_mode, "Perplexity"]
    if args.cumulative and args.reset_on_sentences:
        title_parts.append("(resets on sentences)")
    if args.cumulative and args.reset_on_max_value is not None:
        title_parts.append(f"(reset on ppl>{args.reset_on_max_value})")
    title = " ".join(title_parts)

    logger.info(
        "Rendering plot (points=%d, cumulative=%s, output=%s, show=%s).",
        len(x),
        args.cumulative,
        args.output,
        args.show,
    )

    # Ensure lengths match (they should)
    if len(tokens_text) != len(x):
        tokens_text = tokens_text[: len(x)]

    _plot(
        x=x,
        y=y,
        title=title,
        tokens_text=tokens_text,
        # ylim=tuple(args.ylim) if args.ylim is not None else None,
        output_path=args.output,
        show=bool(args.show),
    )

    breakpoint()


if __name__ == "__main__":
    main()
