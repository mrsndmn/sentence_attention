import os
import torch
import torch.nn as nn
import pytest
import numpy as np
from transformers import AutoTokenizer, AutoModel

from sentence_attention.boundaries.predictor import BolmoConfig, BolmoLocalEncoder


CKPT_PATH = "artifacts/experiments/bp/unsloth__Llama-3.2-1B/model.pt"


@pytest.mark.skipif(not os.path.exists(CKPT_PATH), reason="checkpoint missing")
def test_bolmo_boundary_inference_pretrained():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    config = BolmoConfig(**ckpt["config"])

    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    model_checkpoint = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    base_model = AutoModel.from_pretrained(model_checkpoint)

    base_embedding_weight = base_model.get_input_embeddings().weight.detach().clone()
    del base_model
    embedding = nn.Embedding.from_pretrained(base_embedding_weight, freeze=True)

    model = BolmoLocalEncoder(config, embedding)
    model.load_state_dict(sd)
    model.to("cuda")
    model.eval()

    sentences = " ".join(
        [
            # '{"test": "value",\n"test2": "value2"}',
            #             """What is swap math ?
            # label: 4
            # When does the average teenager first
            # have intercourse ?
            # label: 5"""
            # "Elizabeth's refusal of Mr. Collins offer of marriage showed an independence seldom seen in heroines of the day. Her refusal of Mr. Darcy while triggered by anger showed a level of independence that left him shocked and stunned.",
        ]
    )
    tokenized = tokenizer.encode(sentences, add_special_tokens=False)
    # combined_ids: list[int] = []
    # boundary_positions: list[int] = []

    # for sent_ids in tokenized:
    #     start = len(combined_ids)
    #     combined_ids.extend(sent_ids)
    #     boundary_positions.append(start + len(sent_ids) - 1)

    input_ids = torch.tensor(tokenized, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        boundary_logprobs, boundary_mask = model(input_ids.to("cuda"))

    boundary_mask[0, 0] = 0

    assert boundary_logprobs.shape == input_ids.shape
    assert boundary_mask.shape == input_ids.shape

    # Debug print tokens with predicted boundaries highlighted in red
    tokens = tokenizer.convert_ids_to_tokens(tokenized)
    tokens = [tok.replace("Ġ", "") for tok in tokens]
    RED, RESET = "\033[91m", "\033[0m"
    pretty_tokens = [
        # f"{RED}{tok} [{tid}]{RESET}" if bool(mask.item()) else f"{tok} [{tid}]"
        f"{RED}{tok}{RESET}" if bool(mask.item()) else f"{tok}"
        for tok, mask, tid in zip(tokens, boundary_mask[0], tokenized)
    ]
    # pretty_tokens = [f"{RED}{tok}{RESET}" if bool(mask.item()) else tok for tok, mask in zip(tokens, boundary_mask[0])]
    print("Tokens:", " ".join(pretty_tokens))

    # for pos in boundary_positions:
    #     assert boundary_mask[0, pos].item(), f"Expected boundary at position {pos} token {tokens[pos]}"

    breakpoint()


def benchmark_bolmo_local_encoder():
    """Benchmark BolmoLocalEncoder inference time for different sequence lengths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found at {CKPT_PATH}. Skipping benchmark.")
        return

    print("Loading model...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    config = BolmoConfig(**ckpt["config"])

    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    model_checkpoint = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    base_model = AutoModel.from_pretrained(model_checkpoint)

    base_embedding_weight = base_model.get_input_embeddings().weight.detach().clone()
    del base_model

    # Create GPU model if available
    model_gpu = None
    if torch.cuda.is_available():
        embedding_gpu = nn.Embedding.from_pretrained(base_embedding_weight.clone(), freeze=True)
        model_gpu = BolmoLocalEncoder(config, embedding_gpu)
        model_gpu.load_state_dict(sd)
        model_gpu.to("cuda")
        model_gpu.eval()

    # Generate test sequences of different lengths
    # Use a base sentence and repeat it to create longer sequences
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base_sentence, add_special_tokens=False)

    # Sequence lengths to test (in tokens)
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    num_warmup = 3
    num_runs = 10

    gpu_times = []

    if model_gpu is not None:
        print("\nBenchmarking GPU inference...")
        for seq_len in seq_lengths:
            # Create input sequence
            num_repeats = (seq_len // len(base_tokens)) + 1
            input_tokens = (base_tokens * num_repeats)[:seq_len]
            input_ids = torch.tensor([input_tokens], dtype=torch.long).to("cuda")

            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model_gpu(input_ids)

            # Benchmark
            torch.cuda.synchronize()
            times = []
            for _ in range(num_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    _ = model_gpu(input_ids)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                times.append(elapsed_time_ms / 1000.0)  # Convert to seconds

            avg_time = np.mean(times)
            std_time = np.std(times)
            gpu_times.append(avg_time)
            print(f"  Seq length {seq_len:5d}: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")

    # Create visualization
    print("\nCreating visualization...")
    os.makedirs("artifacts", exist_ok=True)

    # Linear scale plot
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seq_lengths, [t * 1000 for t in gpu_times], marker="s", label="GPU", linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("BolmoLocalEncoder Inference Time vs Sequence Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path_linear = "artifacts/bolmo_local_encoder_benchmark_linear.png"
    plt.savefig(output_path_linear, dpi=150, bbox_inches="tight")
    print(f"Saved linear scale plot to {output_path_linear}")
    plt.close()

    # Log scale plot
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seq_lengths, [t * 1000 for t in gpu_times], marker="s", label="GPU", linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("BolmoLocalEncoder Inference Time vs Sequence Length (Log Scale)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    output_path_log = "artifacts/bolmo_local_encoder_benchmark_log.png"
    plt.savefig(output_path_log, dpi=150, bbox_inches="tight")
    print(f"Saved log scale plot to {output_path_log}")
    plt.close()


if __name__ == "__main__":
    benchmark_bolmo_local_encoder()
