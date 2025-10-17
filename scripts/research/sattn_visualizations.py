import os

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from sentence_attention.evaluation.my_recall import generate_random_sample
from sentence_attention.models.checkpoint import load_model_from_checkpoint
from torch.nn import functional as F


def visualize_tokens_logits(tokens_logits, tokens, figsize=(18, 6), output_path="/tmp/tokens_logits.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=figsize)
    positions = list(range(len(tokens)))
    plt.bar(positions, tokens_logits)
    plt.xticks(positions, tokens, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved to {output_path}")


def visualize_hidden_states(hidden_states, output_path="/tmp/hidden_states.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Normalize input to a list of layer tensors: [(batch, seq, hidden), ...]
    layers = list(hidden_states)

    # Compute number of subplots
    num_layers = len(layers)
    cols = min(4, max(1, num_layers))
    rows = (num_layers + cols - 1) // cols

    figsize = (cols * 4, rows * 4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Build a flat list of axes
    if isinstance(axes, Axes):
        axes_flat = [axes]
    else:
        try:
            axes_flat = list(axes.ravel())
        except AttributeError:
            # Fallback for list/tuple
            axes_flat = []
            if isinstance(axes, (list, tuple)):
                for item in axes:
                    if isinstance(item, (list, tuple)):
                        axes_flat.extend(item)
                    else:
                        axes_flat.append(item)
            else:
                axes_flat = [axes]

    last_im = None
    for idx, layer_hs in enumerate(layers):
        ax = axes_flat[idx]
        # Use first item in batch
        hs = layer_hs[0]  # (seq_len, hidden_size)
        # Compute cosine similarity matrix across tokens
        hs = F.normalize(hs, p=2, dim=-1)
        sim = torch.matmul(hs, hs.transpose(0, 1))  # (seq_len, seq_len)
        sim_np = sim.detach().float().cpu().numpy()

        last_im = ax.imshow(sim_np, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Layer {idx}")
        ax.set_xlabel("Token index")
        ax.set_ylabel("Token index")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(num_layers, len(axes_flat)):
        axes_flat[j].axis("off")

    # Add a single colorbar for the figure
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_flat, fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/tmp/tokens_logits.png")
    parser.add_argument("--output_hidden_states_path", type=str, default="/tmp/hidden_states.png")

    args = parser.parse_args()

    model_name = args.model_name

    model, tokenizer = load_model_from_checkpoint(model_name)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # type: ignore[arg-type]
        model.eval()

        text = generate_random_sample(num_examples=2)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs, output_hidden_states=True)

        log_probs = F.softmax(outputs.logits, dim=-1)
        input_ids_tensor = inputs.input_ids  # type: ignore[attr-defined]
        labels = input_ids_tensor[:, 1:]

        tokens_logits = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1))

        tokens_logits = tokens_logits[0, :, 0].float().cpu().numpy()
        labels = labels[0, :].cpu().numpy()

        # Convert token ids to readable token strings for x-axis labels
        token_texts = tokenizer.convert_ids_to_tokens(labels.tolist())
        token_texts = [t.removeprefix("Ġ") for t in token_texts]
        token_texts = [t.replace("Ċ", "\\n") for t in token_texts]

        # Visualize tokens_logits with proper x-axis positions and labels
        visualize_tokens_logits(tokens_logits, tokens=token_texts, output_path=args.output_path)

        visualize_hidden_states(outputs.hidden_states, output_path=args.output_hidden_states_path)

        breakpoint()
