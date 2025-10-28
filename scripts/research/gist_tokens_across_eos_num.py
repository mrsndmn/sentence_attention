import os

import matplotlib.pyplot as plt
import torch
from sentence_attention.models.checkpoint import load_model_from_checkpoint
from torch.nn import functional as F


def visualize_gist_embeddings(gist_embeddings, output_path="/tmp/gist_embeddings.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize input to a 2D tensor of shape (num_gists, hidden_size)
    if isinstance(gist_embeddings, (list, tuple)):
        if len(gist_embeddings) == 0:
            raise ValueError("gist_embeddings is empty")
        emb = torch.stack([e.detach() for e in gist_embeddings], dim=0)
    elif isinstance(gist_embeddings, torch.Tensor):
        emb = gist_embeddings.detach()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        elif emb.dim() != 2:
            raise ValueError(f"gist_embeddings tensor must be 1D or 2D, got shape {tuple(emb.shape)}")
    else:
        raise TypeError("gist_embeddings must be a list/tuple of tensors or a tensor")

    emb = emb.float().cpu()

    # Cosine similarity
    emb_norm = F.normalize(emb, p=2, dim=-1)
    cos_sim = emb_norm @ emb_norm.T  # (N, N)

    # # round to 2 decimal places
    # # sim_round = torch.round(sim_np * 100) / 100
    # torch.set_printoptions(precision=2, linewidth=1000)
    # print("sim_round\n", cos_sim)
    # breakpoint()

    # Pairwise distances
    l2_dist = torch.cdist(emb, emb, p=2)
    l1_dist = torch.cdist(emb, emb, p=1)

    cos_sim_np = cos_sim.numpy()
    l2_dist_np = l2_dist.numpy()
    l1_dist_np = l1_dist.numpy()

    # Plot side-by-side heatmaps
    figsize = (12, 4)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(cos_sim_np, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    axes[0].set_title("Cosine similarity")
    axes[0].set_xlabel("Gist index")
    axes[0].set_ylabel("Gist index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(l2_dist_np, vmin=0.0, cmap="viridis", interpolation="nearest")
    axes[1].set_title("L2 distance")
    axes[1].set_xlabel("Gist index")
    axes[1].set_ylabel("Gist index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(l1_dist_np, vmin=0.0, cmap="viridis", interpolation="nearest")
    axes[2].set_title("L1 distance")
    axes[2].set_xlabel("Gist index")
    axes[2].set_ylabel("Gist index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--output_gist_embeddings_path", type=str, default="/tmp/gist_embeddings_across_eos_num.png")

    args = parser.parse_args()

    model_name = args.model_name

    model_checkpoints = [
        "./artifacts/experiments/eos_1/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_1_Z9JQXWP2/checkpoint-9067",
        "./artifacts/experiments/eos_2/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_2_XTU300TU/checkpoint-9067",
        "./artifacts/experiments/eos_4/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_4_KS38WK9A/checkpoint-9067",
        "./artifacts/experiments/eos_8/sentence_Llama-3.2-3B_ft_4k_full_num_eos_tokens_8_XF4BDKFN/checkpoint-9067",
    ]

    with torch.no_grad():

        gist_embeddings = []
        for checkpoint in model_checkpoints:
            model, tokenizer = load_model_from_checkpoint(checkpoint)  # type: ignore[assignment]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)  # type: ignore[arg-type]
            model.eval()

            token_embeddings = model.model.embed_tokens.weight

            for eos_token_id in model.config.end_of_sentence_token_ids:
                gist_embeddings.append(token_embeddings[eos_token_id].detach())  # type: ignore[attr-defined]

            zeros_embeddings = torch.zeros(token_embeddings.shape[1], device=device)  # type: ignore[attr-defined]
            gist_embeddings.append(zeros_embeddings.detach())

        print("Found gist embeddings:", len(gist_embeddings))
        visualize_gist_embeddings(gist_embeddings, output_path=args.output_gist_embeddings_path)

        breakpoint()
