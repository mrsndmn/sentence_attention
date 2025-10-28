import ast
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_attention.evaluation.my_recall import generate_random_sample
from sentence_attention.models.checkpoint import load_model_from_checkpoint
from sentence_attention.models.sentence_llama.modeling_sentence_llama import SentenceLlamaForCausalLM
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


def _get_gist_token_ids(model: SentenceLlamaForCausalLM) -> torch.Tensor:
    eos_ids = getattr(model.config, "end_of_sentence_token_ids", None)
    if eos_ids is None or len(eos_ids) == 0:
        raise ValueError("Model config has no end_of_sentence_token_ids")
    return torch.tensor(list(eos_ids), dtype=torch.long)


def visualize_embedding_stats(
    model: SentenceLlamaForCausalLM,
    output_path: str = "/tmp/gist_embedding_stats.png",
    num_random_tokens: int = 2048,
):
    with torch.no_grad():
        embed_matrix: torch.Tensor = model.model.embed_tokens.weight.detach().float().cpu()  # type: ignore[attr-defined]

    gist_token_ids = _get_gist_token_ids(model)
    gist_emb = embed_matrix[gist_token_ids]

    vocab_size = embed_matrix.size(0)
    all_ids = torch.arange(vocab_size, dtype=torch.long)
    mask = torch.ones(vocab_size, dtype=torch.bool)
    mask[gist_token_ids] = False
    candidate_ids = all_ids[mask]
    if candidate_ids.numel() == 0:
        print("No non-gist tokens found for stats; skipping.")
        return
    num_rand = min(num_random_tokens, candidate_ids.numel())
    rand_ids = candidate_ids[torch.randperm(candidate_ids.numel())[:num_rand]]
    rand_emb = embed_matrix[rand_ids]

    gist_norms = gist_emb.norm(dim=-1).numpy()
    rand_norms = rand_emb.norm(dim=-1).numpy()

    # Pairwise cosine similarities
    def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, p=2, dim=-1)
        return x_n @ x_n.T

    gist_cos = cosine_matrix(gist_emb).cpu().numpy()
    # sample a subset of random embeddings for pairwise sims to keep it light
    rand_for_pairs = rand_emb[: min(256, rand_emb.size(0))]
    rand_cos = cosine_matrix(rand_for_pairs).cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figsize = (12, 8)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Norm hist (frequency-normalized)
    axes[0, 0].hist(
        rand_norms,
        bins=50,
        alpha=0.6,
        label="random tokens",
        weights=np.ones_like(rand_norms) / max(len(rand_norms), 1),
    )
    axes[0, 0].hist(
        gist_norms,
        bins=50,
        alpha=0.8,
        label="gist tokens",
        weights=np.ones_like(gist_norms) / max(len(gist_norms), 1),
    )
    axes[0, 0].set_title("Embedding L2 norms")
    axes[0, 0].set_xlabel("norm")
    axes[0, 0].set_ylabel("frequency")
    axes[0, 0].legend()

    # Cosine sim distributions (off-diagonal)
    def off_diag_values(mat_np):
        n = mat_np.shape[0]
        return mat_np[~torch.eye(n, dtype=torch.bool).numpy()].ravel()

    rv = off_diag_values(rand_cos)
    gv = off_diag_values(gist_cos)
    axes[0, 1].hist(
        rv,
        bins=50,
        alpha=0.6,
        label="random vs random",
        weights=np.ones_like(rv) / max(len(rv), 1),
    )
    axes[0, 1].hist(
        gv,
        bins=50,
        alpha=0.8,
        label="gist vs gist",
        weights=np.ones_like(gv) / max(len(gv), 1),
    )
    axes[0, 1].set_title("Pairwise cosine similarity (off-diagonal)")
    axes[0, 1].set_xlabel("cosine similarity")
    axes[0, 1].set_ylabel("frequency")
    axes[0, 1].legend()

    # Heatmaps
    im0 = axes[1, 0].imshow(gist_cos, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    axes[1, 0].set_title("Gist embeddings cosine (GxG)")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(rand_cos, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    axes[1, 1].set_title("Random embeddings cosine (KxK)")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved to {output_path}")


def visualize_pca_of_gists(model: SentenceLlamaForCausalLM, output_path: str = "/tmp/gist_pca.png", max_components: int = 16):
    gist_token_ids = _get_gist_token_ids(model)
    with torch.no_grad():
        emb = model.model.embed_tokens.weight.detach().float().cpu()[gist_token_ids]  # type: ignore[attr-defined]

    # Center
    emb_centered = emb - emb.mean(dim=0, keepdim=True)

    # SVD
    # emb_centered: G x D; compute top components up to max_components
    U, S, Vh = torch.linalg.svd(emb_centered, full_matrices=False)
    k = min(max_components, S.numel())
    explained_var = S**2
    explained_ratio = (explained_var / explained_var.sum()).cpu().numpy()

    # 2D projection
    PCs = Vh[:2]  # 2 x D
    proj2d = (emb_centered @ PCs.T).cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figsize = (12, 5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(range(1, k + 1), explained_ratio[:k], marker="o")
    axes[0].set_title("Scree (explained variance ratio)")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("ratio")

    axes[1].scatter(proj2d[:, 0], proj2d[:, 1], c=range(proj2d.shape[0]), cmap="viridis")
    for i, (x, y) in enumerate(proj2d):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title("Gist embeddings: first 2 PCs")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved to {output_path}")


def _find_gist_positions(input_ids: torch.Tensor, gist_token_ids: torch.Tensor) -> torch.Tensor:
    # input_ids: (B, T)
    # returns boolean mask (B, T) where positions are gist tokens
    gist_token_ids = gist_token_ids.to(input_ids.device)
    # Efficient set membership: compare against each id and OR
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for gid in gist_token_ids:
        mask |= input_ids == gid
    return mask


def analyze_layerwise_gist_hidden_states(
    hidden_states: tuple,
    input_ids: torch.Tensor,
    model: SentenceLlamaForCausalLM,
    output_prefix: str = "/tmp/gist_hidden_states",
):
    # hidden_states: tuple of (layer_0, ..., layer_L, final) or model-dependent; expect layerwise tensors (B, T, D)
    if hidden_states is None or len(hidden_states) == 0:
        print("No hidden_states provided; skipping layerwise analysis.")
        return
    # Some HF models include embeddings as layer 0; we want all tensors of shape (B, T, D)
    layers = [h for h in hidden_states if isinstance(h, torch.Tensor) and h.dim() == 3]
    if len(layers) == 0:
        print("Hidden states did not contain 3D tensors; skipping.")
        return

    gist_token_ids = _get_gist_token_ids(model)
    gist_mask = _find_gist_positions(input_ids, gist_token_ids)  # (B, T)

    if gist_mask.sum().item() == 0:
        print("No gist tokens present in inputs; skipping layerwise hidden-state analysis.")
        return

    # For each gist id, collect mean hidden state per layer
    device = layers[0].device
    gid_to_index = {int(gid): idx for idx, gid in enumerate(gist_token_ids.tolist())}
    G = len(gist_token_ids)
    L = len(layers)
    D = layers[0].size(-1)
    means = torch.zeros(L, G, D, device=device)
    counts = torch.zeros(L, G, device=device)

    for l, h in enumerate(layers):  # noqa: E741
        # h: (B, T, D)
        for gid, gidx in gid_to_index.items():
            pos_mask = input_ids == gid  # (B, T)
            if pos_mask.any():
                sel = h[pos_mask]  # (N, D)
                means[l, gidx] = sel.mean(dim=0)
                counts[l, gidx] = sel.size(0)

    # Cosine similarity per layer among gist means (GxG)
    with torch.no_grad():
        means_n = F.normalize(means, p=2, dim=-1)  # (L, G, D)
        cos = torch.einsum("lgd,lhd->lgh", means_n, means_n).detach().cpu().numpy()  # (L, G, G)

    # Plot mean off-diagonal similarity vs layer
    off_diag_means = []
    for l in range(cos.shape[0]):  # noqa: E741
        c = cos[l]
        n = c.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool).numpy()
        off_diag_means.append(c[mask].mean() if mask.any() else float("nan"))

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.plot(range(len(off_diag_means)), off_diag_means, marker="o")
    ax1.set_title("Mean off-diagonal cosine similarity of gist states vs layer")
    ax1.set_xlabel("layer index")
    ax1.set_ylabel("mean cosine")
    plt.tight_layout()
    path1 = f"{output_prefix}_offdiag_vs_layer.png"
    plt.savefig(path1, dpi=400)
    plt.close(fig1)
    print(f"Saved to {path1}")

    # Save representative layers: first, middle, last
    indices = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    for l in indices:  # noqa: E741
        fig, ax = plt.subplots(1, 1, figsize=(4 + G * 0.2, 4 + G * 0.2))
        im = ax.imshow(cos[l], vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Gist hidden-state cosine, layer {l}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        outp = f"{output_prefix}_layer{l}.png"
        plt.savefig(outp, dpi=400)
        plt.close(fig)
        print(f"Saved to {outp}")


def analyze_attention_to_gists(
    attentions: tuple,
    input_ids: torch.Tensor,
    model: SentenceLlamaForCausalLM,
    output_path: str = "/tmp/attention_to_gists.png",
    ignore_tokens_positions: List[int] | None = None,
):
    if attentions is None or len(attentions) == 0:
        print("No attentions provided; skipping attention-to-gists analysis.")
        return

    gist_token_ids = _get_gist_token_ids(model)
    gist_mask = _find_gist_positions(input_ids, gist_token_ids)  # (B, T)
    if gist_mask.sum().item() == 0:
        print("No gist tokens present in inputs; skipping attention analysis.")
        return

    # attentions: tuple of (B, H, T, T) per layer for HF causal LM (sometimes (L, B, H, T, T)); unify
    # Normalize to list of tensors shaped (B, H, T, T)
    att_list = []
    for a in attentions:
        if a.dim() == 4:
            # (B, H, T, T)
            att_list.append(a)
        elif a.dim() == 5:
            # Some models: (B, num_heads, T, T) already; or (L, B, H, T, T) in a single tensor (unlikely)
            # If it's (1, B, H, T, T), squeeze
            if a.size(0) == 1:
                att_list.append(a.squeeze(0))
            else:
                # Assume a is (B, H, T, T) stacked differently; fallback to skip
                att_list.append(a[0])
        else:
            # Unexpected shape; try best-effort squeeze
            att_list.append(a.squeeze())

    # Compute average attention mass to gist tokens among keys
    masses = []  # per layer, per head
    gist_mask_keys = gist_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)

    # Build ignore indices
    ignore_set: set[int] = set()
    T_all = gist_mask.size(-1)
    if ignore_tokens_positions:
        for p in ignore_tokens_positions:
            p_int = int(p)
            if 0 <= p_int < T_all:
                ignore_set.add(p_int)
    cols = torch.tensor(sorted(list(ignore_set)), dtype=torch.long, device=gist_mask.device) if ignore_set else None
    keep_queries = None
    if ignore_set:
        keep_queries = torch.ones(T_all, dtype=torch.bool, device=gist_mask.device)
        keep_queries[cols] = False

    for a in att_list:
        # a: (B, H, T, T)
        a = a.detach()
        if a.dim() != 4:
            a = a.view(a.size(-4), a.size(-3), a.size(-2), a.size(-1))
        B, H, Tq, Tk = a.shape
        # Zero out ignored key columns
        if cols is not None and cols.numel() > 0:
            a[:, :, :, cols] = 0.0
        # attention mass towards gist keys
        mass_to_gist = (a * gist_mask_keys.to(a.dtype)).sum(dim=-1)  # (B, H, Tq)
        # average over queries and batch (exclude ignored queries if provided)
        if keep_queries is not None and keep_queries.any():
            avg_mass = mass_to_gist[:, :, keep_queries].mean(dim=-1).mean(dim=0)  # (H,)
        else:
            avg_mass = mass_to_gist.mean(dim=-1).mean(dim=0)  # (H,)
        masses.append(avg_mass.float().cpu())

    masses_stacked = torch.stack(masses, dim=0).numpy()  # (L, H)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(masses_stacked, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_title("Average attention mass to gist tokens (layers x heads)")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved to {output_path}")


def visualize_mean_attention_map(
    attentions: tuple,
    output_path: str = "/tmp/mean_attention_map.png",
    ignore_tokens_positions: List[int] | None = None,
):
    if attentions is None or len(attentions) == 0:
        print("No attentions provided; skipping mean attention map.")
        return

    # Normalize to list of (B, H, T, T)
    att_list = []
    for a in attentions:
        if a.dim() == 4:
            att_list.append(a)
        elif a.dim() == 5:
            if a.size(0) == 1:
                att_list.append(a.squeeze(0))
            else:
                att_list.append(a[0])
        else:
            att_list.append(a.squeeze())

    layer_maps = []
    ignore_set: set[int] = set()
    for a in att_list:
        a = a.detach()
        if a.dim() != 4:
            a = a.view(a.size(-4), a.size(-3), a.size(-2), a.size(-1))
        # Mean over batch and heads -> (T, T)
        lm = a.mean(dim=(0, 1))
        if ignore_tokens_positions:
            if not ignore_set:
                T_local = lm.size(0)
                for p in ignore_tokens_positions:
                    p_int = int(p)
                    if 0 <= p_int < T_local:
                        ignore_set.add(p_int)
            if ignore_set:
                cols = torch.tensor(sorted(list(ignore_set)), dtype=torch.long, device=lm.device)
                lm[:, cols] = 0.0
                lm[cols, :] = 0.0
        layer_maps.append(lm)

    mean_map = torch.stack(layer_maps, dim=0).mean(dim=0).cpu().float().numpy()
    # Normalize per rows
    mean_map = mean_map / mean_map.sum(axis=1, keepdims=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(mean_map, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_title("Mean attention map (avg over layers, heads, batch)")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved to {output_path}")

    for layer_idx in [0, 4, 8, 12, 16, 20, 24]:
        plt.figure(figsize=(6, 5))
        layer_map = layer_maps[layer_idx].cpu().float().numpy()
        # Normalize per rows
        layer_map = layer_map / layer_map.sum(axis=1, keepdims=True)
        im = plt.imshow(
            layer_map,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            interpolation="nearest",
            aspect="auto",
        )
        plt.title(f"Mean attention map, layer {layer_idx}")
        plt.xlabel("key position")
        plt.ylabel("query position")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{output_path}_layer{layer_idx}.png", dpi=400)
        plt.close()
        print(f"Saved to {output_path}_layer{layer_idx}.png")


def visualize_per_token_mean_attention(
    attentions: tuple,
    input_ids: torch.Tensor,
    model: SentenceLlamaForCausalLM,
    output_path: str = "/tmp/per_token_attention.png",
    ignore_tokens_positions: List[int] | None = None,
):
    if attentions is None or len(attentions) == 0:
        print("No attentions provided; skipping per-token mean attention.")
        return

    # Normalize to list of (B, H, T, T)
    att_list = []
    for a in attentions:
        if a.dim() == 4:
            att_list.append(a)
        elif a.dim() == 5:
            if a.size(0) == 1:
                att_list.append(a.squeeze(0))
            else:
                att_list.append(a[0])
        else:
            att_list.append(a.squeeze())

    # Average over batch and heads -> (T, T) per layer
    layer_maps = []
    for a in att_list:
        a = a.detach()
        if a.dim() != 4:
            a = a.view(a.size(-4), a.size(-3), a.size(-2), a.size(-1))
        layer_maps.append(a.mean(dim=(0, 1)))

    # Average over layers -> (T, T)
    mean_map = torch.stack(layer_maps, dim=0).mean(dim=0)

    # Apply ignore mask for specified token positions (both query rows and key columns)
    T = mean_map.size(0)
    ignore_set: set[int] = set()
    if ignore_tokens_positions:
        for p in ignore_tokens_positions:
            p_int = int(p)
            if 0 <= p_int < T:
                ignore_set.add(p_int)
    if ignore_set:
        cols = torch.tensor(sorted(list(ignore_set)), dtype=torch.long, device=mean_map.device)
        # Zero out ignored key columns
        mean_map[:, cols] = 0.0
        # Exclude ignored queries from averaging
        keep_queries = torch.ones(T, dtype=torch.bool, device=mean_map.device)
        keep_queries[cols] = False
        if keep_queries.any():
            per_key_mass = mean_map[keep_queries].mean(dim=0)
        else:
            per_key_mass = mean_map.mean(dim=0)
    else:
        # Per-key mass: average over queries -> (T,)
        per_key_mass = mean_map.mean(dim=0)

    # Normalize to frequency (should already sum to 1, but be safe numerically)
    denom = per_key_mass.sum().clamp_min(1e-8)
    per_key_mass = (per_key_mass / denom).cpu().float().numpy()

    # Identify gist token positions
    gist_ids = _get_gist_token_ids(model).to(input_ids.device)
    gist_mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    for gid in gist_ids:
        gist_mask |= input_ids[0] == gid
    gist_mask_np = gist_mask.cpu().numpy()
    # Remove markers for ignored positions
    if ignore_set:
        for p in ignore_set:
            if 0 <= p < gist_mask_np.shape[0]:
                gist_mask_np[p] = False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})

    # Line plot with gist markers
    x = np.arange(len(per_key_mass))
    axes[0].plot(x, per_key_mass, label="per-key mean attention", color="#1f77b4")
    if gist_mask_np.any():
        axes[0].scatter(x[gist_mask_np], per_key_mass[gist_mask_np], color="red", marker="x", label="gist tokens")
    axes[0].set_title("Per-token mean normalized attention (avg over layers, heads, queries, batch)")
    axes[0].set_xlabel("token position (key)")
    axes[0].set_ylabel("frequency")
    axes[0].legend()

    # Heatmap stripe for quick overview
    stripe = per_key_mass[None, :]
    im = axes[1].imshow(
        stripe, vmin=0.0, vmax=stripe.max() if stripe.max() > 0 else 1.0, cmap="viridis", aspect="auto", interpolation="nearest"
    )
    axes[1].set_yticks([])
    axes[1].set_xlabel("token position (key)")
    axes[1].set_title("Per-token mass (normalized)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved to {output_path}")

    # Print aggregate mass on gist vs non-gist
    if gist_mask_np.any():
        gist_mass = float(per_key_mass[gist_mask_np].sum())
        non_gist_mass = float(per_key_mass[~gist_mask_np].sum())
        print(f"Per-token mass: gist={gist_mass:.4f}, non-gist={non_gist_mass:.4f}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_gist_embeddings_path", type=str, default="/tmp/gist_embeddings.png")
    parser.add_argument("--output_gist_embeddings_residuals_path", type=str, default="/tmp/gist_embeddings_residuals.png")
    parser.add_argument("--output_embedding_stats_path", type=str, default="/tmp/gist_embedding_stats.png")
    parser.add_argument("--output_pca_path", type=str, default="/tmp/gist_pca.png")
    parser.add_argument("--output_hidden_states_prefix", type=str, default="/tmp/gist_hidden_states")
    parser.add_argument("--output_attention_heatmap_path", type=str, default="/tmp/attention_to_gists.png")
    parser.add_argument("--output_mean_attention_map_path", type=str, default="/tmp/mean_attention_map.png")
    parser.add_argument("--output_per_token_attention_path", type=str, default="/tmp/per_token_attention.png")
    parser.add_argument(
        "--ignore_tokens_positions",
        type=str,
        default="",
        help="Comma-separated list or Python list of token positions to ignore (e.g., '12,24' or '[12,24]')",
    )
    parser.add_argument("--num_random_tokens_for_stats", type=int, default=2048)
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--text", type=str, default="")

    args = parser.parse_args()

    model_name = args.model_name

    model, tokenizer = load_model_from_checkpoint(model_name)  # type: ignore[assignment]
    model: SentenceLlamaForCausalLM

    model.config._attn_implementation = "sentence_attention"

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # type: ignore[arg-type]
        model.eval()

        if args.text:
            text = args.text
        else:
            text = generate_random_sample(num_examples=args.num_examples)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        # Visualize gist tokens similarities
        gist_embeddings = []
        for eos_token_id in model.config.end_of_sentence_token_ids:
            gist_embeddings.append(model.model.embed_tokens.weight[eos_token_id])  # type: ignore[attr-defined]
        print("Found gist embeddings:", len(gist_embeddings))
        visualize_gist_embeddings(gist_embeddings, output_path=args.output_gist_embeddings_path)

        gist_embeddings_residuals = []
        for i, gi in enumerate(gist_embeddings):
            for gj in gist_embeddings[i + 1 :]:
                gist_embeddings_residuals.append(gi - gj)

        visualize_gist_embeddings(gist_embeddings_residuals, output_path=args.output_gist_embeddings_residuals_path)

        # Embedding stats and PCA
        visualize_embedding_stats(
            model,
            output_path=args.output_embedding_stats_path,
            num_random_tokens=args.num_random_tokens_for_stats,
        )
        visualize_pca_of_gists(model, output_path=args.output_pca_path)

        # Layerwise hidden-state analysis
        analyze_layerwise_gist_hidden_states(
            hidden_states=outputs.hidden_states,
            input_ids=inputs["input_ids"],
            model=model,
            output_prefix=args.output_hidden_states_prefix,
        )

        # Parse ignore positions once
        ignore_positions = None
        if args.ignore_tokens_positions:
            s = args.ignore_tokens_positions.strip()
            try:
                if s.startswith("[") and s.endswith("]"):
                    ignore_positions = [int(x) for x in ast.literal_eval(s)]
                else:
                    ignore_positions = [int(x) for x in s.split(",") if x.strip()]
            except Exception:
                print(f"Failed to parse --ignore_tokens_positions='{args.ignore_tokens_positions}', ignoring.")
                ignore_positions = None

        # Attention-to-gists analysis
        analyze_attention_to_gists(
            attentions=outputs.attentions,
            input_ids=inputs["input_ids"],
            model=model,
            output_path=args.output_attention_heatmap_path,
            ignore_tokens_positions=ignore_positions,
        )

        # Mean attention map
        visualize_mean_attention_map(
            attentions=outputs.attentions,
            output_path=args.output_mean_attention_map_path,
            ignore_tokens_positions=ignore_positions,
        )

        # Per-token mean normalized attention
        visualize_per_token_mean_attention(
            attentions=outputs.attentions,
            input_ids=inputs["input_ids"],
            model=model,
            output_path=args.output_per_token_attention_path,
            ignore_tokens_positions=ignore_positions,
        )

        breakpoint()
