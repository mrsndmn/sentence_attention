import os
import torch
import torch.nn as nn
import pytest
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
            "The cat sat on the mat.",
            "Then it slept on a sunny porch!",
            "Finally, it woke up?",
            "The cat sat on the mat",
            "Then it slept on a sunny porch!",
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
        f"{RED}{tok} [{tid}]{RESET}" if bool(mask.item()) else f"{tok} [{tid}]"
        for tok, mask, tid in zip(tokens, boundary_mask[0], tokenized)
    ]
    # pretty_tokens = [f"{RED}{tok}{RESET}" if bool(mask.item()) else tok for tok, mask in zip(tokens, boundary_mask[0])]
    print("Tokens:", " | ".join(pretty_tokens))

    # for pos in boundary_positions:
    #     assert boundary_mask[0, pos].item(), f"Expected boundary at position {pos} token {tokens[pos]}"

    breakpoint()
