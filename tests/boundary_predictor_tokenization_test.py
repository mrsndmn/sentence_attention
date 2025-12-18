import os

import torch
import torch.nn as nn
import pytest
from transformers import AutoModel, AutoTokenizer

from sentence_attention.boundaries.predictor import BolmoConfig, BolmoLocalEncoder
from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS


CKPT_PATH = "artifacts/experiments/bp/unsloth__Llama-3.2-1B/model.pt"
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B"


def test_boundary_predictor_inserts_gist_tokens_after_boundaries():
    class DummyBoundaryPredictor(torch.nn.Module):
        def forward(self, input_ids):
            # Mark boundaries after original positions 1 and 3
            seq_len = input_ids.shape[1]
            boundary_mask = torch.zeros((1, seq_len), dtype=torch.long, device=input_ids.device)
            boundary_mask[0, 1] = 1
            boundary_mask[0, 3] = 1
            boundary_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
            return boundary_logprobs, boundary_mask

    tokenizer = object.__new__(PreTrainedTokenizerFastEOS)
    tokenizer.end_of_sentence_token_ids = [101, 102]
    tokenizer.boundary_predictor_model = DummyBoundaryPredictor()
    tokenizer.boundary_predictor_device = "cpu"

    encoding = {
        "input_ids": [10, 11, 12, 13],
        "attention_mask": [1, 1, 1, 1],
        "special_tokens_mask": [0, 0, 0, 0],
        "offset_mapping": [(0, 1), (2, 3), (4, 5), (6, 7)],
    }

    updated = tokenizer._insert_boundary_predictor_gist_into_encoding(encoding)

    assert updated["input_ids"] == [10, 11, 101, 102, 12, 13, 101, 102]
    assert updated["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1]
    assert updated["special_tokens_mask"] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert updated["offset_mapping"] == [
        (0, 1),
        (2, 3),
        None,
        None,
        (4, 5),
        (6, 7),
        None,
        None,
    ]


def test_boundary_predictor_honors_max_length():
    class DummyBoundaryPredictor(nn.Module):
        def forward(self, input_ids):
            boundary_mask = torch.zeros((1, input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
            boundary_mask[0, 1] = 1
            boundary_mask[0, 3] = 1
            boundary_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
            return boundary_logprobs, boundary_mask

    tokenizer = object.__new__(PreTrainedTokenizerFastEOS)
    tokenizer.end_of_sentence_token_ids = [101, 102]
    tokenizer.boundary_predictor_model = DummyBoundaryPredictor()
    tokenizer.boundary_predictor_device = "cpu"

    encoding = {
        "input_ids": [10, 11, 12, 13],
        "attention_mask": [1, 1, 1, 1],
        "special_tokens_mask": [0, 0, 0, 0],
        "offset_mapping": [(0, 1), (2, 3), (4, 5), (6, 7)],
    }

    updated = tokenizer._insert_boundary_predictor_gist_into_encoding(encoding, max_length=5)

    assert updated["input_ids"] == [10, 11, 101, 102, 12]
    assert updated["attention_mask"] == [1, 1, 1, 1, 1]
    assert updated["special_tokens_mask"] == [0, 0, 1, 1, 0]
    assert updated["offset_mapping"] == [(0, 1), (2, 3), None, None, (4, 5)]
    assert len(updated["input_ids"]) == 5


@pytest.mark.skipif(
    not os.path.exists(CKPT_PATH) or not torch.cuda.is_available(),
    reason="checkpoint missing or CUDA unavailable",
)
def test_boundary_predictor_integration_with_real_model():
    device = "cuda"

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    config = BolmoConfig(**ckpt["config"])

    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModel.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    base_model.to(device)

    base_embedding_weight = base_model.get_input_embeddings().weight.detach().clone()
    del base_model
    torch.cuda.empty_cache()

    embedding = nn.Embedding.from_pretrained(base_embedding_weight, freeze=True).to(device)
    bp_model = BolmoLocalEncoder(config, embedding)
    bp_model.load_state_dict(sd)
    bp_model.to(device)
    bp_model.to(torch.bfloat16)
    bp_model.eval()

    eos_tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        BASE_MODEL_NAME,
        num_eos_tokens=2,
        gist_placement="boundary_predictor",
        boundary_predictor_model=bp_model,
        boundary_predictor_device=device,
    )

    text = "The quick brown fox jumps over the lazy dog. Another line follows. Mr. Test tested test."

    base_encoding = tokenizer_base.encode_plus(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    input_ids_tensor = torch.tensor([base_encoding["input_ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, boundary_mask = bp_model(input_ids_tensor)
    boundary_mask[0, 0] = 0
    boundary_positions = boundary_mask[0].nonzero(as_tuple=True)[0].tolist()

    gist_ids = eos_tokenizer.end_of_sentence_token_ids
    boundary_set = set(boundary_positions)

    expected_input_ids = []
    expected_attention_mask = []
    expected_special_tokens_mask = []
    expected_offset_mapping = []

    for idx, token_id in enumerate(base_encoding["input_ids"]):
        expected_input_ids.append(token_id)
        expected_attention_mask.append(base_encoding["attention_mask"][idx])
        expected_special_tokens_mask.append(base_encoding["special_tokens_mask"][idx])
        expected_offset_mapping.append(base_encoding["offset_mapping"][idx])

        if idx in boundary_set:
            expected_input_ids.extend(gist_ids)
            expected_attention_mask.extend([1] * len(gist_ids))
            expected_special_tokens_mask.extend([1] * len(gist_ids))
            expected_offset_mapping.extend([None] * len(gist_ids))

    expected = {
        "input_ids": expected_input_ids,
        "attention_mask": expected_attention_mask,
        "special_tokens_mask": expected_special_tokens_mask,
        "offset_mapping": expected_offset_mapping,
    }

    actual = eos_tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    assert actual["input_ids"] == expected["input_ids"]
    assert actual["attention_mask"] == expected["attention_mask"]
    assert actual["special_tokens_mask"] == expected["special_tokens_mask"]
    assert actual["offset_mapping"] == expected["offset_mapping"]

    breakpoint()
