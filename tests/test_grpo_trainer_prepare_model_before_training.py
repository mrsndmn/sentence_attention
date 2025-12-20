import pytest

from sentence_attention.trainer.grpo_trainer import GRPOTrainer

import torch
import torch.nn as nn


class FakeTokenizer:
    def __init__(self, vocab, num_eos_tokens: int):
        self._vocab = dict(vocab)
        self.num_eos_tokens = num_eos_tokens
        self.pad_token_id = self._vocab.get("<pad>", 0)

    def add_special_tokens(self, special_tokens_dict):
        additional = special_tokens_dict.get("additional_special_tokens", [])
        for tok in additional:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token)

    def __len__(self):
        return len(self._vocab)


class FakeConfig:
    def __init__(self, tie_word_embeddings: bool):
        self.tie_word_embeddings = tie_word_embeddings


class FakeModel:
    def __init__(self, vocab_size: int, hidden_size: int, tie_word_embeddings: bool):
        self.config = FakeConfig(tie_word_embeddings=tie_word_embeddings)
        self._hidden_size = hidden_size

        self._input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=True)

        if tie_word_embeddings:
            # Share parameters exactly like tied embeddings models do.
            self._lm_head.weight = self._input_embeddings.weight

    def get_input_embeddings(self):
        return self._input_embeddings

    def get_output_embeddings(self):
        return self._lm_head

    def resize_token_embeddings(self, new_num_tokens: int, mean_resizing: bool = False):
        old_input = self._input_embeddings
        old_vocab_size = old_input.num_embeddings

        new_input = nn.Embedding(new_num_tokens, self._hidden_size)
        with torch.no_grad():
            new_input.weight[:old_vocab_size].copy_(old_input.weight)
        self._input_embeddings = new_input

        old_lm_head = self._lm_head
        new_lm_head = nn.Linear(self._hidden_size, new_num_tokens, bias=True)
        with torch.no_grad():
            rows_to_copy = min(old_vocab_size, new_num_tokens)
            new_lm_head.weight[:rows_to_copy].copy_(old_lm_head.weight[:rows_to_copy])
            new_lm_head.bias[:rows_to_copy].copy_(old_lm_head.bias[:rows_to_copy])
        self._lm_head = new_lm_head

        if self.config.tie_word_embeddings:
            self._lm_head.weight = self._input_embeddings.weight

        return self._input_embeddings


def _make_base_vocab(num_eos_tokens: int):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for i in range(num_eos_tokens):
        vocab[f"<end_of_sentence_{i}>"] = len(vocab)
    vocab["hello"] = len(vocab)
    vocab["world"] = len(vocab)
    return vocab


def _expected_new_tokens(num_eos_tokens: int):
    out = []
    for num_gist_tokens in range(1, num_eos_tokens + 1):
        for i in range(num_gist_tokens):
            out.append((f"<end_of_sentence_{i}_{num_gist_tokens}>", f"<end_of_sentence_{i}>"))
    return out


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_prepare_model_before_training_copies_embeddings_for_new_gist_tokens(tie_word_embeddings: bool):
    torch.manual_seed(0)

    num_eos_tokens = 2
    hidden_size = 8
    vocab = _make_base_vocab(num_eos_tokens)
    tokenizer = FakeTokenizer(vocab=vocab, num_eos_tokens=num_eos_tokens)
    model = FakeModel(vocab_size=len(tokenizer), hidden_size=hidden_size, tie_word_embeddings=tie_word_embeddings)

    with torch.no_grad():
        # Make all rows unique and deterministic for robust equality checks.
        w_in = model.get_input_embeddings().weight
        w_in.copy_(torch.arange(w_in.numel(), dtype=w_in.dtype).reshape_as(w_in))

        if not tie_word_embeddings:
            w_out = model.get_output_embeddings().weight
            b_out = model.get_output_embeddings().bias
            w_out.copy_(1000 + torch.arange(w_out.numel(), dtype=w_out.dtype).reshape_as(w_out))
            b_out.copy_(2000 + torch.arange(b_out.numel(), dtype=b_out.dtype))

    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.prepare_model_before_training(model, tokenizer)

    # Ensure tokenizer got the new tokens.
    for new_tok, _orig_tok in _expected_new_tokens(num_eos_tokens):
        assert tokenizer.convert_tokens_to_ids(new_tok) is not None

    # Ensure embeddings are copied from original gist tokens.
    for new_tok, orig_tok in _expected_new_tokens(num_eos_tokens):
        new_id = tokenizer.convert_tokens_to_ids(new_tok)
        orig_id = tokenizer.convert_tokens_to_ids(orig_tok)

        assert torch.equal(
            model.get_input_embeddings().weight[new_id],
            model.get_input_embeddings().weight[orig_id],
        )

        if tie_word_embeddings:
            assert torch.equal(
                model.get_output_embeddings().weight[new_id],
                model.get_input_embeddings().weight[new_id],
            )
        else:
            assert torch.equal(
                model.get_output_embeddings().weight[new_id],
                model.get_output_embeddings().weight[orig_id],
            )
            assert torch.equal(
                model.get_output_embeddings().bias[new_id],
                model.get_output_embeddings().bias[orig_id],
            )


def test_sample_gist_tokens_for_input_replaces_runs_and_updates_masks():
    torch.manual_seed(0)

    num_eos_tokens = 3
    hidden_size = 4
    vocab = _make_base_vocab(num_eos_tokens)
    tokenizer = FakeTokenizer(vocab=vocab, num_eos_tokens=num_eos_tokens)

    # Add all variable-length gist tokens (like prepare_model_before_training does).
    additional = []
    for k in range(1, num_eos_tokens + 1):
        for i in range(k):
            additional.append(f"<end_of_sentence_{i}_{k}>")
    tokenizer.add_special_tokens({"additional_special_tokens": additional})

    model = FakeModel(vocab_size=len(tokenizer), hidden_size=hidden_size, tie_word_embeddings=True)

    # Build a single example with two gist runs of length 3, right padded.
    base_ids = [
        tokenizer.convert_tokens_to_ids("hello"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_0>"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_1>"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_2>"),
        tokenizer.convert_tokens_to_ids("world"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_0>"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_1>"),
        tokenizer.convert_tokens_to_ids("<end_of_sentence_2>"),
    ]
    seq_len = 12
    pad_id = tokenizer.pad_token_id
    base_ids = base_ids + [pad_id] * (seq_len - len(base_ids))

    input_ids = torch.tensor([base_ids], dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    special_embeddings_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for pos in [1, 2, 3, 5, 6, 7]:
        special_embeddings_mask[0, pos] = 1

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "special_embeddings_mask": special_embeddings_mask,
    }

    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.processing_class = tokenizer

    class Args:
        grpo_num_generate_samples = 5

    trainer.args = Args()

    sampled = trainer.sample_gist_tokens_for_input(model, inputs)
    assert isinstance(sampled, list)
    assert len(sampled) == 5

    for out in sampled:
        assert out["input_ids"].shape == input_ids.shape
        assert out["attention_mask"].shape == attention_mask.shape
        assert out["special_embeddings_mask"].shape == special_embeddings_mask.shape
        assert out["clothest_end_of_sentence_token_idx"].shape == input_ids.shape

        # Masks must be consistent with padding.
        assert torch.equal(out["attention_mask"], (out["input_ids"] != pad_id).long())
        assert (out["special_embeddings_mask"] <= out["attention_mask"]).all()

        # Each original run must be replaced by a run starting with <end_of_sentence_0_k>.
        out_ids = out["input_ids"][0].tolist()
        out_spec = out["special_embeddings_mask"][0].tolist()

        # Find all contiguous gist runs in the output (within attention_mask).
        runs = []
        i = 0
        valid_len = int(out["attention_mask"][0].sum().item())
        while i < valid_len:
            if out_spec[i] == 1:
                j = i
                while j < valid_len and out_spec[j] == 1:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1

        # We expect exactly two gist runs (same as input), but each can have variable length 1..3.
        assert len(runs) == 2
        for start, end in runs:
            run_len = end - start
            assert 1 <= run_len <= num_eos_tokens
            # First token must be <end_of_sentence_0_{run_len}>
            first_expected = tokenizer.convert_tokens_to_ids(f"<end_of_sentence_0_{run_len}>")
            assert out_ids[start] == first_expected
