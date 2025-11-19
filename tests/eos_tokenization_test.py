from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS


def test_sentence_placement_inserts_at_sentence_end():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence",
    )

    text = "Hello world."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])  # includes added special tokens as text strings

    assert decoded.count("<end_of_sentence_") == 2, "two gist tokens appended after sentence end"
    assert decoded.count("<end_of_sentence_0>") == 1
    assert decoded.count("<end_of_sentence_1>") == 1


def test_uniform_placement_inserts_without_sentence_end_and_aligns_masks():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="uniform",
        uniform_interval_tokens=2,
    )

    # No punctuation; gist tokens should still appear due to uniform insertion
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    encoding = tokenizer.encode_plus(text, return_special_tokens_mask=True)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    special_tokens_mask = encoding["special_tokens_mask"]

    assert len(input_ids) == len(attention_mask)
    assert len(special_tokens_mask) == len(input_ids)

    decoded = tokenizer.decode(input_ids)

    # Compute exact expected number of uniform insertions: floor(N / interval) blocks
    baseline_tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence",  # no punctuation -> no sentence-end insertion
    )
    baseline_ids = baseline_tokenizer.encode_plus(text)["input_ids"]
    interval = 2
    expected_blocks = len(baseline_ids) // interval

    # Each block inserts num_eos_tokens occurrences per token id
    assert decoded.count("<end_of_sentence_0>") == expected_blocks
    assert decoded.count("<end_of_sentence_1>") == expected_blocks
    assert decoded.count("<end_of_sentence_") == expected_blocks * 2

    gist_ids = set(tokenizer.end_of_sentence_token_ids)
    # Every gist token id must have special_tokens_mask == 1
    for idx, tok_id in enumerate(input_ids):
        if tok_id in gist_ids:
            assert special_tokens_mask[idx] == 1


def test_uniform_placement_does_not_add_at_sentence_boundaries_when_interval_large():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=1,
        gist_placement="uniform",
        uniform_interval_tokens=1_000_000,
    )

    text = "Short sentence with punctuation."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])  # should not include gist tokens due to large interval

    assert decoded.count("<end_of_sentence_0>") == 0


def test_batch_uniform_placement_shapes():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=1,
        gist_placement="uniform",
        uniform_interval_tokens=3,
    )

    batch = [
        "one two three four five six seven eight nine ten",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
    ]

    enc = tokenizer.batch_encode_plus(batch, return_special_tokens_mask=True)

    assert len(enc["input_ids"]) == len(batch)
    assert len(enc["attention_mask"]) == len(batch)
    assert len(enc["special_tokens_mask"]) == len(batch)

    for i in range(len(batch)):
        assert len(enc["input_ids"][i]) == len(enc["attention_mask"][i])
        assert len(enc["input_ids"][i]) == len(enc["special_tokens_mask"][i])


# ------------------------
# Qwen2 tokenizer variants
# ------------------------


def test_qwen_sentence_placement_inserts_at_sentence_end():
    tokenizer = Qwen2TokenizerFastEOS.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_eos_tokens=2,
        gist_placement="sentence",
    )

    text = "Hello world."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    assert decoded.count("<end_of_sentence_") == 2, "two gist tokens appended after sentence end"
    assert decoded.count("<end_of_sentence_0>") == 1
    assert decoded.count("<end_of_sentence_1>") == 1


def test_qwen_uniform_placement_exact_counts_and_aligns_masks():
    tokenizer = Qwen2TokenizerFastEOS.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_eos_tokens=2,
        gist_placement="uniform",
        uniform_interval_tokens=2,
    )

    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    encoding = tokenizer.encode_plus(text, return_special_tokens_mask=True)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    special_tokens_mask = encoding["special_tokens_mask"]

    assert len(input_ids) == len(attention_mask)
    assert len(special_tokens_mask) == len(input_ids)

    decoded = tokenizer.decode(input_ids)

    # Baseline with sentence placement; no punctuation means no sentence-end insertion.
    baseline_tokenizer = Qwen2TokenizerFastEOS.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_eos_tokens=2,
        gist_placement="sentence",
    )
    baseline_ids = baseline_tokenizer.encode_plus(text)["input_ids"]
    interval = 2
    expected_blocks = len(baseline_ids) // interval

    assert decoded.count("<end_of_sentence_0>") == expected_blocks
    assert decoded.count("<end_of_sentence_1>") == expected_blocks
    assert decoded.count("<end_of_sentence_") == expected_blocks * 2

    gist_ids = set(tokenizer.end_of_sentence_token_ids)
    for idx, tok_id in enumerate(input_ids):
        if tok_id in gist_ids:
            assert special_tokens_mask[idx] == 1


def test_qwen_uniform_placement_no_sentence_boundary_insertion_when_interval_large():
    tokenizer = Qwen2TokenizerFastEOS.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_eos_tokens=1,
        gist_placement="uniform",
        uniform_interval_tokens=1_000_000,
    )

    text = "Short sentence with punctuation."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    assert decoded.count("<end_of_sentence_0>") == 0


def test_qwen_batch_uniform_placement_shapes():
    tokenizer = Qwen2TokenizerFastEOS.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_eos_tokens=1,
        gist_placement="uniform",
        uniform_interval_tokens=3,
    )

    batch = [
        "one two three four five six seven eight nine ten",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
    ]

    enc = tokenizer.batch_encode_plus(batch, return_special_tokens_mask=True)

    assert len(enc["input_ids"]) == len(batch)
    assert len(enc["attention_mask"]) == len(batch)
    assert len(enc["special_tokens_mask"]) == len(batch)

    for i in range(len(batch)):
        assert len(enc["input_ids"][i]) == len(enc["attention_mask"][i])
        assert len(enc["input_ids"][i]) == len(enc["special_tokens_mask"][i])
