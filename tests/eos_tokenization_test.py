import pytest

from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS


@pytest.fixture(
    params=[
        (PreTrainedTokenizerFastEOS, "unsloth/Llama-3.2-1B"),
        (Qwen2TokenizerFastEOS, "Qwen/Qwen2.5-1.5B"),
        (GPT2TokenizerFastEOS, "HuggingFaceTB/SmolLM2-135M"),
    ]
)
def tokenizer_spec(request):
    return request.param


def _build_tokenizer(tokenizer_spec, **kwargs):
    tokenizer_cls, model_name = tokenizer_spec
    return tokenizer_cls.from_pretrained(model_name, **kwargs)


def test_sentence_placement_inserts_at_sentence_end(tokenizer_spec):
    tokenizer = _build_tokenizer(tokenizer_spec, num_eos_tokens=2, gist_placement="sentence")

    text = "Hello world."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])  # includes added special tokens as text strings

    assert decoded.count("<end_of_sentence_") == 2
    assert decoded.count("<end_of_sentence_0>") == 1
    assert decoded.count("<end_of_sentence_1>") == 1


def test_uniform_placement_inserts_without_sentence_end_and_aligns_masks(tokenizer_spec):
    tokenizer = _build_tokenizer(tokenizer_spec, num_eos_tokens=2, gist_placement="uniform", uniform_interval_tokens=2)

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
    baseline_tokenizer = _build_tokenizer(tokenizer_spec, num_eos_tokens=2, gist_placement="sentence")
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


def test_uniform_placement_does_not_add_at_sentence_boundaries_when_interval_large(tokenizer_spec):
    tokenizer = _build_tokenizer(tokenizer_spec, num_eos_tokens=1, gist_placement="uniform", uniform_interval_tokens=1_000_000)

    text = "Short sentence with punctuation."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])  # should not include gist tokens due to large interval

    assert decoded.count("<end_of_sentence_0>") == 0


def test_batch_uniform_placement_shapes(tokenizer_spec):
    tokenizer = _build_tokenizer(tokenizer_spec, num_eos_tokens=1, gist_placement="uniform", uniform_interval_tokens=3)

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


# Tests for PreTrainedTokenizerFastEOS with new gist_sets and gist_insertion_strategy arguments
def test_gist_sets_single_mode():
    """Test that 'single' mode initializes num_eos_tokens tokens."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=3,
        gist_placement="sentence",
        gist_sets="single",
    )

    assert len(tokenizer.end_of_sentence_tokens_list) == 3
    assert tokenizer.end_of_sentence_tokens_list == [
        "<end_of_sentence_0>",
        "<end_of_sentence_1>",
        "<end_of_sentence_2>",
    ]
    assert tokenizer.gist_token_sets is None


def test_gist_sets_roll_mode():
    """Test that 'roll' mode creates sets for all sizes from 0 to num_eos_tokens-1."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=3,
        gist_placement="sentence",
        gist_sets="roll",
    )

    # Should have sets: [0], [0,1], [0,1,2] -> total 6 tokens
    expected_tokens = [
        "<end_of_sentence_0_0>",
        "<end_of_sentence_0_1>",
        "<end_of_sentence_1_1>",
        "<end_of_sentence_0_2>",
        "<end_of_sentence_1_2>",
        "<end_of_sentence_2_2>",
    ]
    assert len(tokenizer.end_of_sentence_tokens_list) == 6
    assert set(tokenizer.end_of_sentence_tokens_list) == set(expected_tokens)

    # Check that gist_token_sets is organized correctly
    assert tokenizer.gist_token_sets is not None
    assert len(tokenizer.gist_token_sets) == 3
    assert tokenizer.gist_token_sets[0] == ["<end_of_sentence_0_0>"]
    assert tokenizer.gist_token_sets[1] == ["<end_of_sentence_0_1>", "<end_of_sentence_1_1>"]
    assert tokenizer.gist_token_sets[2] == [
        "<end_of_sentence_0_2>",
        "<end_of_sentence_1_2>",
        "<end_of_sentence_2_2>",
    ]


def test_gist_sets_swarm_mode():
    """Test that 'swarm' mode creates num_gist_sets_in_swarm sets, each with num_eos_tokens tokens."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence",
        gist_sets="swarm",
        num_gist_sets_in_swarm=3,
    )

    # Should have 3 sets * 2 tokens = 6 tokens total
    expected_tokens = [
        "<end_of_sentence_0_swarm_0>",
        "<end_of_sentence_1_swarm_0>",
        "<end_of_sentence_0_swarm_1>",
        "<end_of_sentence_1_swarm_1>",
        "<end_of_sentence_0_swarm_2>",
        "<end_of_sentence_1_swarm_2>",
    ]
    assert len(tokenizer.end_of_sentence_tokens_list) == 6
    assert set(tokenizer.end_of_sentence_tokens_list) == set(expected_tokens)

    # Check that gist_token_sets is organized correctly
    assert tokenizer.gist_token_sets is not None
    assert len(tokenizer.gist_token_sets) == 3
    assert tokenizer.gist_token_sets[0] == [
        "<end_of_sentence_0_swarm_0>",
        "<end_of_sentence_1_swarm_0>",
    ]
    assert tokenizer.gist_token_sets[1] == [
        "<end_of_sentence_0_swarm_1>",
        "<end_of_sentence_1_swarm_1>",
    ]
    assert tokenizer.gist_token_sets[2] == [
        "<end_of_sentence_0_swarm_2>",
        "<end_of_sentence_1_swarm_2>",
    ]


def test_gist_insertion_strategy_default():
    """Test that 'default' strategy inserts all tokens from end_of_sentence_tokens_list."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence_nltk",
        gist_sets="single",
        gist_insertion_strategy="default",
    )

    text = "First sentence. Second sentence. Third sentence."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    # Should have 2 tokens per sentence * 3 sentences = 6 occurrences
    assert decoded.count("<end_of_sentence_0>") == 3
    assert decoded.count("<end_of_sentence_1>") == 3


def test_gist_insertion_strategy_random_with_swarm():
    """Test that 'random' strategy selects random sets from swarm mode."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence_nltk",
        gist_sets="swarm",
        num_gist_sets_in_swarm=3,
        gist_insertion_strategy="random",
    )

    text = "First sentence. Second sentence."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    # Should have tokens from swarm sets (2 tokens per sentence)
    # Check that we have exactly 4 tokens total (2 sentences * 2 tokens each)
    import re

    swarm_matches = re.findall(r"_swarm_(\d+)", decoded)
    assert len(swarm_matches) == 4

    # Each sentence should use a complete set (all tokens from same swarm index)
    # Extract tokens per sentence by finding patterns
    tokens = re.findall(r"<end_of_sentence_\d+_swarm_(\d+)>", decoded)
    assert len(tokens) == 4
    # First two tokens should be from same set, last two from same set (may be different)
    assert tokens[0] == tokens[1]  # First sentence uses same set
    assert tokens[2] == tokens[3]  # Second sentence uses same set


def test_gist_insertion_strategy_random_with_roll():
    """Test that 'random' strategy selects random sets from roll mode."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=3,
        gist_placement="sentence_nltk",
        gist_sets="roll",
        gist_insertion_strategy="random",
    )

    text = "First sentence. Second sentence."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    # Should have tokens from roll sets
    # Each set has different number of tokens: 1, 2, or 3
    import re

    # Extract all tokens matching the roll pattern
    tokens = re.findall(r"<end_of_sentence_(\d+)_(\d+)>", decoded)
    assert len(tokens) > 0

    # Verify tokens follow correct pattern (gist_i_set_i where set_i is 0, 1, or 2)
    set_indices = [t[1] for t in tokens]
    assert all(idx in ["0", "1", "2"] for idx in set_indices)

    # Verify that tokens come in complete sets per sentence
    # Each sentence should have tokens with the same set index
    # We can verify by checking that consecutive tokens from same sentence have same set index
    # Since we have 2 sentences, we expect at least 2 tokens total
    assert len(tokens) >= 2

    print("decoded", decoded)


def test_gist_insertion_strategy_random_validation():
    """Test that 'random' strategy validation works correctly."""
    # Should fail with non-sentence_nltk placement
    with pytest.raises(
        ValueError, match="gist_insertion_strategy='random' can only be used with gist_placement='sentence_nltk'"
    ):
        PreTrainedTokenizerFastEOS.from_pretrained(
            "unsloth/Llama-3.2-1B",
            num_eos_tokens=2,
            gist_placement="sentence",
            gist_sets="swarm",
            gist_insertion_strategy="random",
        )

    # Should fail with single mode
    with pytest.raises(ValueError, match="gist_insertion_strategy='random' can only be used with gist_sets in"):
        PreTrainedTokenizerFastEOS.from_pretrained(
            "unsloth/Llama-3.2-1B",
            num_eos_tokens=2,
            gist_placement="sentence_nltk",
            gist_sets="single",
            gist_insertion_strategy="random",
        )


def test_gist_sets_validation():
    """Test that gist_sets validation works correctly."""
    with pytest.raises(ValueError, match="gist_sets must be one of"):
        PreTrainedTokenizerFastEOS.from_pretrained(
            "unsloth/Llama-3.2-1B",
            num_eos_tokens=2,
            gist_sets="invalid",
        )


def test_gist_insertion_strategy_validation():
    """Test that gist_insertion_strategy validation works correctly."""
    with pytest.raises(ValueError, match="gist_insertion_strategy must be one of"):
        PreTrainedTokenizerFastEOS.from_pretrained(
            "unsloth/Llama-3.2-1B",
            num_eos_tokens=2,
            gist_insertion_strategy="invalid",
        )


def test_default_gist_sets_behavior():
    """Test that default gist_sets='single' maintains backward compatibility."""
    tokenizer1 = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence",
    )

    tokenizer2 = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence",
        gist_sets="single",
    )

    assert tokenizer1.end_of_sentence_tokens_list == tokenizer2.end_of_sentence_tokens_list
    assert tokenizer1.end_of_sentence_tokens_list == [
        "<end_of_sentence_0>",
        "<end_of_sentence_1>",
    ]


def test_random_strategy_uses_full_sets():
    """Test that random strategy uses full sets (all tokens from selected set) per sentence."""
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained(
        "unsloth/Llama-3.2-1B",
        num_eos_tokens=2,
        gist_placement="sentence_nltk",
        gist_sets="swarm",
        num_gist_sets_in_swarm=10,
        gist_insertion_strategy="random",
    )

    text = "Sentence one. Sentence two. Sentence three."
    encoding = tokenizer.encode_plus(text)
    decoded = tokenizer.decode(encoding["input_ids"])

    # Extract all swarm indices
    import re

    swarm_matches = re.findall(r"_swarm_(\d+)", decoded)
    assert len(swarm_matches) == 6  # 3 sentences * 2 tokens each

    # Verify format is correct (should be 0, 1, or 2)
    assert all(match in list(map(str, range(10))) for match in swarm_matches)

    # Verify that each sentence uses a complete set
    # Group tokens by sentence (roughly by position in decoded string)
    tokens = re.findall(r"<end_of_sentence_(\d+)_swarm_(\d+)>", decoded)
    # Each sentence should have 2 tokens with same swarm index
    # First sentence: tokens[0] and tokens[1] should have same swarm index
    # Second sentence: tokens[2] and tokens[3] should have same swarm index
    # Third sentence: tokens[4] and tokens[5] should have same swarm index
    assert tokens[0][1] == tokens[1][1]  # First sentence
    assert tokens[2][1] == tokens[3][1]  # Second sentence
    assert tokens[4][1] == tokens[5][1]  # Third sentence

    print("decoded", decoded)
