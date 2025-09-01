import shutil

from datasets import Dataset
from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS

ARTIFACTS_PREFIX = "/workspace-SR004.nfs2/d.tarasov/sentence_attention/artifacts"


def _test_tokenized_fineweb(tokenizer):
    dataset = Dataset.load_from_disk(f"{ARTIFACTS_PREFIX}/data/fineweb_edu_tokenized_Llama-3.2-1B_with_eos_token_num_1_merged/")
    item = dataset[0]
    decoded = tokenizer.decode(item["input_ids"])
    assert decoded.count("<end_of_sentence>") == 32, "decoded content has 32 eos tokens"


def test_tokenized_fineweb():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained("unsloth/Llama-3.2-1B")
    _test_tokenized_fineweb(tokenizer)

    shutil.rmtree("/tmp/eos_tokenizer", ignore_errors=True)
    tokenizer_path = "/tmp/eos_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_loaded = PreTrainedTokenizerFastEOS.from_pretrained(tokenizer_path)
    assert tokenizer_loaded.num_eos_tokens == 1, "reloaded tokenizer has 1 eos token"
    _test_tokenized_fineweb(tokenizer_loaded)


def _test_tokenized_fineweb_4_eos_tokens(tokenizer):
    dataset = Dataset.load_from_disk(f"{ARTIFACTS_PREFIX}/data/fineweb_edu_tokenized_Llama-3.2-1B_with_eos_token_num_4_merged/")
    item = dataset[0]

    decoded = tokenizer.decode(item["input_ids"])
    assert decoded.count("<end_of_sentence_") == 32 * 4, "decoded content has 32 eos tokens"

    for i in range(4):
        assert decoded.count(f"<end_of_sentence_{i}>") == 32, f"decoded content has 32 eos tokens for {i}"


def test_tokenized_fineweb_4_eos_tokens():
    tokenizer = PreTrainedTokenizerFastEOS.from_pretrained("unsloth/Llama-3.2-1B", num_eos_tokens=4)
    _test_tokenized_fineweb_4_eos_tokens(tokenizer)

    shutil.rmtree("/tmp/eos_tokenizer_4", ignore_errors=True)
    tokenizer_path = "/tmp/eos_tokenizer_4"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_loaded = PreTrainedTokenizerFastEOS.from_pretrained(tokenizer_path)
    assert tokenizer_loaded.num_eos_tokens == 4, "reloaded tokenizer has 4 eos tokens"
    _test_tokenized_fineweb_4_eos_tokens(tokenizer_loaded)

    assert "<end_of_sentence>" not in tokenizer.vocab
