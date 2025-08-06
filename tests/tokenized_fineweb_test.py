
from datasets import Dataset
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
import shutil

def _test_tokenized_fineweb(tokenizer):
    dataset = Dataset.load_from_disk("artifacts/data/fineweb_edu_tokenized_SmolLM2-1.7B_with_eos_token/shard_0/")
    item = dataset[0]
    decoded = tokenizer.decode(item['input_ids'])
    assert decoded.count('<end_of_sentence>') == 32, 'decoded content has 32 eos tokens'

def test_tokenized_fineweb():
    tokenizer = GPT2TokenizerFastEOS.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    _test_tokenized_fineweb(tokenizer)

    shutil.rmtree("/tmp/eos_tokenizer", ignore_errors=True)
    tokenizer_path = "/tmp/eos_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_loaded = GPT2TokenizerFastEOS.from_pretrained(tokenizer_path)
    assert tokenizer_loaded.num_eos_tokens == 1, 'reloaded tokenizer has 1 eos token'
    _test_tokenized_fineweb(tokenizer_loaded)

def _test_tokenized_fineweb_4_eos_tokens(tokenizer):
    dataset = Dataset.load_from_disk("artifacts/data/fineweb_edu_tokenized_SmolLM2-1.7B_with_eos_token_num_4/shard_0/")
    item = dataset[0]

    decoded = tokenizer.decode(item['input_ids'])
    assert decoded.count('<end_of_sentence_') == 32 * 4, 'decoded content has 32 eos tokens'

    for i in range(4):
        assert decoded.count(f'<end_of_sentence_{i}>') == 32, f'decoded content has 32 eos tokens for {i}'

def test_tokenized_fineweb_4_eos_tokens():
    tokenizer = GPT2TokenizerFastEOS.from_pretrained("HuggingFaceTB/SmolLM2-1.7B", num_eos_tokens=4)
    _test_tokenized_fineweb_4_eos_tokens(tokenizer)

    shutil.rmtree("/tmp/eos_tokenizer_4", ignore_errors=True)
    tokenizer_path = "/tmp/eos_tokenizer_4"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_loaded = GPT2TokenizerFastEOS.from_pretrained(tokenizer_path)
    assert tokenizer_loaded.num_eos_tokens == 4, 'reloaded tokenizer has 4 eos tokens'
    _test_tokenized_fineweb_4_eos_tokens(tokenizer_loaded)

