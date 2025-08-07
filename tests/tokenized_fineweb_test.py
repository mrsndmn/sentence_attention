
from datasets import Dataset
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS


def test_tokenized_fineweb():
    dataset = Dataset.load_from_disk("fineweb_edu_tokenized_gpt2_eos/000_00000.parquet/")
    item = dataset[0]
    tokenizer = GPT2TokenizerFastEOS.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    decoded = tokenizer.decode(item['input_ids'])
    print("decoded", decoded)
    breakpoint()
