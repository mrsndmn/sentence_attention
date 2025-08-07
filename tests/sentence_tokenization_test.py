from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama_fast import EOSTokenizerFast


# TODO check multiple tokens for eos sentences
def test_eos_tokenizer():
    tokenizer = EOSTokenizerFast.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    _test_eos_tokenizer(tokenizer)

    tokenizer_path = "/tmp/eos_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_loaded = AutoTokenizer.from_pretrained(tokenizer_path)

    _test_eos_tokenizer(tokenizer_loaded)

    print("Tests are ok!")


def _test_eos_tokenizer(tokenizer):
    assert tokenizer("Hello, how are you? ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("This is a test sentence. ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("What do you think? ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("Amazing! ")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("This ends with a period.\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("This ends with a question mark?\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("This ends with an exclamation mark!\n")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("Multiple sentences. Here is another one. And a third one.")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
    assert tokenizer("Questions? Yes! And more questions?")['input_ids'][-1] == tokenizer.end_of_sentence_token_id
