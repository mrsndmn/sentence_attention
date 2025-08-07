from sentence_attention.trainer.arguments import SentenceTrainingArguments
from sentence_attention.trainer.build_model_tokenizer import build_model_tokenizer

from sentence_attention.models.sentence_gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from sentence_attention.models.sentence_qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from sentence_attention.tokenization_utils_fast import PreTrainedTokenizerFastEOS


def test_build_model_tokenizer():

    model_tokenizer_class = [
        ("HuggingFaceTB/SmolLM2-135M", GPT2TokenizerFastEOS),
        ("unsloth/Llama-3.2-1B", PreTrainedTokenizerFastEOS),
        ("Qwen/Qwen2.5-1.5B", Qwen2TokenizerFastEOS),
    ]

    for model_checkpoint, tokenizer_class in model_tokenizer_class:

        for num_eos_tokens in [1, 4]:
            training_args = SentenceTrainingArguments(
                model_checkpoint=model_checkpoint,
                number_of_eos_tokens=num_eos_tokens,
                model_type="sentence_pretrained_checkpoint",
                add_end_of_sentence_token=True,
            )

            _, tokenizer = build_model_tokenizer(training_args)

            assert (
                tokenizer.num_eos_tokens == num_eos_tokens
            ), f"num_eos_tokens is {tokenizer.num_eos_tokens} but should be {num_eos_tokens} model {model_checkpoint},"
            assert isinstance(tokenizer, tokenizer_class), f"tokenizer is {tokenizer} but should be {tokenizer_class}"

            if num_eos_tokens != 1:
                assert (
                    "<end_of_sentence>" not in tokenizer.get_vocab()
                ), f"<end_of_sentence> is in the vocab for model {model_checkpoint}"
