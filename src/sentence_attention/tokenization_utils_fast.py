import re
from typing import Any

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class PreTrainedTokenizerFastEOS(PreTrainedTokenizerFast):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_eos_tokens = kwargs.get("num_eos_tokens", 1)

        if self.num_eos_tokens >= 1:
            self.end_of_sentence_tokens_list = [f"<end_of_sentence_{i}>" for i in range(self.num_eos_tokens)]
        else:
            raise ValueError("num_eos_tokens cant be negative")

        self.end_of_sentence_token_ids = []
        for end_of_sentence_token in self.end_of_sentence_tokens_list:
            if end_of_sentence_token not in self.get_vocab():
                self.add_special_tokens({"additional_special_tokens": [end_of_sentence_token]})
                print(f"Added <end_of_sentence> token with ID: {self.convert_tokens_to_ids(end_of_sentence_token)}")

            self.end_of_sentence_token_ids.append(self.convert_tokens_to_ids(end_of_sentence_token))

        return

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return False

    def encode_plus(self, text: str, **kwargs):
        text = self.prepare_for_tokenization(text)
        return super().encode_plus(text, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs: str, **kwargs):
        batch_text_or_text_pairs = [self.prepare_for_tokenization(x) for x in batch_text_or_text_pairs]
        return super().batch_encode_plus(batch_text_or_text_pairs, **kwargs)

    def prepare_for_tokenization(self, text: str) -> tuple[str, dict[str, Any]]:

        end_of_sentence_token = "".join(self.end_of_sentence_tokens_list)

        patterns = [
            (r"\. ", f". {end_of_sentence_token}"),
            (r"\? ", f"? {end_of_sentence_token}"),
            (r"! ", f"! {end_of_sentence_token}"),
            (r"\.\n", f".\n{end_of_sentence_token}"),
            (r"\?\n", f"?\n{end_of_sentence_token}"),
            (r"!\n", f"!\n{end_of_sentence_token}"),
            (r"\.$", f". {end_of_sentence_token}"),
            (r"!$", f"!{end_of_sentence_token}"),
            (r"\?$", f"?{end_of_sentence_token}"),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text
