import re
from typing import Any, Dict, List, Union

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
        self.gist_placement: str = kwargs.get("gist_placement", "sentence")
        self.uniform_interval_tokens: int = kwargs.get("uniform_interval_tokens", 40)

        if self.gist_placement not in {"sentence", "uniform"}:
            raise ValueError("gist_placement must be one of {'sentence','uniform'}")
        if not isinstance(self.uniform_interval_tokens, int) or self.uniform_interval_tokens <= 0:
            raise ValueError("uniform_interval_tokens must be a positive integer")

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
        encoding = super().encode_plus(text, **kwargs)
        if self.gist_placement == "uniform":
            encoding = self._insert_uniform_gist_into_encoding(encoding)
        return encoding

    def batch_encode_plus(self, batch_text_or_text_pairs: str, **kwargs):
        batch_text_or_text_pairs = [self.prepare_for_tokenization(x) for x in batch_text_or_text_pairs]
        encodings = super().batch_encode_plus(batch_text_or_text_pairs, **kwargs)
        if self.gist_placement == "uniform":
            # Expect lists (no tensors). If tensors are returned, try converting to lists conservatively.
            input_ids_batch = encodings.get("input_ids")
            if hasattr(input_ids_batch, "tolist"):
                input_ids_batch = input_ids_batch.tolist()
            attention_mask_batch = encodings.get("attention_mask")
            if hasattr(attention_mask_batch, "tolist"):
                attention_mask_batch = attention_mask_batch.tolist()
            special_tokens_mask_batch = encodings.get("special_tokens_mask", None)
            if hasattr(special_tokens_mask_batch, "tolist"):
                special_tokens_mask_batch = special_tokens_mask_batch.tolist()

            new_input_ids_batch: List[List[int]] = []
            new_attention_mask_batch: List[List[int]] = []
            new_special_tokens_mask_batch: List[List[int]] | None = [] if special_tokens_mask_batch is not None else None

            for idx in range(len(input_ids_batch)):
                single_encoding: Dict[str, Any] = {
                    "input_ids": input_ids_batch[idx],
                    "attention_mask": attention_mask_batch[idx] if attention_mask_batch is not None else None,
                }
                if special_tokens_mask_batch is not None:
                    single_encoding["special_tokens_mask"] = special_tokens_mask_batch[idx]

                updated_single = self._insert_uniform_gist_into_encoding(single_encoding)
                new_input_ids_batch.append(updated_single["input_ids"])
                new_attention_mask_batch.append(updated_single["attention_mask"])
                if new_special_tokens_mask_batch is not None:
                    new_special_tokens_mask_batch.append(updated_single.get("special_tokens_mask"))

            encodings["input_ids"] = new_input_ids_batch
            encodings["attention_mask"] = new_attention_mask_batch
            if new_special_tokens_mask_batch is not None:
                encodings["special_tokens_mask"] = new_special_tokens_mask_batch

            # Remove fields that would be out of sync after insertion.
            for removable in ("offset_mapping",):
                if removable in encodings:
                    del encodings[removable]
        return encodings

    def prepare_for_tokenization(self, text: str) -> tuple[str, dict[str, Any]]:

        # When uniform placement is requested, do not add tokens at sentence boundaries.
        if self.gist_placement == "uniform":
            return text

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

    def _insert_uniform_gist_into_encoding(self, encoding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert end-of-sentence (gist) tokens uniformly every `uniform_interval_tokens` tokens.
        Expects and returns plain python lists.
        """
        # Extract lists in a tolerant way (supporting torch/np arrays)
        input_ids: List[int] = encoding.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            # If mask length is inconsistent, drop it to avoid shape misalignment.
            special_tokens_mask = None

        new_input_ids: List[int] = []
        new_attention_mask: List[int] = []
        new_special_tokens_mask: List[int] | None = [] if special_tokens_mask is not None else None

        interval: int = self.uniform_interval_tokens
        gist_ids: List[int] = self.end_of_sentence_token_ids

        tokens_since_last_insert = 0
        for token_id, mask_val in zip(input_ids, attention_mask):
            new_input_ids.append(token_id)
            new_attention_mask.append(mask_val)
            if new_special_tokens_mask is not None:
                new_special_tokens_mask.append(0)

            tokens_since_last_insert += 1
            if tokens_since_last_insert >= interval:
                # Insert gist block
                new_input_ids.extend(gist_ids)
                new_attention_mask.extend([1] * len(gist_ids))
                if new_special_tokens_mask is not None:
                    new_special_tokens_mask.extend([1] * len(gist_ids))
                tokens_since_last_insert = 0

        # Build updated encoding
        updated: Dict[str, Any] = dict(encoding)
        updated["input_ids"] = new_input_ids
        updated["attention_mask"] = new_attention_mask
        if new_special_tokens_mask is not None:
            updated["special_tokens_mask"] = new_special_tokens_mask
        # Remove fields that would be out of sync after insertion.
        for removable in ("offset_mapping",):
            if removable in updated:
                del updated[removable]
        return updated
