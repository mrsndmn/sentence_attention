# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2TokenizerFast

    >>> tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "special_embeddings_mask"]
    slow_tokenizer_class = GPT2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


class GPT2TokenizerFastEOS(GPT2TokenizerFast):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_eos_tokens = kwargs.get("num_eos_tokens", 1)
        self.gist_placement: str = kwargs.get("gist_placement", "sentence")
        self.uniform_interval_tokens: int = kwargs.get("uniform_interval_tokens", 40)
        self.boundary_predictor_model: Optional[torch.nn.Module] = kwargs.get("boundary_predictor_model", None)
        self.boundary_predictor_device: Optional[str] = kwargs.get("boundary_predictor_device", None)

        if self.gist_placement not in {"sentence", "uniform", "boundary_predictor"}:
            raise ValueError("gist_placement must be one of {'sentence','uniform','boundary_predictor'}")
        if not isinstance(self.uniform_interval_tokens, int) or self.uniform_interval_tokens <= 0:
            raise ValueError("uniform_interval_tokens must be a positive integer")
        if self.gist_placement == "boundary_predictor" and self.boundary_predictor_model is None:
            raise ValueError("boundary_predictor_model must be provided when gist_placement='boundary_predictor'")

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

    def _get_boundary_predictor_model(self):
        """Get the boundary predictor model and device."""
        return self.boundary_predictor_model, self.boundary_predictor_device

    def encode_plus(self, text: str, **kwargs):
        text = self.prepare_for_tokenization(text)
        encoding = super().encode_plus(text, **kwargs)
        if self.gist_placement == "uniform":
            encoding = self._insert_uniform_gist_into_encoding(encoding)
        elif self.gist_placement == "boundary_predictor":
            encoding = self._insert_boundary_predictor_gist_into_encoding(encoding)
        return encoding

    def batch_encode_plus(self, batch_text_or_text_pairs: str, **kwargs):
        batch_text_or_text_pairs = [self.prepare_for_tokenization(x) for x in batch_text_or_text_pairs]
        encodings = super().batch_encode_plus(batch_text_or_text_pairs, **kwargs)
        if self.gist_placement in ["uniform", "boundary_predictor"]:
            input_ids_batch = encodings.get("input_ids")
            if hasattr(input_ids_batch, "tolist"):
                input_ids_batch = input_ids_batch.tolist()
            attention_mask_batch = encodings.get("attention_mask")
            if hasattr(attention_mask_batch, "tolist"):
                attention_mask_batch = attention_mask_batch.tolist()
            special_tokens_mask_batch = encodings.get("special_tokens_mask", None)
            if hasattr(special_tokens_mask_batch, "tolist"):
                special_tokens_mask_batch = special_tokens_mask_batch.tolist()
            offset_mapping_batch = encodings.get("offset_mapping", None)
            # offset_mapping is usually a list of (start, end) tuples; keep as lists/tuples (no tensors)

            new_input_ids_batch: List[List[int]] = []
            new_attention_mask_batch: List[List[int]] = []
            new_special_tokens_mask_batch: List[List[int]] | None = [] if special_tokens_mask_batch is not None else None
            new_offset_mapping_batch: List[List[Tuple[int, int]]] | None = [] if offset_mapping_batch is not None else None

            for idx in range(len(input_ids_batch)):
                single_encoding: Dict[str, Any] = {
                    "input_ids": input_ids_batch[idx],
                    "attention_mask": attention_mask_batch[idx] if attention_mask_batch is not None else None,
                }
                if special_tokens_mask_batch is not None:
                    single_encoding["special_tokens_mask"] = special_tokens_mask_batch[idx]
                if offset_mapping_batch is not None:
                    single_encoding["offset_mapping"] = offset_mapping_batch[idx]

                if self.gist_placement == "uniform":
                    updated_single = self._insert_uniform_gist_into_encoding(single_encoding)
                else:  # boundary_predictor
                    updated_single = self._insert_boundary_predictor_gist_into_encoding(single_encoding)

                new_input_ids_batch.append(updated_single["input_ids"])
                new_attention_mask_batch.append(updated_single["attention_mask"])
                if new_special_tokens_mask_batch is not None:
                    new_special_tokens_mask_batch.append(updated_single.get("special_tokens_mask"))
                if new_offset_mapping_batch is not None:
                    new_offset_mapping_batch.append(updated_single.get("offset_mapping"))

            encodings["input_ids"] = new_input_ids_batch
            encodings["attention_mask"] = new_attention_mask_batch
            if new_special_tokens_mask_batch is not None:
                encodings["special_tokens_mask"] = new_special_tokens_mask_batch
            if new_offset_mapping_batch is not None:
                encodings["offset_mapping"] = new_offset_mapping_batch
        return encodings

    def prepare_for_tokenization(self, text: str) -> tuple[str, dict[str, Any]]:

        if self.gist_placement in ["uniform", "boundary_predictor"]:
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
        input_ids: List[int] = encoding.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()
        offset_mapping: Union[List[Tuple[int, int]], None] = encoding.get("offset_mapping")

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            special_tokens_mask = None

        new_input_ids: List[int] = []
        new_attention_mask: List[int] = []
        new_special_tokens_mask: List[int] | None = [] if special_tokens_mask is not None else None
        new_offset_mapping: List[Tuple[int, int]] | None = []

        interval: int = self.uniform_interval_tokens
        gist_ids: List[int] = self.end_of_sentence_token_ids

        tokens_since_last_insert = 0
        for idx, (token_id, mask_val) in enumerate(zip(input_ids, attention_mask)):
            new_input_ids.append(token_id)
            new_attention_mask.append(mask_val)
            if new_special_tokens_mask is not None:
                new_special_tokens_mask.append(0)
            if new_offset_mapping is not None:
                start_end = offset_mapping[idx]
                new_offset_mapping.append(start_end)

            tokens_since_last_insert += 1
            if tokens_since_last_insert >= interval:
                new_input_ids.extend(gist_ids)
                new_attention_mask.extend([1] * len(gist_ids))
                if new_special_tokens_mask is not None:
                    new_special_tokens_mask.extend([1] * len(gist_ids))
                if new_offset_mapping is not None:
                    # Place inserted gist tokens at a zero-length span at last_char_end
                    insertion_offsets = [None] * len(gist_ids)
                    new_offset_mapping.extend(insertion_offsets)
                tokens_since_last_insert = 0

        updated: Dict[str, Any] = dict(encoding)
        updated["input_ids"] = new_input_ids
        updated["attention_mask"] = new_attention_mask
        if new_special_tokens_mask is not None:
            updated["special_tokens_mask"] = new_special_tokens_mask
        if new_offset_mapping is not None:
            updated["offset_mapping"] = new_offset_mapping
        else:
            # If we couldn't reliably realign offsets, drop them to avoid inconsistency.
            if "offset_mapping" in updated:
                del updated["offset_mapping"]
        return updated

    def _insert_boundary_predictor_gist_into_encoding(self, encoding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert end-of-sentence (gist) tokens at positions predicted by the boundary predictor model.
        Expects and returns plain python lists.

        Note: The boundary predictor model expects token IDs from the base model tokenizer
        (boundary_predictor_base_model), but we're using the current tokenizer (GPT2) token IDs.
        This may cause incorrect predictions if the tokenizers differ significantly.
        """
        input_ids: List[int] = encoding.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()
        offset_mapping: Union[List[Tuple[int, int]], None] = encoding.get("offset_mapping")

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            special_tokens_mask = None

        # Get boundary predictor model
        model, device = self._get_boundary_predictor_model()

        # Run boundary predictor with current tokenizer's token IDs directly
        # WARNING: The model expects token IDs from boundary_predictor_base_model tokenizer,
        # but we're using the current tokenizer (GPT2) token IDs. This may produce incorrect
        # results if the tokenizers differ significantly.
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            boundary_logprobs, boundary_mask = model(input_ids_tensor)

        # boundary_mask[0, 0] should be set to 0 (as in test)
        boundary_mask[0, 0] = 0

        # Get boundary positions from boundary_mask
        # boundary_mask shape: [1, seq_len]
        # Boundary positions are already in terms of input_ids indices, so we can use them directly
        boundary_positions = boundary_mask[0].cpu().nonzero(as_tuple=True)[0].tolist()

        # Insert gist tokens at boundary positions
        new_input_ids: List[int] = []
        new_attention_mask: List[int] = []
        new_special_tokens_mask: List[int] | None = [] if special_tokens_mask is not None else None
        new_offset_mapping: List[Tuple[int, int]] | None = []

        gist_ids: List[int] = self.end_of_sentence_token_ids
        boundary_set = set(boundary_positions)

        for idx, (token_id, mask_val) in enumerate(zip(input_ids, attention_mask)):
            new_input_ids.append(token_id)
            new_attention_mask.append(mask_val)
            if new_special_tokens_mask is not None:
                new_special_tokens_mask.append(0)
            if new_offset_mapping is not None:
                start_end = offset_mapping[idx] if offset_mapping is not None else None
                new_offset_mapping.append(start_end)

            # Insert gist tokens after this position if it's a boundary
            if idx in boundary_set:
                new_input_ids.extend(gist_ids)
                new_attention_mask.extend([1] * len(gist_ids))
                if new_special_tokens_mask is not None:
                    new_special_tokens_mask.extend([1] * len(gist_ids))
                if new_offset_mapping is not None:
                    insertion_offsets = [None] * len(gist_ids)
                    new_offset_mapping.extend(insertion_offsets)

        updated: Dict[str, Any] = dict(encoding)
        updated["input_ids"] = new_input_ids
        updated["attention_mask"] = new_attention_mask
        if new_special_tokens_mask is not None:
            updated["special_tokens_mask"] = new_special_tokens_mask
        if new_offset_mapping is not None:
            updated["offset_mapping"] = new_offset_mapping
        else:
            if "offset_mapping" in updated:
                del updated["offset_mapping"]
        return updated


__all__ = ["GPT2TokenizerFast", "GPT2TokenizerFastEOS"]
