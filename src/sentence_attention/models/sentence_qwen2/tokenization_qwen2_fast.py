# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Qwen2."""

import random
import re
from typing import Any, Dict, List, Tuple, Union

import nltk

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.tokenization_utils import AddedToken
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}


MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import Qwen2TokenizerFast

    >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    This is expected.

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
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        # We need to at least pass vocab_file and merges_file to base class
        # in case a slow tokenizer needs to be initialized; other can be
        # configured through files.
        # following GPT2TokenizerFast, also adding unk_token, bos_token, and eos_token

        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


class Qwen2TokenizerFastEOS(PreTrainedTokenizerFast):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_eos_tokens = kwargs.get("num_eos_tokens", 1)
        self.gist_placement: str = kwargs.get("gist_placement", "sentence")
        self.uniform_interval_tokens: int = kwargs.get("uniform_interval_tokens", 40)

        # New parameters for gist sets and insertion strategy with fixed defaults
        self.gist_sets: str = kwargs.get("gist_sets", "single")
        self.gist_insertion_strategy: str = kwargs.get("gist_insertion_strategy", "default")
        self.num_gist_sets_in_swarm: int = kwargs.get("num_gist_sets_in_swarm", 1)

        if self.gist_placement not in {"sentence", "sentence_nltk", "uniform"}:
            raise ValueError("gist_placement must be one of {'sentence','sentence_nltk','uniform'}")
        if not isinstance(self.uniform_interval_tokens, int) or self.uniform_interval_tokens <= 0:
            raise ValueError("uniform_interval_tokens must be a positive integer")
        if self.gist_sets not in {"single", "roll", "swarm"}:
            raise ValueError("gist_sets must be one of {'single', 'roll', 'swarm'}")
        if self.gist_insertion_strategy not in {"default", "random"}:
            raise ValueError("gist_insertion_strategy must be one of {'default', 'random'}")
        if self.gist_insertion_strategy == "random":
            if self.gist_placement != "sentence_nltk":
                raise ValueError("gist_insertion_strategy='random' can only be used with gist_placement='sentence_nltk'")
            if self.gist_sets not in {"swarm", "roll"}:
                raise ValueError("gist_insertion_strategy='random' can only be used with gist_sets in {'swarm', 'roll'}")

        self.start_end_of_sentence_token = []

        if self.num_eos_tokens >= 1:
            # Initialize tokens based on gist_sets mode
            if self.gist_sets == "single":
                self.end_of_sentence_tokens_list = [f"<end_of_sentence_{i}>" for i in range(self.num_eos_tokens)]
                self.start_end_of_sentence_token.append(self.end_of_sentence_tokens_list[0])
            elif self.gist_sets == "roll":
                all_sets_gist_tokens = []
                for current_num_gist_tokens in range(self.num_eos_tokens):
                    for gist_i in range(current_num_gist_tokens + 1):
                        gist_token = f"<end_of_sentence_{gist_i}_{current_num_gist_tokens}>"
                        all_sets_gist_tokens.append(gist_token)
                        if gist_i == 0:
                            self.start_end_of_sentence_token.append(gist_token)
                self.end_of_sentence_tokens_list = all_sets_gist_tokens
            elif self.gist_sets == "swarm":
                all_sets_gist_tokens = []
                for swarm_i in range(self.num_gist_sets_in_swarm):
                    for gist_i in range(self.num_eos_tokens):
                        gist_token = f"<end_of_sentence_{gist_i}_swarm_{swarm_i}>"
                        all_sets_gist_tokens.append(gist_token)
                        if gist_i == 0:
                            self.start_end_of_sentence_token.append(gist_token)

                self.end_of_sentence_tokens_list = all_sets_gist_tokens
        else:
            raise ValueError("num_eos_tokens cant be negative")

        self.end_of_sentence_token_ids = []
        for end_of_sentence_token in self.end_of_sentence_tokens_list:
            if end_of_sentence_token not in self.get_vocab():
                self.add_special_tokens({"additional_special_tokens": [end_of_sentence_token]})
                print(f"Added <end_of_sentence> token with ID: {self.convert_tokens_to_ids(end_of_sentence_token)}")

            self.end_of_sentence_token_ids.append(self.convert_tokens_to_ids(end_of_sentence_token))

        self.start_end_of_sentence_token_ids = [
            self.convert_tokens_to_ids(start_eos_token) for start_eos_token in self.start_end_of_sentence_token
        ]

        # For roll and swarm modes, organize tokens into sets for random selection
        if self.gist_sets == "roll":
            self.gist_token_sets: List[List[str]] = []
            for current_num_gist_tokens in range(self.num_eos_tokens):
                set_tokens = [
                    f"<end_of_sentence_{gist_i}_{current_num_gist_tokens}>" for gist_i in range(current_num_gist_tokens + 1)
                ]
                self.gist_token_sets.append(set_tokens)
        elif self.gist_sets == "swarm":
            self.gist_token_sets: List[List[str]] = []
            for swarm_i in range(self.num_gist_sets_in_swarm):
                set_tokens = [f"<end_of_sentence_{gist_i}_swarm_{swarm_i}>" for gist_i in range(self.num_eos_tokens)]
                self.gist_token_sets.append(set_tokens)
        else:
            self.gist_token_sets = None

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

                updated_single = self._insert_uniform_gist_into_encoding(single_encoding)
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

        # When uniform placement is requested, do not add tokens at sentence boundaries.
        if self.gist_placement == "uniform":
            return text

        if self.gist_placement == "sentence_nltk":
            sentence_tokenizer = nltk.tokenize.PunktTokenizer()
            # Get sentence boundaries
            orig_text_len = len(text)
            sentence_spans = list(sentence_tokenizer.span_tokenize(text))
            # Insert tokens at sentence boundaries (in reverse order to preserve indices)
            text_strip = text.strip()
            len_text_strip = len(text_strip)
            for _sent_start, sent_end in reversed(sentence_spans):
                if sent_end in [orig_text_len, len_text_strip]:
                    continue

                # Determine which tokens to insert
                if self.gist_insertion_strategy == "random" and self.gist_token_sets is not None:
                    # Select a random set of tokens
                    selected_set = random.choice(self.gist_token_sets)
                    end_of_sentence_token = "".join(selected_set)
                else:
                    # Default: use all default tokens
                    end_of_sentence_token = "".join(self.end_of_sentence_tokens_list)
                # Insert tokens at the end of each sentence
                text = text[:sent_end] + end_of_sentence_token + text[sent_end:]
        else:
            # Original regex-based sentence detection for "sentence" placement
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
        new_offset_mapping: List[Tuple[int, int]] | None = [] if offset_mapping is not None else None

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
            if "offset_mapping" in updated:
                del updated["offset_mapping"]
        return updated
