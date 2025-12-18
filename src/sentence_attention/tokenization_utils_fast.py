import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from sentence_attention.models.sentence_llama.modeling_sentence_llama import (
    special_token_mask_to_clothest_token_idx_slow,
)


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
        self.max_cummulative_nlogits_value: float = kwargs.get("max_cummulative_nlogits_value", 50.0)
        self.logits_model_name: str = kwargs.get("logits_model_name", "unsloth/Llama-3.2-1B")
        self.logits_chunk_size: int = kwargs.get("logits_chunk_size", 1024)
        self.boundary_predictor_model: Optional[torch.nn.Module] = kwargs.get(
            "boundary_predictor_model", None
        )  # can also be set with set_boundary_predictor_model
        self.boundary_predictor_device: Optional[str] = kwargs.get("boundary_predictor_device", None)

        if self.gist_placement not in {"sentence", "uniform", "with_logits", "boundary_predictor"}:
            raise ValueError("gist_placement must be one of {'sentence','uniform','with_logits', 'boundary_predictor}")
        if not isinstance(self.uniform_interval_tokens, int) or self.uniform_interval_tokens <= 0:
            raise ValueError("uniform_interval_tokens must be a positive integer")
        # if self.gist_placement == "boundary_predictor" and self.boundary_predictor_model is None:
        #     raise ValueError("boundary_predictor_model must be provided when gist_placement='boundary_predictor'")

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

        # Lazy loading for logits model
        self._logits_model: Optional[torch.nn.Module] = None
        self._logits_tokenizer: Optional[AutoTokenizer] = None
        self._logits_device: Optional[str] = None

        return

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return False

    def _get_logits_model(self):
        """Lazy load the model for computing logits."""
        if self._logits_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._logits_device = device
            self._logits_model = AutoModelForCausalLM.from_pretrained(self.logits_model_name, torch_dtype=torch.bfloat16)
            self._logits_model.to(device)
            self._logits_model.eval()
            self._logits_tokenizer = AutoTokenizer.from_pretrained(self.logits_model_name)
            self._logits_tokenizer.padding_side = "left"
            self._logits_tokenizer.pad_token = self._logits_tokenizer.eos_token
        return self._logits_model, self._logits_tokenizer, self._logits_device

    def set_boundary_predictor_model(self, boundary_predictor_model):
        self.boundary_predictor_model = boundary_predictor_model

        return

    def _get_boundary_predictor_model(self):
        """Get the boundary predictor model and device."""
        assert self.boundary_predictor_model is not None
        return self.boundary_predictor_model, self.boundary_predictor_device

    def encode_plus(self, text: str, **kwargs):
        text = self.prepare_for_tokenization(text)
        encoding = super().encode_plus(text, **kwargs)
        if self.gist_placement == "uniform":
            encoding = self._insert_uniform_gist_into_encoding(encoding)
        elif self.gist_placement == "with_logits":
            encoding = self._insert_with_logits_gist_into_encoding(text, encoding, **kwargs)
        elif self.gist_placement == "boundary_predictor":
            encoding = self._insert_boundary_predictor_gist_into_encoding(encoding, max_length=kwargs.get("max_length"))
        return encoding

    def batch_encode_plus(self, batch_text_or_text_pairs: str, **kwargs):
        batch_text_or_text_pairs = [self.prepare_for_tokenization(x) for x in batch_text_or_text_pairs]
        encodings = super().batch_encode_plus(batch_text_or_text_pairs, **kwargs)

        if self.gist_placement == "with_logits":
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
            offset_mapping_batch = encodings.get("offset_mapping", None)

            new_input_ids_batch: List[List[int]] = []
            new_attention_mask_batch: List[List[int]] = []
            new_special_tokens_mask_batch: List[List[int]] | None = [] if special_tokens_mask_batch is not None else None
            new_offset_mapping_batch: List[List[Tuple[int, int]]] | None = [] if offset_mapping_batch is not None else None
            new_special_embeddings_mask_batch: List[List[int]] = []
            new_clothest_idx_batch: List[List[int]] = []
            new_token_logprobs_batch: List[List[float]] = []
            new_seq_length_batch: List[int] = []

            for idx in range(len(input_ids_batch)):
                single_encoding: Dict[str, Any] = {
                    "input_ids": input_ids_batch[idx],
                    "attention_mask": attention_mask_batch[idx] if attention_mask_batch is not None else None,
                }
                if special_tokens_mask_batch is not None:
                    single_encoding["special_tokens_mask"] = special_tokens_mask_batch[idx]
                if offset_mapping_batch is not None:
                    single_encoding["offset_mapping"] = offset_mapping_batch[idx]

                updated_single = self._insert_with_logits_gist_into_encoding(
                    batch_text_or_text_pairs[idx], single_encoding, **kwargs
                )
                new_input_ids_batch.append(updated_single["input_ids"])
                new_attention_mask_batch.append(updated_single["attention_mask"])
                new_special_embeddings_mask_batch.append(updated_single["special_embeddings_mask"])
                new_clothest_idx_batch.append(updated_single["clothest_end_of_sentence_token_idx"])
                new_token_logprobs_batch.append(updated_single["token_logprobs"])
                new_seq_length_batch.append(updated_single["seq_length"])
                if new_special_tokens_mask_batch is not None:
                    new_special_tokens_mask_batch.append(updated_single.get("special_tokens_mask"))
                if new_offset_mapping_batch is not None:
                    new_offset_mapping_batch.append(updated_single.get("offset_mapping"))

            encodings["input_ids"] = new_input_ids_batch
            encodings["attention_mask"] = new_attention_mask_batch
            encodings["special_embeddings_mask"] = new_special_embeddings_mask_batch
            encodings["clothest_end_of_sentence_token_idx"] = new_clothest_idx_batch
            encodings["token_logprobs"] = new_token_logprobs_batch
            encodings["seq_length"] = new_seq_length_batch
            if new_special_tokens_mask_batch is not None:
                encodings["special_tokens_mask"] = new_special_tokens_mask_batch
            if new_offset_mapping_batch is not None:
                encodings["offset_mapping"] = new_offset_mapping_batch

            if kwargs.get("return_tensors", "") == "pt":
                encodings["input_ids"] = torch.tensor(encodings["input_ids"])
                encodings["attention_mask"] = torch.tensor(encodings["attention_mask"])
                encodings["special_embeddings_mask"] = torch.tensor(encodings["special_embeddings_mask"])
                encodings["clothest_end_of_sentence_token_idx"] = torch.tensor(encodings["clothest_end_of_sentence_token_idx"])
                encodings["token_logprobs"] = torch.tensor(encodings["token_logprobs"])
                encodings["seq_length"] = torch.tensor(encodings["seq_length"])
                if "special_tokens_mask" in encodings:
                    encodings["special_tokens_mask"] = torch.tensor(encodings["special_tokens_mask"])
        elif self.gist_placement in ["uniform", "boundary_predictor"]:
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
            offset_mapping_batch = encodings.get("offset_mapping", None)

            new_input_ids_batch: List[List[int]] = []
            new_attention_mask_batch: List[List[int]] = []
            new_special_tokens_mask_batch: List[List[int]] | None = [] if special_tokens_mask_batch is not None else None
            new_offset_mapping_batch: List[List[Tuple[int, int]]] | None = [] if offset_mapping_batch is not None else None
            insert_fn = (
                self._insert_uniform_gist_into_encoding
                if self.gist_placement == "uniform"
                else self._insert_boundary_predictor_gist_into_encoding
            )
            insert_kwargs: Dict[str, Any] = {}
            if self.gist_placement == "boundary_predictor":
                insert_kwargs["max_length"] = kwargs.get("max_length")
            for idx in range(len(input_ids_batch)):
                single_encoding: Dict[str, Any] = {
                    "input_ids": input_ids_batch[idx],
                    "attention_mask": attention_mask_batch[idx] if attention_mask_batch is not None else None,
                }
                if special_tokens_mask_batch is not None:
                    single_encoding["special_tokens_mask"] = special_tokens_mask_batch[idx]
                if offset_mapping_batch is not None:
                    single_encoding["offset_mapping"] = offset_mapping_batch[idx]

                updated_single = insert_fn(single_encoding, **insert_kwargs)
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

            if kwargs.get("return_tensors", "") == "pt":
                encodings["input_ids"] = torch.tensor(encodings["input_ids"])
                encodings["attention_mask"] = torch.tensor(encodings["attention_mask"])
                if "special_tokens_mask" in encodings:
                    encodings["special_tokens_mask"] = torch.tensor(encodings["special_tokens_mask"])
        return encodings

    def prepare_for_tokenization(self, text: str) -> tuple[str, dict[str, Any]]:

        # When uniform placement is requested, do not add tokens at sentence boundaries.
        if self.gist_placement in ["uniform", "with_logits", "boundary_predictor"]:
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
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            # Flatten batch dimension if present (we only support batch size 1 here)
            input_ids = input_ids[0]

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()
        if attention_mask is not None and len(attention_mask) > 0 and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()
        if special_tokens_mask is not None and len(special_tokens_mask) > 0 and isinstance(special_tokens_mask[0], list):
            special_tokens_mask = special_tokens_mask[0]
        offset_mapping: Union[List[Tuple[int, int]], None] = encoding.get("offset_mapping")
        if offset_mapping is not None and len(offset_mapping) > 0 and isinstance(offset_mapping[0], list):
            offset_mapping = offset_mapping[0]

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            # If mask length is inconsistent, drop it to avoid shape misalignment.
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
                # Insert gist block
                new_input_ids.extend(gist_ids)
                new_attention_mask.extend([1] * len(gist_ids))
                if new_special_tokens_mask is not None:
                    new_special_tokens_mask.extend([1] * len(gist_ids))
                if new_offset_mapping is not None:
                    # Insert None spans for inserted gist tokens
                    insertion_offsets = [None] * len(gist_ids)
                    new_offset_mapping.extend(insertion_offsets)
                tokens_since_last_insert = 0

        # Build updated encoding
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

    def _insert_with_logits_gist_into_encoding(self, text: str, encoding: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Insert end-of-sentence (gist) tokens based on cumulative negative logits
        threshold using logits from a fixed base model (default: unsloth/Llama-3.2-1B).
        Logic mirrors scripts/tokenization/tokenize_with_logits_fineweb_edu.py.
        """
        # Extract lists in a tolerant way (supporting torch/np arrays)
        input_ids: List[int] = encoding.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            # Flatten batch dimension (only batch size 1 supported here)
            input_ids = input_ids[0]

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()
        if attention_mask is not None and len(attention_mask) > 0 and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()
        if special_tokens_mask is not None and len(special_tokens_mask) > 0 and isinstance(special_tokens_mask[0], list):
            special_tokens_mask = special_tokens_mask[0]
        offset_mapping: Union[List[Tuple[int, int]], None] = encoding.get("offset_mapping")
        if offset_mapping is not None and len(offset_mapping) > 0 and isinstance(offset_mapping[0], list):
            offset_mapping = offset_mapping[0]

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            # If mask length is inconsistent, drop it to avoid shape misalignment.
            special_tokens_mask = None

        model, _, device = self._get_logits_model()

        # Compute token-level log probabilities on the *current* tokenization
        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_mask_tensor = torch.tensor([attention_mask], device=device)

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                use_cache=False,
            ).logits

            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
            target_ids = input_ids_tensor[:, 1:].unsqueeze(-1)  # [1, T-1, 1]
            gathered = torch.gather(log_probs, dim=-1, index=target_ids).squeeze(-1)  # [1, T-1]
            token_logprobs: List[float] = gathered[0].detach().to(torch.float32).cpu().tolist()

        expected_num_logits = max(len(input_ids) - 1, 0)
        if len(token_logprobs) < expected_num_logits:
            token_logprobs.extend([0.0] * (expected_num_logits - len(token_logprobs)))
        elif len(token_logprobs) > expected_num_logits:
            token_logprobs = token_logprobs[:expected_num_logits]

        # Identify where real tokens start (padding-side agnostic)
        seq_length = int(sum(attention_mask)) if attention_mask is not None else len(input_ids)
        if attention_mask is None:
            start_position_i = 0
        else:
            try:
                start_position_i = attention_mask.index(1)
            except ValueError:
                start_position_i = len(attention_mask)

        # Prepare mutable buffers
        new_input_ids: List[int] = list(input_ids)
        new_attention_mask: List[int] = list(attention_mask) if attention_mask is not None else [1] * len(input_ids)
        new_special_tokens_mask: List[int] | None = None
        if special_tokens_mask is not None:
            new_special_tokens_mask = list(special_tokens_mask)
        new_offset_mapping: List[Tuple[int, int] | None] | None = None
        if offset_mapping is not None:
            new_offset_mapping = list(offset_mapping)

        special_embeddings_mask: List[int] = [0] * len(new_input_ids)
        gist_ids: List[int] = self.end_of_sentence_token_ids
        gist_id_set = set(gist_ids)

        sum_nlogits = 0.0
        target_position_i = start_position_i
        last_position = len(input_ids) - 1

        for orig_position_i in range(start_position_i, len(input_ids)):
            if target_position_i >= len(new_input_ids):
                break

            if orig_position_i < last_position and orig_position_i < len(token_logprobs):
                logit = token_logprobs[orig_position_i]
                sum_nlogits -= logit
                if sum_nlogits > self.max_cummulative_nlogits_value:
                    sum_nlogits = 0.0
                    for gist_token in gist_ids:
                        if target_position_i >= len(new_input_ids):
                            break
                        new_input_ids[target_position_i] = gist_token
                        special_embeddings_mask[target_position_i] = 1
                        if new_special_tokens_mask is not None:
                            new_special_tokens_mask[target_position_i] = 1
                        if new_offset_mapping is not None:
                            new_offset_mapping[target_position_i] = None
                        target_position_i += 1

            if target_position_i >= len(new_input_ids):
                break

            new_input_ids[target_position_i] = input_ids[orig_position_i]
            special_embeddings_mask[target_position_i] = 1 if input_ids[orig_position_i] in gist_id_set else 0
            if new_special_tokens_mask is not None:
                value = special_tokens_mask[orig_position_i] if special_tokens_mask is not None else 0
                new_special_tokens_mask[target_position_i] = value
            if new_offset_mapping is not None:
                if offset_mapping is not None and orig_position_i < len(offset_mapping):
                    new_offset_mapping[target_position_i] = offset_mapping[orig_position_i]
                else:
                    new_offset_mapping[target_position_i] = None
            target_position_i += 1

        # Ensure any lingering gist tokens are marked in the mask
        for idx, tok_id in enumerate(new_input_ids):
            if tok_id in gist_id_set:
                special_embeddings_mask[idx] = 1

        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
            torch.tensor([special_embeddings_mask], dtype=torch.bool), num_special_tokens=self.num_eos_tokens
        )[0].tolist()

        updated: Dict[str, Any] = dict(encoding)
        updated["input_ids"] = new_input_ids
        updated["attention_mask"] = new_attention_mask
        if new_special_tokens_mask is not None:
            updated["special_tokens_mask"] = new_special_tokens_mask
        if new_offset_mapping is not None:
            updated["offset_mapping"] = new_offset_mapping
        elif "offset_mapping" in updated:
            del updated["offset_mapping"]
        updated["special_embeddings_mask"] = special_embeddings_mask
        updated["clothest_end_of_sentence_token_idx"] = clothest_end_of_sentence_token_idx
        updated["token_logprobs"] = token_logprobs
        updated["seq_length"] = seq_length

        if kwargs.get("return_tensors", "") == "pt":
            updated["input_ids"] = torch.tensor(updated["input_ids"]).unsqueeze(0)
            updated["attention_mask"] = torch.tensor(updated["attention_mask"]).unsqueeze(0)
            if "special_tokens_mask" in updated:
                updated["special_tokens_mask"] = torch.tensor(updated["special_tokens_mask"]).unsqueeze(0)
            updated["special_embeddings_mask"] = torch.tensor(updated["special_embeddings_mask"]).unsqueeze(0)
            updated["clothest_end_of_sentence_token_idx"] = torch.tensor(
                updated["clothest_end_of_sentence_token_idx"]
            ).unsqueeze(0)
            if "token_logprobs" in updated:
                updated["token_logprobs"] = torch.tensor(updated["token_logprobs"])

        # Preserve BatchEncoding interface (so .to works)
        try:
            encodings_attr = getattr(encoding, "encodings", None)
            return encoding.__class__(updated, encoding=encodings_attr)
        except Exception:
            return updated

    def _insert_boundary_predictor_gist_into_encoding(
        self, encoding: Dict[str, Any], max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Insert end-of-sentence (gist) tokens at positions predicted by the boundary predictor model.
        Expects and returns plain python lists.

        Note: The boundary predictor model expects token IDs from the base model tokenizer
        (boundary_predictor_base_model), but we're using the current tokenizer token IDs.
        This may cause incorrect predictions if the tokenizers differ significantly.
        If `max_length` is provided, the returned sequence is truncated to that length
        after inserting gist tokens so the caller's length constraint is preserved.
        """
        input_ids: List[int] = encoding.get("input_ids", [])
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        attention_mask: Union[List[int], None] = encoding.get("attention_mask")
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()
        if attention_mask is not None and len(attention_mask) > 0 and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]

        special_tokens_mask: Union[List[int], None] = encoding.get("special_tokens_mask")
        if hasattr(special_tokens_mask, "tolist"):
            special_tokens_mask = special_tokens_mask.tolist()
        if special_tokens_mask is not None and len(special_tokens_mask) > 0 and isinstance(special_tokens_mask[0], list):
            special_tokens_mask = special_tokens_mask[0]
        offset_mapping: Union[List[Tuple[int, int]], None] = encoding.get("offset_mapping")
        if hasattr(offset_mapping, "tolist"):
            offset_mapping = offset_mapping.tolist()
        if offset_mapping is not None and len(offset_mapping) > 0 and isinstance(offset_mapping[0], list):
            offset_mapping = offset_mapping[0]

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        if special_tokens_mask is not None and len(special_tokens_mask) != len(input_ids):
            special_tokens_mask = None

        model, device = self._get_boundary_predictor_model()

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            boundary_logprobs, boundary_mask = model(input_ids_tensor)
            boundary_mask = boundary_mask.cpu()

        boundary_mask[0, 0] = 0
        # mask pad tokens
        boundary_mask[0, torch.tensor(attention_mask).bool() == False] = 0  ## noqa E712

        boundary_positions = boundary_mask[0].cpu().nonzero(as_tuple=True)[0].tolist()

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

            if idx in boundary_set:
                new_input_ids.extend(gist_ids)
                new_attention_mask.extend([1] * len(gist_ids))
                if new_special_tokens_mask is not None:
                    new_special_tokens_mask.extend([1] * len(gist_ids))
                if new_offset_mapping is not None:
                    insertion_offsets = [None] * len(gist_ids)
                    new_offset_mapping.extend(insertion_offsets)

        if max_length is not None and len(new_input_ids) > max_length:
            new_input_ids = new_input_ids[:max_length]
            new_attention_mask = new_attention_mask[:max_length]
            if new_special_tokens_mask is not None:
                new_special_tokens_mask = new_special_tokens_mask[:max_length]
            if new_offset_mapping is not None:
                new_offset_mapping = new_offset_mapping[:max_length]

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
