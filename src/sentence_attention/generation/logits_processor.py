from collections.abc import Sequence
from typing import List

import torch
from transformers.generation.logits_process import LogitsProcessor


class SequentialEOSTokensLogitsProcessor(LogitsProcessor):
    """Force generation of a chain of EOS tokens.

    If the previously generated token is `<end_of_sentence_0>`, the next token is forced to be
    `<end_of_sentence_1>`, then `<end_of_sentence_2>`, and so on, until the last EOS token in the chain.

    This processor does NOT force any token after the last EOS token in the provided sequence.
    """

    def __init__(self, eos_token_ids: Sequence[int]):
        if len(eos_token_ids) < 2:
            raise ValueError("SequentialEOSTokensLogitsProcessor requires at least 2 EOS token ids")

        self._eos_token_ids: List[int] = list(eos_token_ids)
        # Map each eos_i to eos_{i+1}, except the last one
        self._next_token_by_prev = {
            self._eos_token_ids[i]: self._eos_token_ids[i + 1] for i in range(len(self._eos_token_ids) - 1)
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        if input_ids.size(1) == 0:
            return scores

        prev_tokens = input_ids[:, -1]
        neg_inf = torch.finfo(scores.dtype).min

        for batch_index in range(prev_tokens.size(0)):
            prev_token_id = int(prev_tokens[batch_index].item())
            if prev_token_id in self._next_token_by_prev:
                forced_id = int(self._next_token_by_prev[prev_token_id])
                scores[batch_index, :] = neg_inf
                scores[batch_index, forced_id] = 0.0

        return scores


def build_flexible_eos_logits_processors(model) -> List[LogitsProcessor]:
    """Build logits processors for flexible-EOS generation based on model.config.

    Returns an empty list if the feature is disabled or improperly configured.
    """
    processors: List[LogitsProcessor] = []

    config = getattr(model, "config", None)
    if config is None:
        return processors

    flexible = getattr(config, "flexible_eos_tokens", False)
    eos_ids = getattr(config, "end_of_sentence_token_ids", None)

    if flexible and isinstance(eos_ids, (list, tuple)) and len(eos_ids) > 1:
        processors.append(SequentialEOSTokensLogitsProcessor(eos_ids))

    return processors


__all__ = [
    "SequentialEOSTokensLogitsProcessor",
    "build_flexible_eos_logits_processors",
]
