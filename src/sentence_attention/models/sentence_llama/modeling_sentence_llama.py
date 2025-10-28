# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple, Union, Unpack

import torch
import torch.nn as nn
import torch.utils.checkpoint
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.integrations.flex_attention import compile_friendly_flex_attention
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


CHECK_WITH_PYTHON = False

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


def special_token_mask_to_clothest_token_idx_slow(special_token_mask, num_special_tokens=None):
    # [ bs, seq_len ]

    assert num_special_tokens is not None, "num_special_tokens must be provided"

    special_token_mask_bool = special_token_mask.bool()
    special_token_mask_bool = special_token_mask_bool.cpu()

    clothest_token_idx = torch.zeros_like(special_token_mask, dtype=torch.long, device="cpu")

    for batch_i in range(special_token_mask_bool.shape[0]):
        current_clothest_token_idx = 0
        current_sequrntial_num_special_tokens = 0
        for seq_len_i in range(special_token_mask_bool.shape[1]):
            if special_token_mask_bool[batch_i, seq_len_i].item():
                current_sequrntial_num_special_tokens += 1

            if (
                special_token_mask_bool[batch_i, seq_len_i].item()
                and current_sequrntial_num_special_tokens == num_special_tokens
            ):
                clothest_token_idx[batch_i, seq_len_i] = current_clothest_token_idx
                current_clothest_token_idx = seq_len_i
                current_sequrntial_num_special_tokens = 0
            else:
                clothest_token_idx[batch_i, seq_len_i] = current_clothest_token_idx

    clothest_token_idx = clothest_token_idx.to(special_token_mask.device)

    return clothest_token_idx


def sentence_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # print("is_causal", is_causal)
    # import os
    # if True:
    #     print("query", query.shape)
    #     print("key", key.shape)
    #     print("value", value.shape)
    #     print("module.num_key_value_groups", module.num_key_value_groups)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_weights = None
    if kwargs.get("output_attentions", False):
        print("Output attentions")
        attn_weights = torch.nn.functional.softmax((query @ key.permute(0, 1, 3, 2)) + causal_mask, dim=-1)

    return attn_output, attn_weights


def sentence_attention_forward_flex(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: float | None = None,
    softcap: float | None = None,
    head_mask: torch.Tensor | None = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # if hasattr(module, "num_key_value_groups"):
    #     # print("module.num_key_value_groups", module.num_key_value_groups)
    #     key = repeat_kv(key, module.num_key_value_groups)
    #     value = repeat_kv(value, module.num_key_value_groups)

    assert isinstance(attention_mask, torch.nn.attention.flex_attention.BlockMask), "attention_mask must be a BlockMask"

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # attn_output = torch.nn.functional.flex_attention.flex_attention(
    attn_output = compile_friendly_flex_attention(
        # attn_output = torch.nn.attention.flex_attention.flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        # enable_gqa=False,  # explicit repeat kv is already done above
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=False,
    )
    # attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


ALL_ATTENTION_FUNCTIONS["sentence_attention"] = sentence_attention_forward
ALL_ATTENTION_FUNCTIONS["sentence_attention_flex"] = sentence_attention_forward_flex


@dataclass
class SentenceBaseModelOutputWithPast(BaseModelOutputWithPast):
    pass


@dataclass
class SentenceCausalLMOutputWithPast(CausalLMOutputWithPast):
    last_hidden_state: torch.Tensor = None


class SentenceLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor | None, Tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        assert cos.shape[1] == query_states.shape[2] and key_states.shape[2]

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # if self.layer_idx == 0:
        #     print("cache_position", cache_position[:3], cache_position[-3:])
        #     print("query_states", query_states.shape)
        #     print("key_states", key_states.shape)
        #     print("value_states", value_states.shape)
        #     print("attention_mask", attention_mask.shape)
        #     # breakpoint()

        attention_interface: Callable = eager_attention_forward

        # assert self.config._attn_implementation == 'sentence_attention', f"self.config._attn_implementation os not sentence_attention: {self.config._attn_implementation}"

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "sentence_attention_flex":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                ft_with_bos_token=self.config.ft_with_bos_token,
                **kwargs,
            )
        elif self.config._attn_implementation == "sentence_attention":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid attention implementation: {self.config._attn_implementation}")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class SentenceLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.layer_idx = layer_idx
        # config._attn_implementation = 'eager'
        # config._attn_implementation = 'sdpa'
        # config._attn_implementation = 'flash_attention_2'
        self.self_attn = SentenceLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        special_embeddings_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class SentenceLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SentenceLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class SentenceLlamaModel(SentenceLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SentenceLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config._attn_implementation = "sentence_attention"
        # self.config._attn_implementation = "sentence_attention_flex"

        print("SentenceLlamaModel self.config._attn_implementation", self.config._attn_implementation)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        assert config.num_hidden_layers % 2 == 0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [SentenceLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_decoder_layer(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,
        special_embeddings_mask,
        clothest_end_of_sentence_token_idx,
    ):

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                special_embeddings_mask,
                clothest_end_of_sentence_token_idx,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            )

        return layer_outputs

    def forward_decoder_layers(
        self,
        decoder_layers,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        output_hidden_states,
        use_cache,
        cache_position,
        position_embeddings,
        all_hidden_states,
        all_self_attns,
        special_embeddings_mask,
        clothest_end_of_sentence_token_idx,
    ):

        for decoder_layer in decoder_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = self.forward_decoder_layer(
                decoder_layer,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                special_embeddings_mask,
                clothest_end_of_sentence_token_idx,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        return hidden_states, all_hidden_states, all_self_attns

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @torch.compiler.disable(recursive=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        token_frequency: torch.Tensor | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        stop_words_tokens_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | List[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_sentence_chunked_prefill: bool = False,
        num_items_in_batch=None,  # not used
        # **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple | SentenceBaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_ids is not None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.training and use_cache:
            logger.warning_once("use_cache=True should not be used in training")
            use_cache = False

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        assert self.config._attn_implementation in [
            "sentence_attention",
            "sentence_attention_flex",
        ], f"config._attn_implementation is expected to be 'sentence_attention', but got {self.config._attn_implementation}"

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            past_key_values_len = 0
            if past_key_values is not None:
                past_key_values_len = past_key_values.get_seq_length()

            attention_mask = torch.ones(
                [input_ids.shape[0], input_ids.shape[1] + past_key_values_len], device=input_ids.device, dtype=torch.long
            )

        if special_embeddings_mask is None:
            special_embeddings_mask = torch.zeros_like(attention_mask)
            if self.config.end_of_sentence_token_ids is not None:
                total_eos_tokens = 0
                for end_of_sentence_token_id in self.config.end_of_sentence_token_ids:
                    special_embeddings_mask[input_ids == end_of_sentence_token_id] = 1
                    total_eos_tokens += (input_ids == end_of_sentence_token_id).sum().item()
                print("number of end of sentence tokens", total_eos_tokens)

        assert special_embeddings_mask is not None

        if clothest_end_of_sentence_token_idx is None:
            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                special_embeddings_mask,
                num_special_tokens=len(self.config.end_of_sentence_token_ids),
            )

        assert len(attention_mask.shape) == 2

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            special_embeddings_mask=special_embeddings_mask,
            is_sentence_chunked_prefill=is_sentence_chunked_prefill,
        )
        # if causal_mask is not None:
        #     print("causal_mask", causal_mask.shape)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Before FanIn
        hidden_states, all_hidden_states, all_self_attns = self.forward_decoder_layers(
            self.layers,
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            output_attentions,
            output_hidden_states,
            use_cache,
            cache_position,
            position_embeddings,
            all_hidden_states,
            all_self_attns,
            special_embeddings_mask,
            clothest_end_of_sentence_token_idx,
        )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        assert return_dict

        return SentenceBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        clothest_end_of_sentence_token_idx: torch.Tensor,
        special_embeddings_mask: torch.Tensor,
        is_sentence_chunked_prefill: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
            )

        # print("update causal mask: target_length", target_length)
        # print("update causal mask: sequence_length", sequence_length)

        if self.config._attn_implementation in ["sentence_attention"] and not is_sentence_chunked_prefill:
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                special_embeddings_mask=special_embeddings_mask,
                ft_with_bos_token=self.config.ft_with_bos_token,
            )
        elif self.config._attn_implementation in ["sentence_attention_flex"]:
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention_flex(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                special_embeddings_mask=special_embeddings_mask,
            )
        else:
            # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
            )

        if (
            (self.config._attn_implementation in ["sdpa", "sentence_attention", "eager"])
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position_sentence_attention_flex(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        clothest_end_of_sentence_token_idx: torch.Tensor,
        special_embeddings_mask: torch.Tensor,
    ):

        attention_mask_bool = attention_mask.bool()
        special_embeddings_mask = special_embeddings_mask.bool()

        # [ bs, seq_len ]
        assert len(attention_mask_bool.shape) == 2

        def mask_mod(b, h, q_idx, kv_idx):
            eos_token_idx = clothest_end_of_sentence_token_idx[b, q_idx]

            causal_mask = (kv_idx <= q_idx) & attention_mask_bool[b, q_idx] & attention_mask_bool[b, kv_idx]
            eos_sync_tokens = causal_mask & special_embeddings_mask[b, kv_idx]
            causal_triu_mask = causal_mask & (kv_idx >= eos_token_idx)

            return causal_triu_mask | eos_sync_tokens

        batch_size = attention_mask.shape[0]
        q_idx = sequence_length
        kv_idx = target_length

        # print("input_tensor.device", input_tensor.device)
        block_mask = torch.nn.attention.flex_attention.create_block_mask(
            mask_mod,
            batch_size,
            None,
            q_idx,
            kv_idx,
            device=attention_mask.device,
            # BLOCK_SIZE=256,
            BLOCK_SIZE=128,
            # BLOCK_SIZE=64,
            # BLOCK_SIZE=32,
        )

        # print("block_mask", block_mask)

        return block_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        ft_with_bos_token: bool = False,
        **kwargs,
    ):
        """
        Same signature as the original implementation, but now fully vectorized (no Python `for` loops).

        The mask obeys the following rules (see the user-provided `expected_mask` example):

        1. **Causality** – a token can only attend to previous (or itself) positions.
        2. **Block causality** – for every query position *q* the first token that follows an *end-of-sentence* (EOS)
           symbol acts as a new causal block.  All keys *k* such that
           `eos_idx(q) ≤ k ≤ q` are therefore visible, while tokens before `eos_idx(q)` are masked.
        3. **Special embeddings** – any key that is marked in `special_embeddings_mask` is always visible as long as
           it is not positioned *after* the current query (i.e. it still must satisfy `k ≤ q`).
        4. **Padding** – the classical 2-D `attention_mask` is honoured for both query and key positions.

        The resulting tensor has shape ``(batch_size, 1, sequence_length, target_length)`` and contains
        ``1`` for visible positions and ``-inf`` (the minimum of ``dtype``) for masked positions, matching the
        heuristics that were previously produced by the Python triple-nested loop.
        """

        # Fast return if a 4-D mask is already provided – keep the original behaviour.
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        # ----------------------------------------------------------------------------
        # Preconditions & common tensors
        # ----------------------------------------------------------------------------
        assert attention_mask is not None, "`attention_mask` must be provided for sentence attention."  # keeps behaviour
        assert attention_mask.dim() == 2, "`attention_mask` is expected to be 2-D (batch, seq_len)."

        bs = batch_size
        q_len = sequence_length
        k_len = target_length

        min_val = torch.finfo(dtype).min

        # Convert the 2-D masks to bool for logical operations
        attention_mask_bool = attention_mask.to(dtype=torch.bool)  # [bs, k_len]

        clothest_end_of_sentence_token_idx = kwargs["clothest_end_of_sentence_token_idx"]  # [bs, q_len]
        special_embeddings_mask = kwargs["special_embeddings_mask"].to(torch.bool)  # [bs, k_len]

        # ----------------------------------------------------------------------------
        # Build broadcastable index grids for queries (q) and keys (k)
        # ----------------------------------------------------------------------------
        q_idx = torch.arange(q_len, device=device).view(1, q_len, 1)  # shape: (1, q_len, 1)
        k_idx = torch.arange(k_len, device=device).view(1, 1, k_len)  # shape: (1, 1, k_len)

        # ----------------------------------------------------------------------------
        # Base causal condition: k ≤ q
        # ----------------------------------------------------------------------------
        diff_q_idx_k_idx = k_idx.max() - q_idx.max()

        causal_base = k_idx <= (q_idx + diff_q_idx_k_idx)  # (1, q_len, k_len)

        # ----------------------------------------------------------------------------
        # Padding masks for queries and keys
        # ----------------------------------------------------------------------------
        if q_len < attention_mask_bool.shape[1]:
            # assert sequence is left padded
            q_valid = attention_mask_bool[:, -q_len:].view(bs, q_len, 1)  # (bs, q_len, 1)
        else:
            q_valid = attention_mask_bool.view(bs, q_len, 1)  # (bs, q_len, 1)

        k_valid = attention_mask_bool.view(bs, 1, k_len)  # (bs, 1, k_len)
        valid_positions = q_valid & k_valid  # (bs, q_len, k_len)

        causal_valid_positions = valid_positions.clone()
        # if q_len < attention_mask_bool.shape[1]:
        #     causal_valid_positions[:, :, :-q_len] = False

        # Apply base causal & validity
        causal_and_valid = causal_base & causal_valid_positions  # (bs, q_len, k_len)
        full_causal_and_valid = causal_base & valid_positions  # (bs, q_len, k_len)

        # ----------------------------------------------------------------------------
        # Block-causal component via EOS index
        # ----------------------------------------------------------------------------
        # clothest_end_of_sentence_token_idx gives, for every query, the index of the closest EOS *at or before* q.
        if q_len < clothest_end_of_sentence_token_idx.shape[1]:
            eos_idx = clothest_end_of_sentence_token_idx[:, -q_len:].view(bs, q_len, 1)  # (bs, q_len, 1)
        else:
            eos_idx = clothest_end_of_sentence_token_idx.view(bs, q_len, 1)  # (bs, q_len, 1)

        # block_causal = causal_and_valid.clone()
        block_causal = causal_and_valid & (k_idx >= eos_idx)
        # block_causal[:, :, -q_len:] = causal_and_valid[:, :, -q_len:] & (k_idx[:, :, -q_len:] >= eos_idx)

        # ----------------------------------------------------------------------------
        # Special embedding visibility (within causal window)
        # ----------------------------------------------------------------------------
        special_keys = special_embeddings_mask.view(bs, 1, k_len)  # (bs, 1, k_len)
        special_visible = full_causal_and_valid & special_keys  # (bs, q_len, k_len)

        # ----------------------------------------------------------------------------
        # Final visibility mask
        # ----------------------------------------------------------------------------
        allowed = block_causal | special_visible  # (bs, q_len, k_len)

        # ----------------------------------------------------------------------------
        # Convert boolean visibility to the floating mask expected by the model
        # ----------------------------------------------------------------------------
        score_val = torch.tensor(0.0, dtype=dtype, device=device)
        final_mask = torch.full((bs, 1, q_len, k_len), min_val, dtype=dtype, device=device)
        final_mask.masked_fill_(allowed.unsqueeze(1), score_val)

        # breakpoint()
        # torch.save(final_mask, "final_mask_partial.pt")
        if ft_with_bos_token:
            # TODO assert tokenizer is right padded
            final_mask[:, :, :, 0] = 0

        # torch.set_printoptions(profile='full', linewidth=10000)
        # print("final_mask", final_mask == 0)
        # print("final_mask", final_mask.shape)
        # breakpoint()

        return final_mask


# Mostly Copy paste of LlamaForCausalLM
class SentenceLlamaForCausalLM(SentenceLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def generate(self, *args, **kwargs):

        # if self.config.flexible_eos_tokens:
        #     if "logits_processor" in kwargs:
        #         raise ValueError("custom logits_processor is not supported for flexible_eos_tokens models")

        #     print("Force Flexible EOS Logits Processor")

        # kwargs["logits_processor"] = build_flexible_eos_logits_processors(self)

        input_ids = kwargs["input_ids"]
        assert input_ids.shape[0] == 1, "only batch size == 1 is supported"

        attention_mask = kwargs["attention_mask"]

        special_embeddings_mask = kwargs.get("special_embeddings_mask")

        if special_embeddings_mask is None:
            special_embeddings_mask = torch.zeros_like(attention_mask)
            if self.config.end_of_sentence_token_ids is not None:
                for end_of_sentence_token_id in self.config.end_of_sentence_token_ids:
                    special_embeddings_mask[input_ids == end_of_sentence_token_id] = 1

        clothest_end_of_sentence_token_idx = kwargs.get("clothest_end_of_sentence_token_idx")
        if clothest_end_of_sentence_token_idx is None:
            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                special_embeddings_mask,
                num_special_tokens=len(self.config.end_of_sentence_token_ids),
            )

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None:
            past_key_values = DynamicCache()

        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            assert past_key_values.get_seq_length() == 0, "cache_position must be provided if past_key_values is provided"
            cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)

        prev_attention_implementation = self.config._attn_implementation

        initial_input_ids = input_ids.clone()

        if input_ids.shape[1] > 1 and past_key_values.get_seq_length() == 0:
            # Prefill

            from sentence_attention.models.sentence_llama.scrooge_prefill import full_prefill_small_kv_cache

            # PREFILL_DEFAULT_ATTN_IMPLEMENTATION = "sentence_attention_flex"
            PREFILL_DEFAULT_ATTN_IMPLEMENTATION = "sentence_attention"
            self.config._attn_implementation = PREFILL_DEFAULT_ATTN_IMPLEMENTATION

            scrooge_prefill_outputs = full_prefill_small_kv_cache(
                self,
                input_ids.clone(),
                attention_mask.clone(),
                special_embeddings_mask.clone(),
                clothest_end_of_sentence_token_idx.clone(),
            )

            input_ids = scrooge_prefill_outputs["input_ids"]
            attention_mask = scrooge_prefill_outputs["attention_mask"]
            past_key_values = scrooge_prefill_outputs["past_key_values"]
            cache_position = scrooge_prefill_outputs["cache_position"]
            special_embeddings_mask = scrooge_prefill_outputs["special_embeddings_mask"]
            clothest_end_of_sentence_token_idx = scrooge_prefill_outputs["clothest_end_of_sentence_token_idx"]

        max_new_tokens = kwargs.get("max_new_tokens", 10)
        # do_sample = kwargs["do_sample"]
        assert "logits_processor" not in kwargs, "logits_processor is not supported for flexible_eos_tokens models"

        generated_tokens = [input_ids.item()]
        attention_mask_continuation = []
        special_embeddings_mask_continuation = []
        current_cache_position = cache_position.clone()

        current_input_ids = input_ids.clone()

        device = input_ids.device

        DECODE_DEFAULT_ATTN_IMPLEMENTATION = "sentence_attention"
        # Forse eager attention for decoding
        self.config._attn_implementation = DECODE_DEFAULT_ATTN_IMPLEMENTATION

        for _new_token_id in range(max_new_tokens):

            attention_mask_cont_t = torch.tensor([attention_mask_continuation], device=device, dtype=torch.long)

            current_attention_mask = torch.cat([attention_mask, attention_mask_cont_t], dim=-1)

            special_embeddings_mask_cont_t = torch.tensor(
                [special_embeddings_mask_continuation], device=device, dtype=torch.long
            )
            current_special_embeddings_mask = torch.cat([special_embeddings_mask, special_embeddings_mask_cont_t], dim=-1)

            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                current_special_embeddings_mask,
                num_special_tokens=len(self.config.end_of_sentence_token_ids),
            )

            model_out = self(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                special_embeddings_mask=current_special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                past_key_values=past_key_values,
                cache_position=current_cache_position,
                output_hidden_states=False,
            )

            current_input_ids = None

            # Logits processing
            if generated_tokens[-1] in self.config.end_of_sentence_token_ids:
                prev_token_is_eos_idx = self.config.end_of_sentence_token_ids.index(generated_tokens[-1])
                if prev_token_is_eos_idx < len(self.config.end_of_sentence_token_ids) - 1:
                    current_input_ids = torch.tensor(
                        [[self.config.end_of_sentence_token_ids[prev_token_is_eos_idx + 1]]],
                        device=input_ids.device,
                        dtype=torch.long,
                    )

            if current_input_ids is None:
                current_input_ids = model_out.logits[:, -1:].argmax(dim=-1)

            next_token_id = current_input_ids.item()
            generated_tokens.append(next_token_id)

            current_cache_position += 1
            # print("cache_position", current_cache_position)

            attention_mask_continuation.append(1)
            if next_token_id in self.config.end_of_sentence_token_ids:
                special_embeddings_mask_continuation.append(1)
            else:
                special_embeddings_mask_continuation.append(0)

            # if next_token_id == 1732:
            #     breakpoint()
            #     print("top tokens", model_out.logits[:, -1].argsort(dim=-1, descending=True)[:, :10])
            #     # SP tokens    : [430, 1364, 374, 264, 1695, 1732, 13, 220, 128256, 128257, 128258, 128259, 8100, 374, 264, 1695, 1732, 1606, 1364, 374, 3169]
            #     # Decode tokens: [430, 1364, 374, 264, 1695, 1732, 13, 220, 128256, 128257, 128258, 128259, 8100, 374, 264, 1695, 6691, 323, 264, 1695, 7555]
            #     print("generated_tokens", generated_tokens)

        if prev_attention_implementation is not None:
            self.config._attn_implementation = prev_attention_implementation
            print(
                "Restore attention implementation for decoding",
                "self.config._attn_implementation",
                self.config._attn_implementation,
            )

        generated_tokens_t = torch.tensor([generated_tokens], device=input_ids.device, dtype=torch.long)

        # print("initial_input_ids", initial_input_ids.shape)
        # print("generated_tokens_t", generated_tokens_t.shape)
        concatenated_tokens = torch.cat([initial_input_ids, generated_tokens_t], dim=-1)
        # print("concatenated_tokens", concatenated_tokens.shape)
        # breakpoint()
        if kwargs.get("return_dict_in_generate", False):
            return {
                "sequences": concatenated_tokens,
            }

        return concatenated_tokens

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        outputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs
        )

        input_ids = outputs["input_ids"]

        if "past_key_values" in outputs:
            if "special_embeddings_mask" not in outputs:
                special_embeddings_mask = torch.zeros_like(attention_mask)
            else:
                special_embeddings_mask = outputs["special_embeddings_mask"]
                if special_embeddings_mask.shape != attention_mask.shape:
                    special_embeddings_mask = torch.cat(
                        [outputs["special_embeddings_mask"], torch.zeros_like(input_ids)], dim=-1
                    )

            assert (
                special_embeddings_mask.shape == attention_mask.shape
            ), f"special_embeddings_mask.shape {special_embeddings_mask.shape} != attention_mask.shape {attention_mask.shape}"

            if self.config.end_of_sentence_token_ids is not None:
                special_tokens_count = 0
                for end_of_sentence_token_id in self.config.end_of_sentence_token_ids:
                    special_embeddings_mask[:, -input_ids.shape[1] :][input_ids == end_of_sentence_token_id] = 1
                    special_tokens_count += special_embeddings_mask.sum().item()
                    # if (input_ids == end_of_sentence_token_id).any():
                    #     print("input_ids", input_ids)
                    #     print("special_embeddings_mask", special_embeddings_mask)
                    #     breakpoint()
                # print("prepare_inputs_for_generation: number of end of sentence tokens", special_tokens_count)
                # print("prepare_inputs_for_generation: total_tokens", input_ids.shape[1])

            outputs["special_embeddings_mask"] = special_embeddings_mask
        else:
            # past_key_values is provided, so we don't need to recalculate special embeddings mask
            special_embeddings_mask = torch.zeros_like(attention_mask)
            if self.config.end_of_sentence_token_ids is not None:
                special_tokens_count = 0
                for end_of_sentence_token_id in self.config.end_of_sentence_token_ids:
                    special_embeddings_mask[:, -input_ids.shape[1] :][input_ids == end_of_sentence_token_id] = 1
                    special_tokens_count += special_embeddings_mask.sum().item()
            outputs["special_embeddings_mask"] = special_embeddings_mask

        # if "clothest_end_of_sentence_token_idx" not in outputs or need_recalc_special_embeddings_mask:
        # print('outputs["special_embeddings_mask"]', outputs["special_embeddings_mask"])
        outputs["clothest_end_of_sentence_token_idx"] = special_token_mask_to_clothest_token_idx_slow(
            outputs["special_embeddings_mask"],
            num_special_tokens=len(self.config.end_of_sentence_token_ids),
        )
        # print('outputs["clothest_end_of_sentence_token_idx"]', outputs["clothest_end_of_sentence_token_idx"])

        return outputs

    def _init_adaptive_layers(self):
        return self.model._init_adaptive_layers()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @replace_return_docstrings(output_type=AdaptiveCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @torch.compiler.disable(recursive=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        stop_words_tokens_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        is_sentence_chunked_prefill: bool = False,
        fused_linear_cross_entropy: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ) -> Tuple | SentenceCausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AdaptiveLlamaForCausalLM

        >>> model = AdaptiveLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print("use_cache", use_cache)

        # print("input_ids", input_ids)
        # print("cache_position", cache_position)
        # breakpoint()

        outputs: SentenceBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            stop_words_tokens_mask=stop_words_tokens_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            is_sentence_chunked_prefill=is_sentence_chunked_prefill,
        )

        hidden_states = outputs[0]

        if fused_linear_cross_entropy:

            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(hidden_states.device)
            # loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)

            assert num_items_in_batch is not None

            liger_lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")

            # Ensure 2D shape (B*T, H) even if the input accidentally becomes 2D or non-contiguous
            hidden_2d = hidden_states.flatten(0, 1)
            # print('lm_head_weight', self.lm_head.weight.shape, 'hidden_2d', hidden_2d.shape, 'shift_labels', shift_labels.shape)
            loss = liger_lce(self.lm_head.weight, hidden_2d, shift_labels)

            loss = loss / num_items_in_batch

            return SentenceCausalLMOutputWithPast(
                loss=loss,
                logits=None,
                last_hidden_state=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SentenceCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
