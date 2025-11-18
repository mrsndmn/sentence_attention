from typing import Tuple
from dataclasses import dataclass

import torch
from sentence_attention.models.sentence_llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    KwargsForCausalLM,
    Qwen2Config,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2MLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)


@dataclass
class SentenceBaseModelOutputWithPast(BaseModelOutputWithPast):
    moe_aux_loss: torch.Tensor | None = None


@dataclass
class SentenceCausalLMOutputWithPast(CausalLMOutputWithPast):
    last_hidden_state: torch.Tensor | None = None
    moe_aux_loss: torch.Tensor | None = None


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class SentenceQwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "SentenceQwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
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
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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


class SentenceQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = SentenceQwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class SentenceQwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

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
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to eager attention."
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        sliding_window = None
        if (
            getattr(self.config, "use_sliding_window", False)
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= getattr(self.config, "max_window_layers", 0)
        ):
            sliding_window = self.config.sliding_window

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
                sliding_window=sliding_window,
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
                sliding_window=sliding_window,
                **kwargs,
            )
        else:
            # Delegate to other registered attention implementations via ALL_ATTENTION_FUNCTIONS
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class SentenceQwen2Model(SentenceQwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        self.config._attn_implementation = "sentence_attention_flex"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SentenceQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple | SentenceBaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        assert self.config._attn_implementation in [
            "sentence_attention",
            "sentence_attention_flex",
        ], f"config._attn_implementation is expected to be 'sentence_attention', but got {self.config._attn_implementation}"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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
                for end_of_sentence_token_id in self.config.end_of_sentence_token_ids:
                    special_embeddings_mask[input_ids == end_of_sentence_token_id] = 1
                print("number of end of sentence tokens", special_embeddings_mask.sum().item())

        assert special_embeddings_mask is not None

        if clothest_end_of_sentence_token_idx is None:
            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(
                special_embeddings_mask,
                num_special_tokens=len(self.config.end_of_sentence_token_ids),
            )

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            clothest_end_of_sentence_token_idx,
            special_embeddings_mask,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
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
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    special_embeddings_mask=special_embeddings_mask,
                    clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = SentenceBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_aux_loss=None,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        clothest_end_of_sentence_token_idx: torch.Tensor,
        special_embeddings_mask: torch.Tensor,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # Flex Attention passthrough for generic flex masks
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor) and is_torch_flex_attn_available():
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if is_torch_flex_attn_available() and isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        if self.config._attn_implementation in ["sentence_attention"]:
            causal_mask = SentenceQwen2Model._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention(
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
            causal_mask = SentenceQwen2Model._prepare_4d_causal_attention_mask_with_cache_position_sentence_attention_flex(
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
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                config=self.config,
                past_key_values=past_key_values,
            )

        if (
            self.config._attn_implementation in ["sdpa", "sentence_attention", "eager"]
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
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
            final_mask[:, :, :, 0] = 0

        # torch.set_printoptions(profile='full', linewidth=10000)
        # print("final_mask", final_mask.shape, "\n", final_mask == 0)
        # breakpoint()

        return final_mask

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
        assert is_torch_flex_attn_available(), "Flex attention is not available in this environment"
        attention_mask_bool = attention_mask.bool()
        special_embeddings_mask = special_embeddings_mask.bool()

        assert len(attention_mask_bool.shape) == 2

        def mask_mod(b, h, q_idx, kv_idx):
            eos_token_idx = clothest_end_of_sentence_token_idx[b, q_idx]
            causal_mask = (kv_idx <= q_idx) & attention_mask_bool[b, q_idx] & attention_mask_bool[b, kv_idx]
            eos_sync_tokens = causal_mask & special_embeddings_mask[b, kv_idx]
            causal_triu_mask = causal_mask & (kv_idx >= eos_token_idx)
            return causal_triu_mask | eos_sync_tokens

        q_idx = sequence_length
        kv_idx = target_length

        block_mask = torch.nn.attention.flex_attention.create_block_mask(
            mask_mod,
            batch_size,
            None,
            q_idx,
            kv_idx,
            device=attention_mask.device,
            BLOCK_SIZE=128,
        )

        return block_mask


class SentenceQwen2ForCausalLM(SentenceQwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def generate(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        assert input_ids.shape[0] == 1, "only batch size == 1 is supported"

        attention_mask = kwargs.get("attention_mask")
        assert attention_mask is not None, "attention_mask must be provided"

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

        # Prefill for multi-token prompts to compress KV cache to EOS tokens
        if input_ids.shape[1] > 1 and past_key_values.get_seq_length() == 0:
            from sentence_attention.models.sentence_llama.scrooge_prefill import full_prefill_small_kv_cache

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
        assert "logits_processor" not in kwargs, "logits_processor is not supported for flexible_eos_tokens models"

        generated_tokens = [input_ids.item()]
        attention_mask_continuation = []
        special_embeddings_mask_continuation = []
        current_cache_position = cache_position.clone()

        current_input_ids = input_ids.clone()

        device = input_ids.device

        DECODE_DEFAULT_ATTN_IMPLEMENTATION = "sentence_attention"
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

            # Logits processing: force next EOS tokens sequence
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

            attention_mask_continuation.append(1)
            if next_token_id in self.config.end_of_sentence_token_ids:
                special_embeddings_mask_continuation.append(1)
            else:
                special_embeddings_mask_continuation.append(0)

        if prev_attention_implementation is not None:
            self.config._attn_implementation = prev_attention_implementation

        generated_tokens_t = torch.tensor([generated_tokens], device=input_ids.device, dtype=torch.long)
        concatenated_tokens = torch.cat([initial_input_ids, generated_tokens_t], dim=-1)

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
        outputs["clothest_end_of_sentence_token_idx"] = special_token_mask_to_clothest_token_idx_slow(
            outputs["special_embeddings_mask"],
            num_special_tokens=len(self.config.end_of_sentence_token_ids),
        )
        # print('outputs["clothest_end_of_sentence_token_idx"]', outputs["clothest_end_of_sentence_token_idx"])

        return outputs

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        clothest_end_of_sentence_token_idx: torch.Tensor | None = None,
        special_embeddings_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        is_sentence_chunked_prefill: bool = False,
        fused_linear_cross_entropy: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Tuple | SentenceCausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            special_embeddings_mask=special_embeddings_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
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
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

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
                moe_aux_loss=outputs.moe_aux_loss,
            )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_aux_loss=getattr(outputs, "moe_aux_loss", None),
        )
