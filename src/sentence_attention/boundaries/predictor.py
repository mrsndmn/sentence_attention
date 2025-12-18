import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, cast
from transformers.activations import ACT2FN

import math
import copy

from dataclasses import dataclass

from xlstm.xlstm_large.model import mLSTMLayer, mLSTMLayerConfig, mLSTMLayerStateType, soft_cap, mLSTMBackendConfig


class MaskState:
    def __init__(self, mask):
        self.cpu_mask = mask.cpu()

        self.mask = mask
        self.inv_mask = ~mask
        self._all = self.cpu_mask.all().item()
        self._any = self.cpu_mask.any().item()

    def any(self):
        return self._any

    def all(self):
        return self._all

    def selective_get(self, x, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return x[[]]
            elif not self.any():
                return x
            else:
                return x[self.inv_mask]
        else:
            if self.all():
                return x
            elif not self.any():
                return x[[]]
            else:
                return x[self.mask]

    def selective_put(self, x, out, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return
            elif not self.any():
                out.copy_(x)
            else:
                out[self.inv_mask] = x
        else:
            if self.all():
                out.copy_(x)
            elif not self.any():
                return
            else:
                out[self.mask] = x

    def selective_add(self, x, out, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return
            elif not self.any():
                out.add_(x)
            else:
                out[self.inv_mask] += x
        else:
            if self.all():
                out.add_(x)
            elif not self.any():
                return
            else:
                out[self.mask] += x


@dataclass
class BolmoConfig:
    # Model architecture
    hidden_size: int = 2048
    num_local_heads: int = 16
    num_local_encoder_layers: int = 2

    # MLP configuration
    intermediate_size: int = 8192
    local_intermediate_size: int = 2816
    hidden_act: str = "silu"

    # Normalization
    rms_norm_eps: float = 1e-6
    local_rms_norm_eps: float = 1e-5

    # Boundary prediction
    boundary_threshold: str = "sample:0"
    boundary_predictor_lookahead: int = 1
    balance_boundary_loss: bool = True


class BolmoXLSTMLayer(mLSTMLayer):
    def __init__(self, config: BolmoConfig):
        super().__init__(
            mLSTMLayerConfig(
                embedding_dim=config.hidden_size,
                num_heads=config.num_local_heads,
                mlstm_backend=mLSTMBackendConfig(
                    chunkwise_kernel="chunkwise--triton_limit_chunk",
                    sequence_kernel="native_sequence__triton",
                    step_kernel="triton",
                    mode="train",
                    return_last_states=True,
                    autocast_kernel_dtype="float32",
                ),
            )
        )

    # original forward adapted to support sequence_start_indices
    # i.e. set the forget gate to zero at the start of sequence
    def _original_forward(
        self,
        x: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        if self.config.weight_mode == "single":
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
            f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)
        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(self.ifgate_preact(x), cap_value=self.config.gate_soft_cap)
            i_preact, f_preact = torch.tensor_split(if_preact, (self.config.num_heads,), dim=-1)
        else:
            raise ValueError(f"Unknown weight_mode: {self.config.weight_mode}")

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)

        if sequence_start_indices is not None:
            f_preact[torch.arange(B, device=f_preact.device), sequence_start_indices] = -100_000

        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        h, state = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
        )
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert h.shape == expected_h_shape, f"Got {h.shape}, expected {expected_h_shape}"

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        past_key_values: Optional[dict] = None,
        use_cache: bool = False,
        sequence_start_indices: Optional[torch.Tensor] = None,
        cache_mask: Optional[MaskState] = None,
    ):
        if self.training:
            self.mlstm_backend.config.mode = "train"
        else:
            self.mlstm_backend.config.mode = "inference"

        if use_cache:
            assert past_key_values is not None

            prev_mode = self.mlstm_backend.config.mode
            state = past_key_values.get("state", None)

            if cache_mask is not None:
                state_for_model = cast(
                    mLSTMLayerStateType,
                    tuple(cache_mask.selective_get(x, inv=True) for x in state) if state is not None else None,
                )
            else:
                state_for_model = state

            h, new_state = self._original_forward(x, state=state_for_model, sequence_start_indices=sequence_start_indices)
            assert new_state is not None

            if state is None or cache_mask is None:
                state = new_state
            else:
                if cache_mask is not None:
                    for i in range(len(state)):
                        cache_mask.selective_put(new_state[i], state[i], inv=True)

            past_key_values["state"] = state
            self.mlstm_backend.config.mode = prev_mode

            return h
        else:
            h, _ = super().forward(x)
            return h


class BolmoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BolmoRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def compute_boundary_mask(boundary_logprobs: torch.Tensor, boundary_threshold: str) -> torch.Tensor:
    if boundary_threshold.startswith("sample:"):
        _, temperature = boundary_threshold.split(":")
        temperature = float(temperature)

        if temperature == 0:
            return boundary_logprobs > math.log(0.5)
        elif temperature == 1:
            return torch.bernoulli(torch.exp(boundary_logprobs)).to(torch.bool)
        else:
            raise NotImplementedError("Temperatures outside {0,1} are not implemented yet.")
    elif boundary_threshold.startswith("topk:"):
        _, topk = boundary_threshold.split(":")
        topk = int(topk)
        thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - (topk / boundary_logprobs.shape[1]))
        return boundary_logprobs >= thresholds.unsqueeze(-1)
    elif boundary_threshold.startswith("topk_percent:"):
        _, topk_percent = boundary_threshold.split(":")
        topk_percent = float(topk_percent)
        assert 0 <= topk_percent <= 1
        thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - topk_percent)
        return boundary_logprobs >= thresholds.unsqueeze(-1)
    else:
        raise ValueError(f"Unknown boundary threshold: {boundary_threshold}")


class BolmoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BolmoLocalLayer(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.act_fn = ACT2FN[config.hidden_act]

        self.xlstm = BolmoXLSTMLayer(config)

        local_mlp_config = copy.deepcopy(config)
        local_mlp_config.intermediate_size = config.local_intermediate_size
        self.mlp = BolmoMLP(local_mlp_config)

        self.pre_xlstm_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_start_indices: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        cache_mask: Optional[MaskState] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        xlstm_out = self.xlstm(
            self.pre_xlstm_layernorm(hidden_states),
            sequence_start_indices=sequence_start_indices,
            past_key_values=past_key_values["xlstm"] if past_key_values is not None else None,
            use_cache=use_cache,
            cache_mask=cache_mask,
        )
        hidden_states = residual + xlstm_out

        # Fully Connected
        residual = hidden_states
        ffn_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + ffn_out

        return hidden_states


class BolmoLocalEncoder(nn.Module):
    def __init__(self, config: BolmoConfig, subword_embedding):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.subword_embedding = subword_embedding

        self.layers = nn.ModuleList([BolmoLocalLayer(config) for _ in range(config.num_local_encoder_layers)])

        self.post_last_block_norm = BolmoRMSNorm(
            self.hidden_size,
            config.local_rms_norm_eps,
        )
        self.out_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
        )

        self.boundary_predictor_module = BolmoBoundaryPredictor(config)

    def freeze_non_embeddings(self):
        for param_name, param in self.named_parameters():
            if "subword_embedding" in param_name:
                continue

            param.requires_grad = False

    def forward(
        self,
        input_ids,
    ):
        embeddings = self.subword_embedding(input_ids)

        h = embeddings
        for _, block in enumerate(self.layers):
            h = block(h)

        boundary_logprobs, boundary_mask = self.boundary_predictor_module(h)

        return boundary_logprobs, boundary_mask


class BolmoBoundaryPredictor(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()

        self.d_model = config.hidden_size
        self.boundary_threshold = config.boundary_threshold
        self.boundary_predictor_lookahead = config.boundary_predictor_lookahead
        self.q_proj_layer = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj_layer = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, : -self.boundary_predictor_lookahead]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, self.boundary_predictor_lookahead :]), dim=-1),
        )
        boundary_logprobs = torch.log1p(-cos_sim.float().clip(max=1.0 - epsilon)) - math.log(2)
        POSITIVE_LOGPROB = 0.0
        NEGATIVE_LOGPROB = -100_000

        boundary_logprobs[:, 0] = POSITIVE_LOGPROB

        boundary_logprobs = F.pad(boundary_logprobs, (0, self.boundary_predictor_lookahead), "constant", NEGATIVE_LOGPROB)
        boundary_mask = compute_boundary_mask(boundary_logprobs, self.boundary_threshold)

        return boundary_logprobs, boundary_mask
