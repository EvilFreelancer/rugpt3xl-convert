"""PyTorch RuGPT-3 XL model.

GPT-3-style decoder-only transformer (1.3B) trained on Russian text.
Architecture: absolute position embeddings, pre-norm layers, GELU activation,
tied LM head. Attention: config.attn_implementation "sdpa" uses
scaled_dot_product_attention (Flash/Memory-efficient/Triton backends on CUDA).
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_rugpt3xl import RuGPT3XLConfig

logger = logging.get_logger(__name__)


def _make_sparse_layout(
    num_heads: int,
    num_blocks: int,
    num_local_blocks: int,
    num_global_blocks: int,
    num_different_global_patterns: int,
    device: torch.device,
) -> torch.Tensor:
    """Build FixedSparsity boolean layout on *device*.

    Returns [num_heads, num_blocks, num_blocks] bool tensor.
    """
    layout = torch.zeros(
        num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device,
    )

    for win in range(0, num_blocks, num_local_blocks):
        end = min(win + num_local_blocks, num_blocks)
        sz = end - win
        layout[:, win:end, win:end] = torch.tril(
            torch.ones(sz, sz, dtype=torch.bool, device=device)
        )

    # Global attention (per-head: different heads use different global block positions)
    for h in range(num_heads):
        first = num_local_blocks - (
            1 + h % num_different_global_patterns
        ) * num_global_blocks
        reg_end = num_blocks - (num_blocks % num_local_blocks)
        for gi in range(first, reg_end, num_local_blocks):
            layout[h, gi:, gi : gi + num_global_blocks] = True
        if reg_end < num_blocks:
            s = min(reg_end + first, num_blocks - num_global_blocks)
            layout[h, s:, s : s + num_global_blocks] = True

    return layout


class RuGPT3XLAttention(nn.Module):
    def __init__(self, config: RuGPT3XLConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.output_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)

        attn_impl = getattr(self.config, "attn_implementation", "sdpa")
        use_sdpa = attn_impl == "sdpa" and not output_attentions

        if use_sdpa:
            dropout_p = self.attn_dropout.p if self.training else 0.0
            sdpa_mask = attention_mask
            if sdpa_mask is not None:
                sdpa_mask = sdpa_mask.to(dtype=query.dtype)
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_weights = self.attn_dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, value)
            attn_weights = attn_weights if output_attentions else None

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (
            attn_output,
            attn_weights if output_attentions else None,
            past_key_value,
        )


class RuGPT3XMLP(nn.Module):
    def __init__(self, config: RuGPT3XLConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.output_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.up_proj(hidden_states))))


class RuGPT3XLDecoderLayer(nn.Module):
    def __init__(self, config: RuGPT3XLConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = RuGPT3XLAttention(config, layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = RuGPT3XMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # Pre-norm: LayerNorm -> Attention -> Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm: LayerNorm -> MLP -> Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class RuGPT3XLPreTrainedModel(PreTrainedModel):
    config_class = RuGPT3XLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RuGPT3XLDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RuGPT3XLModel(RuGPT3XLPreTrainedModel):
    """Bare RuGPT-3 XL transformer outputting raw hidden states."""

    def __init__(self, config: RuGPT3XLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        self.layers = nn.ModuleList(
            [
                RuGPT3XLDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Sparse attention config
        self._sparse_layers: set = set()
        if getattr(config, "sparse_mode", "none") == "alternating":
            self._sparse_layers = {
                i for i in range(config.num_hidden_layers) if i % 2 == 0
            }
        elif getattr(config, "sparse_mode", "none") == "all":
            self._sparse_layers = set(range(config.num_hidden_layers))

        # Sparse layout will be lazily built on first forward.
        # NOT registered as a buffer to avoid meta-device corruption.
        self._sparse_layout: Optional[torch.Tensor] = None

        self.gradient_checkpointing = False
        self.post_init()

    def _get_sparse_layout(self, device: torch.device) -> torch.Tensor:
        """Return sparse layout tensor on *device*, building it if necessary."""
        if self._sparse_layout is not None and self._sparse_layout.device == device:
            return self._sparse_layout

        cfg = self.config
        num_blocks = cfg.max_position_embeddings // cfg.sparse_block_size
        self._sparse_layout = _make_sparse_layout(
            num_heads=cfg.num_attention_heads,
            num_blocks=num_blocks,
            num_local_blocks=cfg.sparse_num_local_blocks,
            num_global_blocks=cfg.sparse_num_global_blocks,
            num_different_global_patterns=cfg.sparse_num_different_global_patterns,
            device=device,
        )
        return self._sparse_layout

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = (
                input_ids.device if input_ids is not None else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeds = self.embed_positions(position_ids)
        hidden_states = self.embed_dropout(inputs_embeds + position_embeds)

        # Dense causal mask
        causal_mask = self._build_causal_mask(
            batch_size,
            seq_length,
            past_key_values_length,
            hidden_states.dtype,
            hidden_states.device,
            attention_mask,
        )

        # Sparse causal mask (lazily build layout on correct device)
        sparse_mask = None
        if self._sparse_layers:
            sparse_layout = self._get_sparse_layout(hidden_states.device)
            sparse_mask = self._build_sparse_causal_mask(
                seq_length,
                past_key_values_length,
                hidden_states.dtype,
                hidden_states.device,
                sparse_layout,
                self.config.sparse_block_size,
                attention_mask,
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask = (
                sparse_mask
                if (layer_idx in self._sparse_layers and sparse_mask is not None)
                else causal_mask
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1
                ]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @staticmethod
    def _build_causal_mask(
        batch_size: int,
        seq_length: int,
        past_length: int,
        dtype: torch.dtype,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_length = past_length + seq_length
        causal = torch.full(
            (seq_length, total_length),
            torch.finfo(dtype).min,
            device=device,
        )
        causal = causal.masked_fill(
            torch.arange(total_length, device=device).unsqueeze(0)
            <= torch.arange(
                past_length, past_length + seq_length, device=device
            ).unsqueeze(1),
            0.0,
        )
        causal = causal.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            pad_mask = (
                (1 - attention_mask[:, None, None, :].to(dtype))
                * torch.finfo(dtype).min
            )
            causal = causal + pad_mask

        return causal

    @staticmethod
    def _build_sparse_causal_mask(
        seq_length: int,
        past_length: int,
        dtype: torch.dtype,
        device: torch.device,
        sparse_layout: torch.Tensor,
        block_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build block-sparse causal mask from precomputed layout.

        Returns additive mask of shape [1, num_heads, seq_length, total_length].
        """
        total_length = past_length + seq_length
        num_blocks = sparse_layout.shape[1]

        q_block = (
            torch.arange(past_length, past_length + seq_length, device=device)
            // block_size
        ).clamp(max=num_blocks - 1)
        k_block = (
            torch.arange(total_length, device=device) // block_size
        ).clamp(max=num_blocks - 1)

        block_ok = sparse_layout[:, q_block][:, :, k_block]

        q_pos = torch.arange(
            past_length, past_length + seq_length, device=device
        ).unsqueeze(1)
        k_pos = torch.arange(total_length, device=device).unsqueeze(0)
        causal_ok = k_pos <= q_pos

        allowed = block_ok & causal_ok.unsqueeze(0)

        min_val = torch.finfo(dtype).min
        mask = torch.where(allowed, 0.0, min_val).to(dtype).unsqueeze(0)

        if attention_mask is not None:
            pad_mask = (
                (1 - attention_mask[:, None, None, :].to(dtype)) * min_val
            )
            mask = mask + pad_mask

        return mask


class RuGPT3XLForCausalLM(RuGPT3XLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _supports_cache_class = True

    def __init__(self, config: RuGPT3XLConfig):
        super().__init__(config)
        self.model = RuGPT3XLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[
                    :, -(attention_mask.shape[1] - past_length) :
                ]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if position_ids is not None and past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
