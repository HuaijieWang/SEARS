from contextlib import contextmanager
from types import MethodType

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


@contextmanager
def training_attention(model):
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    def padding_mask(_, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return attention_mask

    def forward(layer, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        batch, tokens, _ = hidden_states.shape
        if past_key_value is not None or use_cache or output_attentions:
            raise ValueError("Training attention expects a full sequence without a KV cache")
        q = layer.q_proj(hidden_states).view(batch, tokens, layer.num_heads, layer.head_dim).transpose(1, 2)
        k = layer.k_proj(hidden_states).view(batch, tokens, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
        v = layer.v_proj(hidden_states).view(batch, tokens, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
        cos, sin = layer.rotary_emb(v, seq_len=tokens)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        k = repeat_kv(k, layer.num_key_value_groups)
        v = repeat_kv(v, layer.num_key_value_groups)
        qkv = torch.stack([q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)], dim=2)
        if attention_mask is None:
            output = flash_attn_qkvpacked_func(qkv, causal=True)
        else:
            packed, indices, lengths, longest = unpad_input(qkv, attention_mask)
            attended = flash_attn_varlen_qkvpacked_func(packed, lengths, longest, causal=True)
            output = pad_input(attended, indices, batch, tokens)
        return layer.o_proj(output.reshape(batch, tokens, -1)), None, None

    decoder = model.model
    mask = decoder._prepare_decoder_attention_mask
    layers = [(layer.self_attn, layer.self_attn.forward) for layer in decoder.layers]
    decoder._prepare_decoder_attention_mask = MethodType(padding_mask, decoder)
    for layer, _ in layers:
        layer.forward = MethodType(forward, layer)
    try:
        yield
    finally:
        decoder._prepare_decoder_attention_mask = mask
        for layer, original in layers:
            layer.forward = original
