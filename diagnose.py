#!/usr/bin/env python3
"""Diagnose conversion quality by comparing original Megatron checkpoint
with converted HuggingFace model using a manual forward pass."""

import math
import torch
import torch.nn.functional as F
import argparse


def load_original_checkpoint(path):
    print(f"Loading original checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("module", ckpt.get("model", ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v
    print(f"  Keys: {len(cleaned)}")
    return cleaned


def load_converted_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading converted model: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    return model, tokenizer


def megatron_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    ))


def megatron_forward(state_dict, input_ids, num_layers=24, num_heads=16):
    """Manual forward pass using original Megatron weights."""
    hidden_size = 2048
    head_dim = hidden_size // num_heads
    bsz, seq_len = input_ids.shape

    # Cast all weights to float32 for stability
    sd = {k: v.float() for k, v in state_dict.items()}

    # Token + position embeddings
    tok_emb = F.embedding(input_ids, sd["word_embeddings.weight"])
    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    pos_emb = F.embedding(pos_ids, sd["position_embeddings.weight"])
    hidden = tok_emb + pos_emb

    # Causal mask: [1, 1, seq, seq], 1.0 for attend, 0.0 for mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    for i in range(num_layers):
        pfx = f"transformer.layers.{i}"

        # Pre-norm
        ln_w = sd[f"{pfx}.input_layernorm.weight"]
        ln_b = sd[f"{pfx}.input_layernorm.bias"]
        normed = F.layer_norm(hidden, [hidden_size], ln_w, ln_b, 1e-5)

        # QKV projection
        qkv_w = sd[f"{pfx}.attention.query_key_value.weight"]
        qkv_b = sd[f"{pfx}.attention.query_key_value.bias"]
        qkv = F.linear(normed, qkv_w, qkv_b)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head
        q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        # Attention (dense, ignoring sparse for now)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores * causal_mask - 10000.0 * (1.0 - causal_mask)
        attn_probs = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        # Output projection
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        dense_w = sd[f"{pfx}.attention.dense.weight"]
        dense_b = sd[f"{pfx}.attention.dense.bias"]
        attn_out = F.linear(context, dense_w, dense_b)

        hidden = hidden + attn_out

        # MLP with pre-norm
        ln2_w = sd[f"{pfx}.post_attention_layernorm.weight"]
        ln2_b = sd[f"{pfx}.post_attention_layernorm.bias"]
        normed2 = F.layer_norm(hidden, [hidden_size], ln2_w, ln2_b, 1e-5)

        h4h_w = sd[f"{pfx}.mlp.dense_h_to_4h.weight"]
        h4h_b = sd[f"{pfx}.mlp.dense_h_to_4h.bias"]
        mlp_mid = megatron_gelu(F.linear(normed2, h4h_w, h4h_b))

        h2h_w = sd[f"{pfx}.mlp.dense_4h_to_h.weight"]
        h2h_b = sd[f"{pfx}.mlp.dense_4h_to_h.bias"]
        mlp_out = F.linear(mlp_mid, h2h_w, h2h_b)

        hidden = hidden + mlp_out

    # Final layer norm
    fn_w = sd["transformer.final_layernorm.weight"]
    fn_b = sd["transformer.final_layernorm.bias"]
    hidden = F.layer_norm(hidden, [hidden_size], fn_w, fn_b, 1e-5)

    # LM head (weight-tied with token embeddings)
    logits = F.linear(hidden, sd["word_embeddings.weight"])
    return logits


def compare_weights(orig_sd, model):
    """Compare key weights between original and converted."""
    print("\n=== Weight Comparison ===")

    hf_sd = model.state_dict()

    checks = [
        ("word_embeddings.weight", "model.embed_tokens.weight"),
        ("position_embeddings.weight", "model.embed_positions.weight"),
        ("transformer.final_layernorm.weight", "model.norm.weight"),
        ("transformer.final_layernorm.bias", "model.norm.bias"),
        ("transformer.layers.0.input_layernorm.weight", "model.layers.0.input_layernorm.weight"),
        ("transformer.layers.0.attention.dense.weight", "model.layers.0.self_attn.o_proj.weight"),
        ("transformer.layers.0.mlp.dense_h_to_4h.weight", "model.layers.0.mlp.up_proj.weight"),
        ("transformer.layers.0.mlp.dense_4h_to_h.weight", "model.layers.0.mlp.down_proj.weight"),
    ]

    all_ok = True
    for orig_key, hf_key in checks:
        orig_val = orig_sd[orig_key].float()
        hf_val = hf_sd[hf_key].float()
        diff = (orig_val - hf_val).abs().max().item()
        match = "OK" if diff < 1e-6 else "MISMATCH"
        if diff >= 1e-6:
            all_ok = False
        print(f"  {match}: {orig_key} -> {hf_key} | max_diff={diff:.2e} | shape {list(orig_val.shape)}")

    # QKV split check
    print("\n  QKV split check (layer 0):")
    qkv_w = orig_sd["transformer.layers.0.attention.query_key_value.weight"].float()
    q_w_orig, k_w_orig, v_w_orig = qkv_w.chunk(3, dim=0)

    q_w_hf = hf_sd["model.layers.0.self_attn.q_proj.weight"].float()
    k_w_hf = hf_sd["model.layers.0.self_attn.k_proj.weight"].float()
    v_w_hf = hf_sd["model.layers.0.self_attn.v_proj.weight"].float()

    for name, orig, hf in [("Q", q_w_orig, q_w_hf), ("K", k_w_orig, k_w_hf), ("V", v_w_orig, v_w_hf)]:
        diff = (orig - hf).abs().max().item()
        match = "OK" if diff < 1e-6 else "MISMATCH"
        if diff >= 1e-6:
            all_ok = False
        print(f"    {match}: {name}_proj | max_diff={diff:.2e} | shape {list(orig.shape)}")

    # lm_head check
    lm_head_w = hf_sd["lm_head.weight"].float()
    embed_w = orig_sd["word_embeddings.weight"].float()
    diff = (lm_head_w - embed_w).abs().max().item()
    match = "OK" if diff < 1e-6 else "MISMATCH"
    if diff >= 1e-6:
        all_ok = False
    print(f"\n  {match}: lm_head.weight == word_embeddings.weight | max_diff={diff:.2e}")

    return all_ok


def compare_forward(orig_sd, model, tokenizer):
    """Compare forward pass outputs."""
    print("\n=== Forward Pass Comparison ===")

    text = "Москва - столица"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"  Input: '{text}' -> ids: {input_ids[0].tolist()}")

    with torch.no_grad():
        # Original forward
        orig_logits = megatron_forward(orig_sd, input_ids)

        # Converted forward
        hf_outputs = model(input_ids)
        hf_logits = hf_outputs.logits.float()

    print(f"  Original logits shape: {list(orig_logits.shape)}")
    print(f"  Converted logits shape: {list(hf_logits.shape)}")

    diff = (orig_logits - hf_logits).abs()
    print(f"  Max absolute diff: {diff.max().item():.6f}")
    print(f"  Mean absolute diff: {diff.mean().item():.6f}")

    # Compare top predictions
    print("\n  Top-5 predictions for last token:")
    orig_top = orig_logits[0, -1].topk(5)
    hf_top = hf_logits[0, -1].topk(5)
    print(f"    Original:  {[tokenizer.decode([t]) for t in orig_top.indices.tolist()]}")
    print(f"    Converted: {[tokenizer.decode([t]) for t in hf_top.indices.tolist()]}")

    # Compute PPL for this short sequence
    for name, logits in [("Original", orig_logits), ("Converted", hf_logits)]:
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels)
        ppl = math.exp(loss.item())
        print(f"    {name} PPL on this input: {ppl:.2f}")

    return diff.max().item()


def main():
    parser = argparse.ArgumentParser(description="Diagnose ruGPT3XL conversion")
    parser.add_argument("--original", type=str, required=True, help="Path to mp_rank_00_model_states.pt")
    parser.add_argument("--converted", type=str, required=True, help="Path to converted model dir")
    args = parser.parse_args()

    orig_sd = load_original_checkpoint(args.original)
    model, tokenizer = load_converted_model(args.converted)

    weights_ok = compare_weights(orig_sd, model)
    max_diff = compare_forward(orig_sd, model, tokenizer)

    print("\n=== Summary ===")
    print(f"  Weights match: {'YES' if weights_ok else 'NO'}")
    print(f"  Max logit diff: {max_diff:.6f}")
    if max_diff < 0.01:
        print("  PASS: Conversion appears correct")
    elif max_diff < 1.0:
        print("  WARNING: Small numerical differences detected")
    else:
        print("  FAIL: Significant differences - conversion has bugs")


if __name__ == "__main__":
    main()
