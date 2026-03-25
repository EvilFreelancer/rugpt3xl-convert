#!/usr/bin/env python3
"""Convert original Megatron/DeepSpeed ruGPT-3 XL checkpoint to HuggingFace format.

Converts the raw mp_rank_00_model_states.pt checkpoint from ai-forever/rugpt3xl
into a self-contained HuggingFace model directory with safetensors weights and
custom model classes.

Usage:
    python convert.py \
        --input_path path/to/mp_rank_00_model_states.pt \
        --tokenizer_dir path/to/rugpt3xl \
        --output_dir ./rugpt3xl-hf
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import save_file


def convert_megatron_checkpoint(input_path: str, output_dir: str, tokenizer_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if "module" in ckpt:
        state_dict = ckpt["module"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v
    state_dict = cleaned

    hidden_size = state_dict["word_embeddings.weight"].shape[1]
    num_layers = 0
    while f"transformer.layers.{num_layers}.input_layernorm.weight" in state_dict:
        num_layers += 1

    print(f"Detected {num_layers} layers, hidden_size={hidden_size}")

    new_state_dict = {}

    new_state_dict["model.embed_tokens.weight"] = state_dict[
        "word_embeddings.weight"
    ]
    new_state_dict["model.embed_positions.weight"] = state_dict[
        "position_embeddings.weight"
    ]

    new_state_dict["model.norm.weight"] = state_dict[
        "transformer.final_layernorm.weight"
    ]
    new_state_dict["model.norm.bias"] = state_dict["transformer.final_layernorm.bias"]

    for i in range(num_layers):
        prefix_old = f"transformer.layers.{i}"
        prefix_new = f"model.layers.{i}"

        new_state_dict[f"{prefix_new}.input_layernorm.weight"] = state_dict[
            f"{prefix_old}.input_layernorm.weight"
        ]
        new_state_dict[f"{prefix_new}.input_layernorm.bias"] = state_dict[
            f"{prefix_old}.input_layernorm.bias"
        ]

        qkv_weight = state_dict[f"{prefix_old}.attention.query_key_value.weight"]
        qkv_bias = state_dict[f"{prefix_old}.attention.query_key_value.bias"]
        q_w, k_w, v_w = qkv_weight.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_bias.chunk(3, dim=0)

        new_state_dict[f"{prefix_new}.self_attn.q_proj.weight"] = q_w
        new_state_dict[f"{prefix_new}.self_attn.q_proj.bias"] = q_b
        new_state_dict[f"{prefix_new}.self_attn.k_proj.weight"] = k_w
        new_state_dict[f"{prefix_new}.self_attn.k_proj.bias"] = k_b
        new_state_dict[f"{prefix_new}.self_attn.v_proj.weight"] = v_w
        new_state_dict[f"{prefix_new}.self_attn.v_proj.bias"] = v_b

        new_state_dict[f"{prefix_new}.self_attn.o_proj.weight"] = state_dict[
            f"{prefix_old}.attention.dense.weight"
        ]
        new_state_dict[f"{prefix_new}.self_attn.o_proj.bias"] = state_dict[
            f"{prefix_old}.attention.dense.bias"
        ]

        new_state_dict[f"{prefix_new}.post_attention_layernorm.weight"] = state_dict[
            f"{prefix_old}.post_attention_layernorm.weight"
        ]
        new_state_dict[f"{prefix_new}.post_attention_layernorm.bias"] = state_dict[
            f"{prefix_old}.post_attention_layernorm.bias"
        ]

        new_state_dict[f"{prefix_new}.mlp.up_proj.weight"] = state_dict[
            f"{prefix_old}.mlp.dense_h_to_4h.weight"
        ]
        new_state_dict[f"{prefix_new}.mlp.up_proj.bias"] = state_dict[
            f"{prefix_old}.mlp.dense_h_to_4h.bias"
        ]
        new_state_dict[f"{prefix_new}.mlp.down_proj.weight"] = state_dict[
            f"{prefix_old}.mlp.dense_4h_to_h.weight"
        ]
        new_state_dict[f"{prefix_new}.mlp.down_proj.bias"] = state_dict[
            f"{prefix_old}.mlp.dense_4h_to_h.bias"
        ]

    new_state_dict["lm_head.weight"] = new_state_dict[
        "model.embed_tokens.weight"
    ].clone()

    print(f"Converted {len(new_state_dict)} tensors")

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving weights to {safetensors_path} ...")
    save_file(new_state_dict, safetensors_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in [
        "config.json",
        "configuration_rugpt3xl.py",
        "modeling_rugpt3xl.py",
        "generation_config.json",
        "tokenizer_config.json",
    ]:
        src = os.path.join(script_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    for fname in ["vocab.json", "merges.txt"]:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    print(f"\nDone! Model saved to {output_dir}")
    print("Load with:")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_dir}", trust_remote_code=True)')


def main():
    parser = argparse.ArgumentParser(
        description="Convert Megatron ruGPT-3 XL checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Directory containing vocab.json and merges.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted model",
    )
    args = parser.parse_args()
    convert_megatron_checkpoint(args.input_path, args.output_dir, args.tokenizer_dir)


if __name__ == "__main__":
    main()
