#!/usr/bin/env python3
"""
Perplexity evaluation for ruGPT-3 models.

Measures perplexity using the standard LM approach: PPL = exp(avg cross-entropy loss).
Supports two strategies:
  - "non_overlapping" (Megatron-LM style): split text into chunks of seq_len
  - "strided" (HuggingFace style): sliding window with configurable stride

Usage examples:

  # Evaluate converted ruGPT3XL on a local text file
  python eval_perplexity.py --model_path ./ruGPT3XL --input_file test_data.txt

  # Evaluate on a HuggingFace dataset
  python eval_perplexity.py --model_path ./ruGPT3XL --dataset IlyaGusev/gazeta --split test

  # Compare original HF models with converted XL
  python eval_perplexity.py --model_path ai-forever/rugpt3large_based_on_gpt2
  python eval_perplexity.py --model_path evilfreelancer/ruGPT3XL

  # Use strided (overlapping) evaluation for tighter PPL estimate
  python eval_perplexity.py --model_path ./ruGPT3XL --strategy strided --stride 512
"""

import argparse
import math
import sys

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def load_model_and_tokenizer(model_path, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_path}")
    print(f"  dtype: {dtype}, device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[dtype],
    ).to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    return model, tokenizer


def load_text_from_file(path):
    print(f"Loading text from file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Characters: {len(text):,}")
    return text


def load_text_from_dataset(dataset_name, config, split, text_column, max_samples):
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} (config={config}, split={split})")
    ds = load_dataset(dataset_name, config, split=split, trust_remote_code=True)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"  Using first {max_samples} samples")

    if text_column not in ds.column_names:
        available = ", ".join(ds.column_names)
        print(f"  Column '{text_column}' not found. Available: {available}")
        for col in ds.column_names:
            if ds.features[col].dtype == "string":
                text_column = col
                print(f"  Auto-selected column: {text_column}")
                break

    texts = ds[text_column]
    text = "\n\n".join([t for t in texts if t and t.strip()])
    print(f"  Samples: {len(texts):,}, Characters: {len(text):,}")
    return text


def compute_perplexity_non_overlapping(model, encodings, seq_len, device, batch_size):
    """
    Megatron-LM style: split into non-overlapping chunks of seq_len,
    compute average CE loss, return PPL = exp(avg_loss).
    """
    input_ids = encodings["input_ids"].squeeze()
    total_len = input_ids.size(0)

    n_chunks = total_len // seq_len
    if n_chunks == 0:
        print("WARNING: text is shorter than seq_len, using full text as single chunk")
        n_chunks = 1

    truncated_len = n_chunks * seq_len
    input_ids = input_ids[:truncated_len].view(n_chunks, seq_len)

    print(f"  Total tokens: {total_len:,}")
    print(f"  Chunks: {n_chunks} x {seq_len} = {truncated_len:,} tokens")

    total_loss = 0.0
    total_tokens = 0

    loss_fct = CrossEntropyLoss(reduction="sum")

    for i in tqdm(range(0, n_chunks, batch_size), desc="Evaluating"):
        batch = input_ids[i : i + batch_size].to(device)
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits

        # Compute loss per sample to avoid OOM when casting full batch to float32
        for j in range(logits.size(0)):
            sample_logits = logits[j, :-1, :].float()
            sample_labels = batch[j, 1:]
            loss = loss_fct(sample_logits, sample_labels)
            total_loss += loss.item()
            total_tokens += sample_labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens


def compute_perplexity_strided(model, encodings, seq_len, stride, device):
    """
    HuggingFace style: sliding window with stride.
    Tokens in the overlap region use full context but only count loss on new tokens.
    """
    input_ids = encodings["input_ids"].squeeze()
    total_len = input_ids.size(0)

    print(f"  Total tokens: {total_len:,}")
    print(f"  Window: {seq_len}, Stride: {stride}")

    nlls = []
    total_tokens = 0
    prev_end = 0

    positions = list(range(0, total_len - 1, stride))
    for begin in tqdm(positions, desc="Evaluating"):
        end = min(begin + seq_len, total_len)
        target_len = end - prev_end

        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(chunk)
            logits = outputs.logits

        # Cast to float32 to avoid overflow in CE loss with float16 models
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Only count loss on the "new" tokens (after the overlap)
        token_losses = token_losses[-target_len + 1 :]
        nlls.append(token_losses.sum().item())
        total_tokens += token_losses.numel()

        prev_end = end
        if end >= total_len:
            break

    avg_loss = sum(nlls) / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for ruGPT-3 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="evilfreelancer/ruGPT3XL",
        help="Path or HF model id",
    )
    parser.add_argument("--input_file", type=str, help="Plain text file to evaluate on")
    parser.add_argument(
        "--dataset",
        type=str,
        default="IlyaGusev/gazeta",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="non_overlapping",
        choices=["non_overlapping", "strided"],
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length (default: model's max_position_embeddings)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for strided strategy",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = getattr(model.config, "max_position_embeddings", 2048)
    print(f"  Sequence length: {seq_len}")

    if args.input_file:
        text = load_text_from_file(args.input_file)
    else:
        text = load_text_from_dataset(
            args.dataset, args.dataset_config, args.split,
            args.text_column, args.max_samples,
        )

    print("Tokenizing...")
    encodings = tokenizer(text, return_tensors="pt")
    total_tokens = encodings["input_ids"].size(1)
    print(f"  Tokenized: {total_tokens:,} tokens")

    if total_tokens < seq_len:
        print(f"WARNING: text ({total_tokens} tokens) is shorter than seq_len ({seq_len})")
        seq_len = total_tokens

    print(f"\nComputing perplexity (strategy: {args.strategy})...")
    if args.strategy == "non_overlapping":
        avg_loss, ppl, n_tokens = compute_perplexity_non_overlapping(
            model, encodings, seq_len, args.device, args.batch_size,
        )
    else:
        avg_loss, ppl, n_tokens = compute_perplexity_strided(
            model, encodings, seq_len, args.stride, args.device,
        )

    print("\n" + "=" * 60)
    print(f"Model:            {args.model_path}")
    print(f"Strategy:         {args.strategy}")
    print(f"Sequence length:  {seq_len}")
    if args.strategy == "strided":
        print(f"Stride:           {args.stride}")
    print(f"Tokens evaluated: {n_tokens:,}")
    print(f"Avg CE loss:      {avg_loss:.6f}")
    print(f"Perplexity:       {ppl:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
