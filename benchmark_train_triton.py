#!/usr/bin/env python3
"""Test training loop for RuGPT-3 XL with SDPA (Flash/Triton backends) and optional torch.compile.

Uses synthetic batches to stress GPU memory and measure steps/sec.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from triton_utils import compile_rugpt3xl_for_triton, triton_runtime_available


def _find_max_batch(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    start_batch: int = 1,
    max_batch: int = 256,
) -> Tuple[int, Optional[str]]:
    """Binary search for largest batch that fits in VRAM."""
    work = model
    low, high = start_batch, max_batch
    best = start_batch
    last_err: Optional[str] = None
    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()
        gc.collect()
        try:
            opt = torch.optim.AdamW(work.parameters(), lr=1e-5)
            input_ids = torch.randint(
                0, vocab_size, (mid, seq_len), device=device, dtype=torch.long
            )
            labels = input_ids.clone()
            opt.zero_grad(set_to_none=True)
            out = work(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss
            if loss is None:
                raise RuntimeError("expected loss")
            loss.backward()
            opt.step()
            del out, loss, input_ids, labels, opt
            best = mid
            low = mid + 1
        except RuntimeError as e:
            last_err = str(e)
            high = mid - 1
        finally:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    return best, last_err


def _run_steps(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int,
    steps: int,
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    for _ in range(warmup):
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
        )
        labels = input_ids.clone()
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_loss = 0.0
    for _ in range(steps):
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
        )
        labels = input_ids.clone()
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        opt.step()
        total_loss += float(loss.detach())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples_per_sec = (steps * batch_size) / elapsed
    tokens_per_sec = (steps * batch_size * seq_len) / elapsed
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "warmup": warmup,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "avg_loss": total_loss / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("RUGPT3_MODEL", "./ruGPT3XL"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("RUGPT3_DEVICE", "cuda:0"),
    )
    parser.add_argument(
        "--attn",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="Maps to model.config.attn_implementation (sdpa uses scaled_dot_product_attention).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--find-max-batch",
        action="store_true",
        help="Search for largest batch (seq fixed) before timed run.",
    )
    parser.add_argument("--max-batch-search", type=int, default=256)
    parser.add_argument("--output-json", default="benchmark_train_triton.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This benchmark expects CUDA.")

    print(f"triton_available={triton_runtime_available()}")
    print(f"Loading model from {args.model_path} ...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map=None,
    )
    model = model.to(device)
    model.config.attn_implementation = args.attn

    vocab_size = model.config.vocab_size
    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    seq_len = min(args.seq_len, max_pos)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    train_model: torch.nn.Module = model
    if args.compile:
        print(f"Applying torch.compile (Inductor / Triton), mode={args.compile_mode} ...")
        train_model = compile_rugpt3xl_for_triton(
            model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=True,
        )

    results: Dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "triton_runtime": triton_runtime_available(),
        "attn_implementation": args.attn,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "seq_len": seq_len,
    }

    batch_size = args.batch_size

    if args.find_max_batch:
        print("Searching max batch size (may take a while)...")
        max_b, err = _find_max_batch(
            train_model,
            device,
            seq_len,
            vocab_size,
            start_batch=1,
            max_batch=args.max_batch_search,
        )
        batch_size = max_b
        results["find_max_batch"] = True
        results["max_batch_found"] = max_b
        results["find_max_batch_last_error"] = err
        print(f"max batch for seq_len={seq_len}: {max_b}")
        if max_b > 2:
            batch_size = max_b - 1
            results["batch_size_timed_run"] = batch_size
            print(
                f"timed run uses batch_size={batch_size} (max-1) to avoid post-search OOM"
            )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(
        f"Running batch_size={batch_size} seq_len={seq_len} "
        f"attn={args.attn} compile={args.compile} steps={args.steps} warmup={args.warmup}"
    )

    stats = _run_steps(
        train_model,
        device,
        batch_size,
        seq_len,
        vocab_size,
        args.warmup,
        args.steps,
    )
    results.update(stats)

    print(json.dumps(results, indent=2))
    out_path = os.path.join(os.getcwd(), args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
