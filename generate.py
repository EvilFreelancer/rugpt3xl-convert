#!/usr/bin/env python3
"""Simple text generation demo for the converted ruGPT-3 XL model."""

import argparse
import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="ruGPT-3 XL text generation demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="evilfreelancer/ruGPT3XL",
        help="Path to the converted model directory",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive multi-turn mode"
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path} ...")
    print(f"Device: {args.device}, dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
    ).to(args.device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    if args.interactive:
        run_interactive(model, tokenizer, args)
    elif args.prompt:
        run_single(model, tokenizer, args.prompt, args)
    else:
        prompts = [
            "Москва - столица",
            "Искусственный интеллект - это",
            "В далеком космосе",
        ]
        for prompt in prompts:
            run_single(model, tokenizer, prompt, args)
            print("-" * 60)


def run_single(model, tokenizer, prompt, args):
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}\n")


def run_interactive(model, tokenizer, args):
    print("Interactive mode. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not prompt:
            continue

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_marker = "Ответ: "
        if answer_marker in full_text:
            answer = full_text.split(answer_marker)[-1].strip()
        else:
            answer = full_text[len(text) :].strip()

        print(f"Model: {answer}\n")


if __name__ == "__main__":
    main()
