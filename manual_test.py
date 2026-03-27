#!/usr/bin/env python3
"""Manual generation tests for ruGPT-3 XL on GPU."""

import json
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_PATH = "./ruGPT3XL"

PROMPTS = [
    "Москва - столица",
    "Искусственный интеллект - это",
    "В далеком космосе",
    "Россия - это страна",
    "Программирование на Python",
    "Нейронные сети способны",
    "Однажды в студеную зимнюю пору",
    "Квантовая физика изучает",
    "История Древнего Рима",
    "Машинное обучение применяется",
    "Главной проблемой современного общества является",
    "Байкал - самое глубокое озеро",
    "Для приготовления борща нужно",
    "Теория относительности Эйнштейна",
    "Как стать хорошим программистом?",
    "В 2025 году технологии",
    "Русская литература XIX века",
    "Экономика Российской Федерации",
    "Философия Аристотеля утверждает",
    "Космическая станция МКС",
    "Вопрос: Какая столица России?\n\nОтвет:",
    "Вопрос: Сколько планет в Солнечной системе?\n\nОтвет:",
    "Вопрос: Кто написал 'Войну и мир'?\n\nОтвет:",
    "Вопрос: Что такое фотосинтез?\n\nОтвет:",
]


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {next(model.parameters()).device}")
    print()

    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Prompt: {prompt[:60]}...")
        t0 = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        elapsed = time.time() - t0
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        result = {
            "prompt": prompt,
            "generated": generated,
            "new_tokens": new_tokens,
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        results.append(result)
        print(f"  Generated ({new_tokens} tokens, {elapsed:.2f}s, {tokens_per_sec:.1f} tok/s):")
        print(f"  {generated[:200]}")
        print()

    with open("manual_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"All {len(results)} tests completed.")
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    print(f"Average speed: {avg_tps:.1f} tokens/sec")
    print("Results saved to manual_test_results.json")

    with open("manual_test_results.md", "w", encoding="utf-8") as f:
        f.write("# Manual Generation Tests - ruGPT-3 XL\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Device: NVIDIA RTX 4090 (GPU 1)\n")
        f.write(f"Precision: float16\n")
        f.write(f"Average speed: {avg_tps:.1f} tokens/sec\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Test {i}\n\n")
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Generated** ({r['new_tokens']} tokens, {r['time_sec']}s, {r['tokens_per_sec']} tok/s):\n\n")
            f.write(f"> {r['generated']}\n\n")
            f.write("---\n\n")

    print("Markdown report saved to manual_test_results.md")


if __name__ == "__main__":
    main()
