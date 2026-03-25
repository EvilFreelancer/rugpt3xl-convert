# ruGPT-3 XL: Megatron-LM to HuggingFace Conversion

This repository contains tools to convert the original [ai-forever/rugpt3xl](https://huggingface.co/ai-forever/rugpt3xl)
Megatron-LM/DeepSpeed checkpoint into a modern HuggingFace `transformers`-compatible format with custom model classes.

## Table of Contents

- [Background](#background)
- [Model Architecture](#model-architecture)
  - [High-Level Overview](#high-level-overview)
  - [Layer Structure](#layer-structure)
  - [Tokenizer](#tokenizer)
- [Original Checkpoint Format](#original-checkpoint-format)
  - [File Structure](#file-structure)
  - [State Dict Layout](#state-dict-layout)
- [Conversion Process](#conversion-process)
  - [Weight Mapping](#weight-mapping)
  - [QKV Projection Split](#qkv-projection-split)
  - [LM Head](#lm-head)
  - [Vocabulary Padding](#vocabulary-padding)
- [How to Convert](#how-to-convert)
  - [Prerequisites](#prerequisites)
  - [Step 1 - Download the Original Model](#step-1---download-the-original-model)
  - [Step 2 - Run Conversion](#step-2---run-conversion)
  - [Step 3 - Verify](#step-3---verify)
- [Output Structure](#output-structure)
- [Links](#links)

## Background

ruGPT-3 XL is a 1.3B-parameter GPT-3-style language model trained on Russian text by the SberDevices team
(now AI-Forever). It was trained using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for tensor
parallelism and [DeepSpeed](https://github.com/microsoft/DeepSpeed) for sparse attention, on 80B tokens
for 4 epochs with a 512 sequence length, then fine-tuned for 1 epoch at 2048 sequence length.

The original model was distributed as a raw Megatron-LM checkpoint (`mp_rank_00_model_states.pt`), which
requires the full Megatron-LM + DeepSpeed stack to load. This conversion eliminates that dependency entirely,
producing a self-contained HuggingFace model that loads with standard `AutoModelForCausalLM`.

## Model Architecture

### High-Level Overview

| Parameter | Value |
|---|---|
| Architecture | GPT-3 (decoder-only transformer) |
| Parameters | 1.3B (1,418,678,272 total) |
| Hidden size | 2048 |
| Layers | 24 |
| Attention heads | 16 |
| Head dimension | 128 |
| Intermediate (FFN) size | 8192 (4x hidden) |
| Max position embeddings | 2048 |
| Vocabulary size | 50,264 (50,257 base + 7 padding) |
| Activation | GELU (approximate, `gelu_new`) |
| Normalization | Pre-LayerNorm (LayerNorm before attention and FFN) |
| Position encoding | Learned absolute position embeddings |
| Precision | float16 |
| Test perplexity | 12.05 |

### Layer Structure

Each of the 24 decoder layers follows the pre-norm pattern:

```
Input
  |
  +---> LayerNorm -> MultiHeadAttention -> Dropout -> + (residual)
  |                                                    |
  +----------------------------------------------------+
  |
  +---> LayerNorm -> FFN(GELU) -> Dropout -> + (residual)
  |                                           |
  +-----------------------------------------+
  |
Output
```

**Attention block**: Separate Q, K, V linear projections (each `[2048, 2048]`), scaled dot-product
attention with causal mask, output projection, dropout.

**FFN block**: Up-projection `[2048, 8192]` -> GELU -> Down-projection `[8192, 2048]` -> Dropout.

The model uses a final LayerNorm after the last decoder layer, and a linear LM head that maps
hidden states back to vocabulary logits.

### Tokenizer

The model uses a BPE tokenizer (GPT-2 style) with the following special tokens:

| Token | String | ID |
|---|---|---|
| Pad | `<pad>` | 0 |
| EOS | `<\|endoftext\|>` | 1 |
| BOS | `<s>` | 2 |
| UNK | `<unk>` | 3 |

The base vocabulary contains 50,257 tokens from `vocab.json` and `merges.txt`.
The embedding matrix is padded to 50,264 (divisible by 8) for GPU memory alignment efficiency.

## Original Checkpoint Format

### File Structure

The original HuggingFace repository contains:

```
rugpt3xl/
  mp_rank_00_model_states.pt   # 2.6 GB - Megatron-LM checkpoint (float16)
  vocab.json                    # BPE vocabulary (50,257 tokens)
  merges.txt                    # BPE merge rules
  deepspeed_config.json         # DeepSpeed training configuration
```

### State Dict Layout

The `.pt` file is a standard PyTorch checkpoint. The state dict is nested under `module` key
(from the FP16 optimizer wrapper), then prefixed with another `module.` (from DistributedDataParallel):

```python
checkpoint = torch.load("mp_rank_00_model_states.pt")
state_dict = checkpoint["module"]  # top-level key
# Keys look like: "module.word_embeddings.weight", "module.transformer.layers.0...."
```

After stripping the `module.` prefix, the parameter names follow the Megatron-LM convention:

```
word_embeddings.weight                           # [50264, 2048]
position_embeddings.weight                       # [2048, 2048]
transformer.layers.{i}.input_layernorm.weight    # [2048]
transformer.layers.{i}.input_layernorm.bias      # [2048]
transformer.layers.{i}.attention.query_key_value.weight  # [6144, 2048] - combined QKV
transformer.layers.{i}.attention.query_key_value.bias    # [6144]
transformer.layers.{i}.attention.dense.weight    # [2048, 2048]
transformer.layers.{i}.attention.dense.bias      # [2048]
transformer.layers.{i}.post_attention_layernorm.weight   # [2048]
transformer.layers.{i}.post_attention_layernorm.bias     # [2048]
transformer.layers.{i}.mlp.dense_h_to_4h.weight # [8192, 2048]
transformer.layers.{i}.mlp.dense_h_to_4h.bias   # [8192]
transformer.layers.{i}.mlp.dense_4h_to_h.weight # [2048, 8192]
transformer.layers.{i}.mlp.dense_4h_to_h.bias   # [2048]
transformer.final_layernorm.weight               # [2048]
transformer.final_layernorm.bias                 # [2048]
```

## Conversion Process

### Weight Mapping

The conversion remaps Megatron-LM parameter names to the HuggingFace-style naming convention:

| Megatron-LM name | HuggingFace name |
|---|---|
| `word_embeddings.weight` | `model.embed_tokens.weight` |
| `position_embeddings.weight` | `model.embed_positions.weight` |
| `transformer.final_layernorm.*` | `model.norm.*` |
| `transformer.layers.{i}.input_layernorm.*` | `model.layers.{i}.input_layernorm.*` |
| `transformer.layers.{i}.attention.query_key_value.*` | `model.layers.{i}.self_attn.{q,k,v}_proj.*` (split) |
| `transformer.layers.{i}.attention.dense.*` | `model.layers.{i}.self_attn.o_proj.*` |
| `transformer.layers.{i}.post_attention_layernorm.*` | `model.layers.{i}.post_attention_layernorm.*` |
| `transformer.layers.{i}.mlp.dense_h_to_4h.*` | `model.layers.{i}.mlp.up_proj.*` |
| `transformer.layers.{i}.mlp.dense_4h_to_h.*` | `model.layers.{i}.mlp.down_proj.*` |

### QKV Projection Split

The most important transformation is splitting the fused QKV projection. Megatron-LM stores
Q, K, and V weights in a single `[3 * hidden_size, hidden_size]` matrix for efficiency:

```python
# Original: single [6144, 2048] weight
qkv_weight = state_dict["transformer.layers.{i}.attention.query_key_value.weight"]

# Split into three [2048, 2048] weights
q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
```

The same split applies to biases (`[6144]` -> three `[2048]` vectors).

### LM Head

The original model ties the LM head weight to the input word embeddings (`word_embeddings.weight`).
In the converted model, `lm_head.weight` is stored as an explicit copy of `model.embed_tokens.weight`
with `tie_word_embeddings=false` in the config. This ensures reliable loading without implicit
weight sharing, which can cause issues across different `transformers` versions.

### Vocabulary Padding

The original vocabulary has 50,257 tokens, but the embedding matrix is `[50264, 2048]`.
The extra 7 rows (indices 50,257-50,263) are zero-padded. This padding to a multiple of 8
is a standard Megatron-LM optimization for efficient matrix multiplication on GPUs.

## How to Convert

### Prerequisites

```bash
pip install torch transformers safetensors
```

### Step 1 - Download the Original Model

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/ai-forever/rugpt3xl

# Or using huggingface-cli
huggingface-cli download ai-forever/rugpt3xl --local-dir rugpt3xl
```

### Step 2 - Run Conversion

```bash
python convert.py \
    --input_path rugpt3xl/mp_rank_00_model_states.pt \
    --tokenizer_dir rugpt3xl \
    --output_dir rugpt3xl-hf
```

The script will:
1. Load the Megatron-LM checkpoint (~2.6 GB)
2. Strip the `module.` prefix from all parameter names
3. Auto-detect the number of layers and hidden size
4. Split fused QKV projections into separate Q, K, V weights
5. Remap all parameter names to HuggingFace convention
6. Clone the embedding weight for the LM head
7. Save as `model.safetensors` (~2.6 GB)
8. Copy model class files, config, and tokenizer files to the output directory

Expected output:

```
Loading checkpoint from rugpt3xl/mp_rank_00_model_states.pt ...
Detected 24 layers, hidden_size=2048
Converted 389 tensors
Saving weights to rugpt3xl-hf/model.safetensors ...
  Copied config.json
  Copied configuration_rugpt3xl.py
  Copied modeling_rugpt3xl.py
  Copied generation_config.json
  Copied vocab.json
  Copied merges.txt
  Copied tokenizer_config.json

Done! Model saved to rugpt3xl-hf
```

### Step 3 - Verify

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("rugpt3xl-hf", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("rugpt3xl-hf", trust_remote_code=True)

inputs = tokenizer("Москва - столица", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Output Structure

After conversion, the output directory contains everything needed to load the model:

```
rugpt3xl-hf/
  model.safetensors          # 2.6 GB - converted weights in safetensors format
  config.json                # model configuration with auto_map for custom classes
  configuration_rugpt3xl.py  # RuGPT3XLConfig class
  modeling_rugpt3xl.py       # RuGPT3XLModel + RuGPT3XLForCausalLM classes
  generation_config.json     # default generation parameters
  tokenizer_config.json      # tokenizer settings + chat template
  vocab.json                 # BPE vocabulary
  merges.txt                 # BPE merge rules
```

The model loads via `trust_remote_code=True` since it uses custom model classes not yet
registered in the `transformers` library.

## Links

- [A family of pretrained transformer language models for Russian](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=yPayeJIAAAAJ&citation_for_view=yPayeJIAAAAJ:Se3iqnhoufwC) - paper on Google Scholar
- [ai-forever/rugpt3xl](https://huggingface.co/ai-forever/rugpt3xl) - original model on HuggingFace
- [ai-forever/ru-gpts](https://github.com/ai-forever/ru-gpts) - original training codebase (Megatron-LM + DeepSpeed wrappers)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Megatron-LM framework used for training
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed framework used for sparse attention
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - target framework
- [HuggingFace Safetensors](https://github.com/huggingface/safetensors) - safe tensor serialization format

## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file in the repository root for details.
