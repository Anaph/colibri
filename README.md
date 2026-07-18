# colibrì

**Tiny engine, immense model.** Pure-C LLM inference with zero runtime
dependencies: no BLAS, no Python at runtime, no GPU required. Weights load
straight from HuggingFace safetensors snapshots.

Three standalone engines, one per architecture:

| Engine | Model family | Notes |
|---|---|---|
| `glm` | GLM-5.2 (744B MoE) | MLA attention, experts streamed from disk, ~25 GB RAM |
| `olmoe` | OLMoE | Reference GQA-MoE engine |
| `qwen` | Qwen3 dense / Qwen3.5 hybrid | GQA + QK-norm; Gated DeltaNet + Gated Attention |

## Build

```
make            # builds all engines (from repo root or c/)
make glm        # just one engine
make portable   # portable CPU baseline (x86-64-v3 / armv8-a / power8)
make test       # C unit tests
```

Requirements: a C compiler (gcc/clang) and GNU make. Linux, macOS, Windows
(MinGW/MSYS2), *BSD and PowerPC are supported. OpenMP is used when available.

## Run

Each engine is driven by environment variables and reads a HuggingFace
snapshot directory (config.json + tokenizer.json + *.safetensors):

```
# one-shot prompt
SNAP=/path/to/Qwen3-4B PROMPT="Hello!" ./c/qwen

# interactive chat (persistent KV cache)
SNAP=/path/to/Qwen3-4B ./c/qwen

# GLM-5.2 (see c/glm.c header for its full env reference)
SNAP=/path/to/glm-snapshot PROMPT="ciao" ./c/glm
```

Common environment variables (qwen engine):

| Var | Default | Meaning |
|---|---|---|
| `SNAP` | — | model snapshot directory (required) |
| `PROMPT` | — | one-shot prompt; if unset, interactive chat on stdin |
| `NGEN` | 256 | max new tokens |
| `CTX` | 4096 | context length |
| `TEMP` / `NUCLEUS` / `SEED` | 0.7 / 0.95 | sampling (TEMP=0 → greedy) |
| `CHAT_TEMPLATE` | 1 | wrap prompt in the model's chat format |
| `THINK` | 0 | Qwen3 thinking mode (0 pre-closes the think block) |
| `QBITS` | 0 | 8 → int8-quantize weights at load (~2.5× less RAM) |
| `REF` | — | ref.json with prompt_ids/full_ids for greedy validation |
| `TOKENS` | 0 | 1 → dump generated token ids to stderr |

## Layout

```
c/glm.c      GLM-5.2 engine (single file)
c/olmoe.c    OLMoE reference engine
c/qwen.c     Qwen3 / Qwen3.5 engine
c/st.h       safetensors loader (multi-shard, BF16/F16/F32)
c/tok.h      byte-level BPE tokenizer (tokenizer.json)
c/json.h     minimal JSON parser
c/compat.h   cross-platform shims
c/tests/     C unit tests (make test)
```

## License

See [LICENSE](LICENSE).
