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

## Qwen engine notes

`qwen` runs two architecture families from the same binary, selected by the
model's config.json:

- **Qwen3 dense** (0.6B–32B): GQA attention with per-head QK-RMSNorm, RoPE
  (theta from config), SwiGLU MLP, tied embeddings where the checkpoint uses
  them.
- **Qwen3.5 hybrid** (Qwen3-Next lineage, e.g. Qwen3.5-4B): `layer_types`
  mixes **Gated DeltaNet** linear-attention layers (recurrent state instead of
  a KV cache — memory does not grow with context) with **Gated Attention**
  full-attention layers (output gate, partial RoPE).

Memory: a 4B model needs ~16 GB RAM at f32; `QBITS=8` halves twice (~4.5 GB).
In chat mode the recurrent DeltaNet state is append-only: editing history
requires a full conversation reset (the engine does this automatically when
the context fills up).

### Validating against transformers

Token-parity validation needs a reference run from a machine with network
access and `transformers` installed:

```
python3 c/tests/make_ref.py Qwen/Qwen3-0.6B "The capital of France is" 24 > ref_qwen.json
REF=ref_qwen.json SNAP=/path/to/Qwen3-0.6B ./c/qwen     # expects full token match at f32
```

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
