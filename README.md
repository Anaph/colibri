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
| `gemma` | Gemma 4 (e.g. 12B-it, text-only) | Sliding/global hybrid, p-RoPE, GeGLU, SP tokenizer |

## Build

```
make            # builds all engines (from repo root or c/)
make glm        # just one engine
make portable   # portable CPU baseline (x86-64-v3 / armv8-a / power8)
make test       # test suite (GoogleTest via a separate CMake build path)
```

Requirements: a C compiler (gcc/clang) and GNU make. Linux, macOS, Windows
(MinGW/MSYS2), *BSD and PowerPC are supported. OpenMP is used when available.

The engines build with the C compiler alone. `make test` additionally needs
cmake ≥ 3.24 and a C++ compiler for the GoogleTest harness (test logic itself
is plain C; the C++ is confined to thin gtest glue). The first configure
downloads a pinned gtest via FetchContent unless a system GTest is installed.

## SIMD

The shared kernels in `c/simd.h` are selected at compile time by `-march`
(one `#ifdef` ladder, no runtime CPUID dispatch):

| Build | f32 kernel | int8 kernel |
| --- | --- | --- |
| `make portable` (x86-64-v3) | AVX2+FMA | AVX2 (maddubs) |
| `make portable-v4` (x86-64-v4) | AVX512F | AVX2 (v4 has no VNNI) |
| `make` / `ARCH=native` (x86) | AVX512F where present | +AVX512-VNNI on supporting CPUs |
| ARM (NEON baseline) | NEON | NEON; +dotprod (SDOT) with `ARCH=native` on supporting cores |

Quantized matmuls (`QBITS=8`) run the per-row Q8_0 scheme: activations are
quantized to int8 per row and the dot is pure integer (`dot_i8i8`). `IDOT=0`
selects the exact f32×int8 path instead — use it for byte-exact `REF`
comparisons (olmoe now shares the same int8 scheme, so its numerics also
shift slightly unless `IDOT=0`). The engines print the compiled tiers in
their startup banner (`idot ... | f32 ...`).

`make test-native` builds and runs the test suite with `-march=native`, so
the reference-based tests exercise the SIMD kernels your CPU actually has.

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
| `MEM_GB` | — | RAM budget in GiB: layers beyond the budget stream from disk each step |
| `MEM_FRAC` | — | same budget as a fraction (0..1) of total physical RAM; `MEM_GB` wins |
| `REF` | — | ref.json with prompt_ids/full_ids for greedy validation |
| `TOKENS` | 0 | 1 → dump generated token ids to stderr |
| `TTA` | off | **experimental** test-time adaptation: `cache` (neural cache) or `bias` (online logit bias); see [docs/online-learning.md](docs/online-learning.md) |
| `TTA_N` / `TTA_LAMBDA` / `TTA_THETA` / `TTA_LR` | 2048 / 0.1 / 1.0 / 0.1 | cache size, mix weight (capped at 0.5), similarity temperature, bias learning rate |

`TTA` (qwen only, default off — zero cost when unset) adapts predictions to
the text being generated: the neural cache mixes in a distribution over
recently seen continuations, the bias variant runs closed-form SGD on a
persistent logit bias. Adaptation state is cleared on every context reset
and REF validation mode structurally bypasses it.

`MEM_GB`/`MEM_FRAC` (qwen and gemma) trade speed for memory: the engine keeps
as many layers resident as fit the budget (embeddings, norms and recurrent
state always stay resident) and re-reads the remaining layers from the
safetensors on every step, prefetching the next layer while the current one
computes. Streamed layers always run f32 (`QBITS` applies to resident layers
only); token output is identical at any budget. Unset → everything resident.

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

### Validating against a reference (REF mode)

`REF=<file> SNAP=<snapshot> ./c/qwen` greedy-decodes and compares token ids
against a reference file, printing the match count (exit 0 on full match,
2 otherwise). The file format is plain JSON:

```json
{"prompt_ids": [151644, 872, ...], "full_ids": [151644, 872, ..., 785, 6722]}
```

`full_ids` must extend `prompt_ids`; the engine generates
`len(full_ids) - len(prompt_ids)` tokens greedily from `prompt_ids` and
requires an exact id-by-id match (f32 build). Produce the reference with any
tool that runs the original model — e.g. with `transformers`:

```python
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=24, do_sample=False, num_beams=1)
json.dump({"prompt_ids": ids[0].tolist(), "full_ids": out[0].tolist()}, open("ref.json","w"))
```

### Tokenizer parity (tok_oracle)

The test build also produces `c/tests/build/tok_oracle` — a corpus-scale
parity harness: `./tok_oracle <tokenizer.json> < cases.tsv` where each line
is `TEXT\tID,ID,...` (escapes: `\n \t \r \\`). Run it against ids produced
by the reference tokenizer before debugging model-level mismatches.

## Layout

```
c/glm.c      GLM-5.2 engine (single file)
c/olmoe.c    OLMoE reference engine
c/qwen.c     Qwen3 / Qwen3.5 engine
c/gemma.c    Gemma 4 engine (text-only)
c/nn.h       shared kernels: matmul f32/int8, quantization, sampler
c/st.h       safetensors loader (multi-shard, BF16/F16/F32)
c/tok.h      BPE tokenizer: byte-level and SentencePiece modes
c/json.h     minimal JSON parser
c/compat.h   cross-platform shims
c/tests/     test suite: C logic + GoogleTest glue (make test)
```

### Gemma engine notes

`gemma` runs the Gemma 4 text stack: sliding-window attention interleaved
with global layers (p-RoPE, optionally larger global head_dim), per-head
q/k/v RMSNorm, sandwich norms, GeGLU, optional KV-sharing / K=V / per-layer
embeddings — all config-driven. A few checkpoint conventions could not be
verified offline and sit behind loud probes (see VERIFY comments in
`c/gemma.c`); on first run against a real snapshot, resolve any reported
tensor-name/shape mismatch, validate the tokenizer with `tok_oracle`, then
gate with REF mode. `GEMMA_NORM_PLAIN=1` switches the RMSNorm convention
from `(1+w)` to `w` if REF parity points at the norm.

## License

See [LICENSE](LICENSE).
