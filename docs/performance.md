# CPU inference performance: caches, memory, and cores

*Research note for colibrì's dense engines (qwen, gemma). Companion to the
optimizations landed on this branch; numbers from `c/tests/build/bench` unless
stated otherwise.*

## 1. The memory-bandwidth wall

Single-token decode of a dense transformer reads **every weight byte exactly
once per token**: each matrix element participates in one multiply of a GEMV
and is never reused. Arithmetic intensity is ~0.5 FLOP/byte at f32 (2 FLOP per
4-byte weight) — orders of magnitude to the left of any CPU roofline knee. The
consequence is a hard ceiling:

&nbsp;&nbsp;&nbsp;&nbsp;**tok/s ≤ RAM bandwidth ÷ weight bytes per token**

At Qwen3-4B scale (~4B params):

| storage | bytes/token | 25 GB/s (2-ch DDR4) | 60 GB/s (2-ch DDR5) | 250 GB/s (Apple M-max) |
|---|---|---|---|---|
| f32 | ~16 GB | 1.5 tok/s | 3.7 | 15 |
| int8 (`QBITS=8`) | ~4 GB | 6 | 15 | 60 |
| int4 (deferred) | ~2 GB | 12 | 30 | 120 |

Everything else is noise at decode: KV-cache reads are ~288 KB/token at 4k
context (0.01% of weight traffic), activations and the DeltaNet S-state are
kilobytes. Cores beyond the point of bandwidth saturation add nothing —
`THREADS` sweeps plateau exactly where aggregate read bandwidth peaks.

**Prefill is the exception.** With S tokens in flight, each weight row can be
applied to all S activations per read — arithmetic intensity scales with S and
the workload becomes compute-bound. This asymmetry drives the batching work
below.

Cache-level view of the hot loops (4B dims): a weight row is 2.5–38 KB — it
transits L1/L2 once and is gone; the only *resident* data are the activation
vector (10–38 KB, stays hot in L1/L2 across all O rows) and, at prefill, the
S×I activation block. The K/V cache layout `[kv_head][t][hd]` streams
t-contiguously per head (full cache-line utilization). There is no blocking
scheme that makes decode weight traffic cacheable — the working set is the
model.

## 2. What this branch implements (measured)

Container: 4 shared cores, AVX512-VNNI, portable-build kernels for the table
(scalar tier — relative effects transfer; native-tier absolute numbers in
`bench` output).

1. **OpenMP hot-thread tuning + re-exec** (`runtime.h: omp_hot_tune`, ported
   from glm.c where it measured **66.9 s → 20.9 s** on a 32-core Zen5 matmul
   run). Dense decode enters ~250–290 tiny parallel regions per token; with
   the default passive wait policy the team goes to sleep between them.
   `OMP_WAIT_POLICY=active` + `GOMP_SPINCOUNT` + `OMP_PROC_BIND=close` +
   `OMP_DYNAMIC=FALSE` are seeded (never overriding user-set values) and the
   process re-execs once so libgomp picks them up. Kill-switch:
   `COLI_NO_OMP_TUNE=1`. On the 4-core noisy container the effect is neutral
   (spin competes with oversubscription); the win grows with core count.
2. **`THREADS=N`** — first thread-count knob in the engines; applied before
   load so weight quantization, scratch sizing and all regions obey it.
   `OMP_NUM_THREADS` still works.
3. **Per-model attention scratch** (`Model.att_sc`): the attention region
   used to `malloc`/`free` a per-thread score buffer on **every layer, every
   token** (2×36 allocations/token at 4B); now one buffer sized
   `threads × max_t` lives on the Model, indexed by `omp_get_thread_num()`
   (glm's pattern).
4. **Batched activation quantization** (`matmul_q_s`): the int8 GEMV now
   quantizes all S activation rows once and reads each weight row **once for
   all S tokens** inside a single parallel region. Measured on the container:
   34.5 → 50.9 → **58.9 GFLOP/s** at S = 1 → 8 → 64 with weight traffic
   constant — the prefill weight-reuse win, bit-identical numerics. The
   batched `mlp()` removes S× region re-forks per layer on top.
5. **DeltaNet single region**: conv + recurrence share one parallel region
   per token (was two). Minor; taken because it is free.

## 3. Deferred optimizations, cost/benefit at 4B

Ordered by expected value:

1. **int4 weights** — halves decode bytes again (≈2 GB/token → ~2× decode
   tok/s on the same RAM). The `dot_i4i8` kernel family already exists in
   glm.c (AVX512-VNNI/AVX2/NEON, validated bit-exact there); needed: lift into
   simd.h, an int4 packing path in `load_mat`/`quantize_rows`, `QBITS=4`.
   The single largest available win.
2. **Speculative decoding** — the structural escape from the wall: draft
   cheaply, verify K tokens in one batched forward (weight bytes amortize
   over K like prefill). Qwen3.5's cheap linear layers or an n-gram draft
   both fit; glm.c has a working MTP/n-gram speculation loop to model on.
3. **Weight interleave for VNNI** — reorder int8 rows so the dot kernel loads
   are perfectly sequential across the unrolled accumulators (llama.cpp /
   [Neural Speed](https://arxiv.org/abs/2411.19542)-style fused layouts
   reach >90% of bandwidth on INT4 GEMV). Moderate win over the current
   row-major int8 (already sequential per row); real gain appears with int4.
4. **NUMA placement** — first-touch or interleaved weight allocation +
   binding the team per socket; only matters on multi-socket / chiplet-split
   machines ([ArcLight](https://arxiv.org/abs/2603.07770) reports the
   cross-NUMA bottleneck dominating many-core CPU inference).
5. **Hugepages** — `MADV_HUGEPAGE` on the big weight buffers cuts TLB misses
   during streaming; single-digit % on Linux, ~10 lines.
6. **KV-cache quantization / paging** — irrelevant at 4k context (288
   KB/token) but becomes real at 100k+; the hybrid architectures (DeltaNet,
   sliding windows) already bound this structurally.
7. **Async prefetch of streamed layers** — the MEM_GB path already issues
   `WILLNEED` for layer i+1; a dedicated I/O thread
   ([async KV prefetching](https://arxiv.org/abs/2504.06319) analog) could
   overlap more aggressively.

## 4. LoRA training cost model

Training (TRAIN mode, f32 base required) is bandwidth-heavy in one place: the
chunked cross-entropy head. Per window of S tokens with chunk Sc, the lm_head
matrix (V×D ≈ 1.5 GB f32 at 4B) streams once forward and once backward per
chunk: ≈ 2·(S/Sc)·1.5 GB ≈ 50 GB per 512-token window at Sc=32 — a few
seconds at desktop bandwidth, dominating the window unless the trained-layer
count is large. The trained-layer backward is ~2× the forward FLOPs of those
layers; activation stash ≈ 124 MB per trained layer at S=512 (dominated by
the H×S×S attention probabilities). Adapter/optimizer state is megabytes.
Practical guidance: keep `TRAIN_CTX` moderate (256–512), raise
`TRAIN_CE_CHUNK` if RAM allows, and prefer few high-layers over many.
