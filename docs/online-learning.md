# Slow test-time learning in a pure-C inference engine

*Research note — feasibility of adapting model behaviour **during** inference
in colibrì's CPU engines (qwen, gemma). Companion to the `TTA=` prototype in
`c/qwen.c`.*

## 1. Scope and constraints

The question: can the engine *slowly fine-tune the model while it runs* —
adapting to the document, the conversation, or the user — without a training
framework, a GPU, or a second copy of the weights?

Constraints imposed by this codebase:

- **No autograd.** The engines are hand-written forward passes. Any gradient
  we use must be derived by hand, and the cost of hand-maintaining backprop
  through N layers grows with N. One layer is cheap; the whole stack is a
  rewrite.
- **Quantized base weights are not updatable in place.** With `QBITS=8` the
  weights live as `int8 + per-row scale`. Accumulating small gradient steps
  into int8 rounds to zero (or oscillates); the standard resolution is
  *adapters*: small f32 tensors that live beside the frozen base
  (LoRA-style), which is also what keeps the update memory bounded.
- **Layer streaming** (`MEM_GB`/`MEM_FRAC`) re-reads streamed layers from
  disk every step. Adapted state attached to a streamed layer would be lost
  or would force write-back; adapters must attach to **always-resident**
  tensors (embeddings, `lm_head`, norms — or dedicated buffers).
- **Single-token decode loop.** The natural training signal is next-token
  prediction on the text the model is actually processing: at position *t*
  the engine predicts, then observes the real next token *x*<sub>t+1</sub>
  (from the prompt, or its own sampled output). This is a full supervised
  pair per token, for free.

## 2. Taxonomy of test-time learning

### 2.1 Full fine-tuning — infeasible here

SGD on all weights needs gradients (1× weight memory) plus optimizer state
(1–2× more) plus a backward pass (~2× forward FLOPs). At 4B parameters that
is ≥32 GB of extra f32 state and triple the per-token compute — against the
engine's whole design. Ruled out.

### 2.2 Online LoRA on selected matrices

TLM-style perplexity-driven adaptation ([test-time learning survey
territory](https://arxiv.org/abs/2511.04847)) updates low-rank adapters
`W' = W + A·B` on a few matrices, minimizing the perplexity of the text
being read. Feasible in principle: adapters are f32, small (rank 8 on one
`down_proj`: `r·(D+I)` ≈ 120k floats), and the base stays frozen/quantized.
The cost is hand-written backprop through everything *above* the adapted
matrix — for a mid-stack adapter that is most of the model. Practical only
for matrices at the **top** of the stack (final layer, `lm_head`), where the
chain is one or two hops.

### 2.3 Fast weights — this engine already does it

The key observation of this note: **Qwen3.5's Gated DeltaNet layers — already
implemented in `c/qwen.c` (`deltanet_token`) — are fast-weight layers trained
online at inference time.** The recurrence

&nbsp;&nbsp;&nbsp;&nbsp;S ← α·S + β·k·(v − Sᵀk)ᵀ

is exactly the Widrow-Hoff / delta rule applied to an associative linear
memory S: each token performs one gradient step on the objective
‖Sᵀk − v‖², with learning rate β = sigmoid(b) and weight decay α = exp(g),
both *input-dependent*. This is the same mathematical object as TTT layers
([Sun et al., *Learning to (Learn at Test Time)*](https://arxiv.org/abs/2407.04620))
and the linear-attention/fast-weight equivalence of Schlag et al. The
engine's "KV-state that does not grow with context" *is* a per-layer neural
network being trained on the fly, 24 of them in Qwen3.5-4B.

Consequence: the interesting engineering question is not "add learning to
the engine" — it is **"add a *slow*, cross-forward-pass learning channel
above the built-in fast weights"**: something that persists across the whole
conversation (or across sessions) and moves orders of magnitude slower than
the per-token delta rule. That framing is the fast/slow-weights programme of
[Learning, Fast and Slow (arXiv 2605.12484)](https://arxiv.org/abs/2605.12484)
and [TTT-NTP (arXiv 2606.21803)](https://arxiv.org/abs/2606.21803), which
places a small fast weight inside selected MLP blocks updated from a
next-token signal — and it is the design axis of the prototype below.

### 2.4 Non-parametric adaptation: neural cache / kNN-LM

The [neural cache (Grave et al. 2016)](https://arxiv.org/abs/1612.04426) and
[kNN-LM (Khandelwal et al. 2019)](https://arxiv.org/abs/1911.00172) adapt
*without touching any weight*: store recent (hidden state → next token)
pairs and mix a similarity-weighted cache distribution into the model's
softmax. Properties that matter here:

- zero gradients, zero backprop, ~30 lines of C;
- bounded state (ring buffer of N entries — the "learning" is windowed, so
  drift cannot accumulate);
- strong documented wins exactly where a CPU engine spends long runs:
  repetitive/structured documents, code, long conversations (cache hit = the
  continuation was seen recently);
- composes with quantization and streaming trivially (reads only the final
  hidden state, which is always resident).

This was the top-scoring adaptation method per unit of complexity in every
practical LM-adaptation comparison until learned retrieval took over — and
learned retrieval is out of scope for a dependency-free C engine.

### 2.5 Closed-form last-layer gradients

For softmax cross-entropy, the gradient with respect to the `lm_head`
matrix W is available in closed form, no backprop:

&nbsp;&nbsp;&nbsp;&nbsp;∂CE/∂W = (p − e_x) · hᵀ &nbsp;&nbsp;(rank-1 per token)

where p = softmax(W·h), e_x the observed next token's one-hot, h the final
hidden state. Special cases, in ascending cost:

1. **Bias only** — add a persistent logit bias b, update
   `b ← b + η·(e_x − p)`. Memory: V floats (0.6 MB at V=151k). Cost: one
   vector op per token. This is exact SGD on the bias-augmented CE — the
   cheapest honest "the model is being trained while it runs" artifact.
2. **Rank-r lm_head adapter** — `logits += A·(Bᵀh)` with A[V,r], B[D,r];
   update both from the rank-1 gradient. Memory: `r·(V+D)` floats — at
   V=151k, r=8 that is ~4.8 MB *and* an A-update touching V·r floats per
   token; drift risk is real (see §4) for marginal benefit over bias+cache.
   Rejected for the prototype.
3. **Final-layer `down_proj` adapter** — needs the CE gradient chained
   through `lm_head` and the final RMSNorm (one hand-derived hop), plus a
   stash of the layer's activation. Feasible (~150 LOC), the natural "step
   2" after the prototype; the TTT-NTP placement (§2.3) is the mid-stack
   generalization.

### 2.6 What the field is converging on

Test-time training dominated the [ARC Prize 2024 leaderboard
(arXiv 2412.04604)](https://arxiv.org/abs/2412.04604) (every top LLM-based
transduction approach used TTT), and 2025–26 work extends it to continual,
perplexity-driven, and agentic settings. The recurring failure mode is also
documented: [self-amplification of the model's own sampled tokens
(arXiv 2604.21327)](https://arxiv.org/abs/2604.21327) — training on your own
outputs amplifies your own biases. Every design below treats "learning from
sampled tokens" as the dangerous half and "learning from prompt/document
tokens" as the safe half.

## 3. Cost model at Qwen3-4B scale

D = 2560, V ≈ 151k, forward ≈ 8 GFLOP/token (f32 resident). Per-token cost
of each option:

| Option | extra FLOPs/token | extra memory | gradients |
|---|---|---|---|
| Neural cache (N=2048) | ~5.2 M MAC (N·D dot) + V softmax | N·(D+4) f32 ≈ 20 MB | none |
| lm_head bias | ~2·V adds + softmax reuse | V f32 ≈ 0.6 MB | closed form |
| rank-8 lm_head LoRA | ~2·V·r + V·r update ≈ 3.6 M MAC | r·(V+D) ≈ 4.8 MB | closed form |
| final down_proj rank-8 | ~2·r·(D+I) + 1-hop chain | r·(D+I) ≈ 0.5 MB + activation stash | 1-layer hand backprop |
| online LoRA mid-stack | backprop through upper half ≈ +60–100% forward | adapters + activations | full chain |
| full FT | ≈ +200% forward | ≥ 8× weights | full chain |

All of the first four are <0.1% of the forward cost. The cliff is exactly
where hand-written backprop through the stack begins.

Interaction notes: with `QBITS=8` the base stays int8, adapters/cache are
f32 — the logits path is already f32, no conversion cost. With
`MEM_GB`-streaming, cache and bias attach to the final hidden/logits — both
always resident; mid-stack adapters would collide with streamed layers
(their layer's weights are re-read each step) and are additionally
disqualified there.

## 4. Risks

- **Self-amplification / repetition loops.** Cache and bias both boost
  recently seen tokens; feeding the model's own samples back as training
  signal is the documented failure mode (arXiv 2604.21327). Mitigations
  used in the prototype: interpolation weight λ is capped and constant (the
  cache can never dominate), bias steps are tiny and decay-free by default,
  and both are **off unless explicitly enabled**.
- **Distribution drift.** Windowed state (ring buffer) cannot drift; the
  bias can — it is the experimental knob, not the recommended one, and it
  resets with the conversation.
- **Prompt-injection persistence.** Adapted state outlives the text that
  created it: a malicious document could bias later answers. Mitigations:
  state is cleared by `state_reset` (context reset), never persisted to
  disk, and REF validation mode is structurally incapable of enabling TTA.
- **Validation.** REF-mode token-parity is defined on the frozen model; the
  shared `run_ref` path bypasses the TTA hooks entirely, so enabling the
  prototype cannot silently change validation results.

## 5. Recommendation and prototype

**Primary: neural cache** (`TTA=cache`) — gradient-free, windowed,
best-documented wins for the measurable setting, ~5 M MAC + 20 MB at 4B
scale, cleared by the existing reset lifecycle.
**Secondary: lm_head bias** (`TTA=bias`) — the ~25-line closed-form-gradient
demonstrator of §2.5.
**Rejected for now:** rank-r lm_head LoRA (§2.5.2, cost/drift), mid-stack
online LoRA (§2.2, backprop wall). **Natural next step** if the cache proves
itself: the final-layer `down_proj` adapter (§2.5.3), then TTT-NTP-style
placements (§2.3).

### Prototype specification (implemented in `c/qwen.c`)

Environment: `TTA=cache|bias` (default off — hooks compile to no-ops for
other engines and cost one branch when off), `TTA_N` (cache entries, default
2048), `TTA_LAMBDA` (mix weight λ, default 0.1), `TTA_THETA` (similarity
temperature θ, default 1.0), `TTA_LR` (bias step η, default 0.1).

Cache: after each decode step the final hidden state h_t (post-`model.norm`,
pre-`lm_head`) is L2-normalized and held; when the next token x_{t+1} is
fixed, the pair (ĥ_t, x_{t+1}) enters a ring buffer of N. At the next
prediction, s_i = θ·⟨ĥ_t, ĥ_i⟩ over the buffer,
p_cache(w) = Σ_{i: x_i = w} softmax(s)_i, and the model distribution is
mixed as `logits′ = log((1−λ)·p_model + λ·p_cache)`.

Bias: `logits′ = logits + b`; on observing x,
`b ← b + η·(e_x − softmax(logits + b))` — exact SGD on the bias-augmented
cross-entropy.

Wiring: `ENGINE_LOGITS_HOOK` (adjust logits after each `step`) and
`ENGINE_OBSERVE` (record the fixed next token) — no-op macros in
`runtime.h`, overridden only by qwen.c. `state_reset` clears all TTA state.
Prefill tokens are currently not cached (only the last hidden state is
exposed per step); exposing all hidden rows is listed as follow-up work.

Measured claims the test suite enforces (`qt_tta_*` in
`c/tests/qwen_tests.c`): TTA off is bit-identical to the engine without the
feature; λ=0 cache is bit-identical to off; the cache strictly increases
the probability of a continuation seen earlier in the sequence; on a
synthetic repetitive sequence the second-half cumulative −log p strictly
improves with the cache on — the self-supervised win, measurable without a
real checkpoint.

### `TTA=lora`: the §2.5.2 demonstrator, implemented after all

The rank-r lm_head adapter rejected above for the *default* path is now
implemented as a third mode, `TTA=lora` — the LoRA runtime and trainer
(`c/qwen_train.h`) made the machinery cheap to reuse. It keeps a rank-r
adapter `logits += (α/r)·B·(A·h)` with A[r,D] a fixed random projection
(deterministic init, α = 2r) and B[V,r] starting at zero, so a fresh
adapter is an exact no-op. On each observed token it runs closed-form SGD
on the CE gradient g = p − e_x: `B ← B − η·(α/r)·g⊗t` and
`A ← A − η·(α/r)·(Bᵀg)⊗h`, with t = A·h stashed from the adjust step and
Bᵀg computed *before* B moves. Cost per token: O(V·r) — the concern that
motivated the original rejection stands, which is why it remains opt-in.
Envs: `TTA=lora`, `TTA_RANK` (default 4, capped at 64), `TTA_LR` (default
1e-3 in this mode). `state_reset` zeroes B (adapter back to no-op; the
random A survives), same lifecycle as cache and bias. Tests enforce:
fresh/reset adapter leaves logits bit-identical; on a repeated token the
adjusted probability grows past the base model's.
