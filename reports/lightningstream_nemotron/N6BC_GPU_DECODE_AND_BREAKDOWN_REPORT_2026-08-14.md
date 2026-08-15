# N6-B/C — GPU decode loop, metrical breakdown, context sweep

Datum: 2026-08-14
Verdict: **The runtime works and is coherent. Throughput FAILS both performance gates. Two named, fixable causes.**
Terminal state: `n6bc_runtime_coherent_performance_negative`

## 1. Werkt het?

Ja. The two halves are joined and the GPU runtime reproduces N6-A's CPU result.

```text
prompt      : "The capital of France is"
GPU top-1   : " Paris"          (N6-A CPU top-1: " Paris")
generated   : ' Paris." No extra punctuation? The sentence includes a period at the end. The'
```

Coherent English, correct fact, from the real NVFP4 checkpoint with the routed
experts streamed from host RAM. Shell load 2.0 s; routed bank pin 10.4 s
(one-time).

## 2. Antwoorden op de directe vragen

| vraag | antwoord |
|---|---|
| **iGPU of alleen RTX?** | **Alleen de RTX PRO 2000.** The Intel Arc Pro 140T is the protected HET-NEXT line's experiment and was never touched. |
| **Werkt het volledige context window?** | **262,144 draait fysiek** — allocated and stepping, 5.816 GiB device use. But the *architectural* limit is 262,144, **not 1M**; the 1M figure is a NIM service claim about a different payload. |
| **tok/s bij start?** | **13.258 tok/s** (ctx 0), 12.975 tok/s over a real 16-token generation. |
| **tok/s bij contextlimiet?** | **0.637 tok/s** at 262,100. |

## 3. Volledige metrische ontleding

At short context, per token, **every figure measured directly** — none obtained by subtraction:

| component | per call | count | total | share |
|---|---:|---:|---:|---:|
| **MoE layer (full)** | 2.501 ms | 23 | **57.531 ms** | **72.6%** |
| LM head | 5.383 ms | 1 | 5.383 ms | 6.8% |
| Mamba-2 layer | 0.313 ms | 23 | 7.205 ms | 9.1% |
| attention layer (ctx≈3) | 0.400 ms | 6 | 2.398 ms | 3.0% |
| RMSNorm | 0.040 ms | 53 | 2.120 ms | 2.7% |
| — sum of measured parts | | | 74.637 ms | 94.2% |
| — **full token measured** | | | **79.195 ms** | 100% |
| — unattributed | | | 4.549 ms | 5.7% |

The 4.549 ms residual is **left unnamed**. It is reported as a number, not as
"overhead" — the project rule after the "glue" term turned out to be attention.

Inside one MoE layer (2.501 ms): router 0.262 ms, shared expert 0.177 ms, so the
**six routed experts plus their transfer cost ≈ 2.06 ms**, i.e. ~47.4 ms/token.

## 4. Context sweep

KV cache populated synthetically and `pos` set, then one decode step timed. This
measures **decode cost at depth**; it is not a real generation to that depth, and
no quality claim attaches.

| context | p50 ms | tok/s |
|---:|---:|---:|
| 0 | 75.42 | **13.258** |
| 4,096 | 99.54 | 10.046 |
| 16,384 | 162.18 | 6.166 |
| 32,768 | 267.27 | 3.742 |
| 65,536 | 449.48 | 2.225 |
| 131,072 | 823.20 | 1.215 |
| 262,100 | 1568.91 | 0.637 |

## 5. Tegen de poorten — beide gefaald

| gate | vereist | gemeten | uitkomst |
|---|---:|---:|:--:|
| 4K decode, primary | ≥ 25 tok/s | 10.05 | ❌ |
| 4K decode, minimum acceptable | ≥ 20 tok/s | 10.05 | ❌ |
| 128K decode, primary | ≥ 15 tok/s | 1.215 | ❌ |
| 128K decode, minimum acceptable | ≥ 10 tok/s | 1.215 | ❌ |

**This is a performance negative, stated plainly.** The runtime is correct and
memory-feasible; it is not fast enough.

## 6. Twee benoemde oorzaken, beide repareerbaar

### 6.1 MoE transfer is niet overlapped — ~2× ligt op tafel

N4-R2 measured the complete 138-record routed path at **31.506 ms p50** with
causal H2D/compute overlap and bit-identical output. The runtime spends
**~47.4 ms** on the same work because `_moe` synchronises the copy stream **per
layer**: layer `L+1`'s six records are not prefetched while layer `L` computes.

The mechanism is already built, verified 34/34 and proven bit-identical. It is
not ported into the runtime yet. Closing that gap alone should move the token
from ~79 ms toward ~63 ms, i.e. roughly **13.3 → 16 tok/s**, without touching
semantics.

### 6.2 De attention-kernel serialiseert over posities

My `attn_decode` kernel runs **one block per query head** and walks all `t`
positions in a loop with a block-wide reduction and two `__syncthreads()` per
position. That is O(t) *serialised* per head. At 262,144 positions it is
~248 ms per attention layer.

The fix is standard flash-decoding: split the position range across many blocks,
compute partial (max, sum, acc) triples, then combine. The online-softmax
rescaling this kernel already implements is exactly what makes that split legal.
With 6 attention layers, 2 KV heads and 128 head_dim, there is a great deal of
idle parallelism to exploit.

**This is my kernel's limitation, not the architecture's.** The architecture is
unusually well suited to long context — N3 measured only 3,072 KV bytes per
token because just 6 of 52 layers carry a KV cache, and N5 confirmed 262,144
positions fit in 5.816 GiB total device use.

## 7. Wat wél bewezen is

1. **The joined runtime is coherent** — GPU output matches the frozen N6-A CPU result.
2. **262,144 context is physically real** — allocated, stepped, 5.816 GiB.
3. **The memory story holds** at the largest context, with the full 15.4 GiB routed bank host-resident and streamed.
4. **The bottleneck is measured, not guessed**: MoE 72.6% at short context, attention at long context.
5. **No iGPU, no protected-line interference.**

## 8. Eerlijk verdict

The line has produced a correct, memory-feasible, full-depth NVFP4 MoE runtime on
an 8 GiB laptop GPU that generates coherent text at **13.3 tok/s** and runs at a
**262,144-token context depth**. It has *not* met its throughput targets, and the
gap is not small: 2× short-context and roughly 10× at 128K.

Both causes are implementation, not physics, and both have a known remedy — one
of which this line already built and verified in N4-R2. That is the honest
position: **a working runtime with two identified engineering debts**, not a
breakthrough claim.

No figure here is extrapolated. The tok/s numbers are full-token measurements at
batch 1 on this specific GPU with no expert cache. A component measurement is
never promoted to tok/s.

## 9. Vervolg, in volgorde van verwachte winst

1. **Port N4-R2 overlap into `_moe`** — mechanism built and verified, ~2× at short context.
2. **Flash-decode split for `attn_decode`** — unlocks long context; the online softmax is already there.
3. **Expert cache (H5)** — N5 measured 572 free slots, but N6-A found all 128 experts used with only an 8.7× popularity spread, so the static prior must be re-derived, not inherited.
4. Only then: quality evaluation, thermal steady state, Dutch and long-context retrieval.

## Artefacten

- `scripts/lightningstream_nemotron/n6b_gpu_decode.py`, `reports/lightningstream_nemotron/n6b_gpu_decode.json`
- `scripts/lightningstream_nemotron/n6c_breakdown_and_context.py`, `reports/lightningstream_nemotron/n6c_breakdown_and_context.json`
- `src/moe_lab/lightningstream_nemotron/runtime.py`, `gpu_kernels.py`, `fused_nvfp4.py`
