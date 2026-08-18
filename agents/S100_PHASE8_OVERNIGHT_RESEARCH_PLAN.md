# S100 phase 8 — overnight Arc 140T adjudication

Date frozen: 2026-08-17.

The overnight campaign decides whether Arc 140T should enter the latency-critical
QFAST routed-down path. It compares current RTX H-SCALE downflow with Arc over
the same real QFAST geometry and expert records.

## Real-data coverage

Four deterministic causal snapshots are captured after decode offsets 0, 1, 4,
and 16. Each snapshot contains all 23 MoE layers and their six actual selected
experts, route weights, ReLU2 activations, masks and panel-major NVFP4 records.

The existing strict Arc kernel harness then tests N={1,2,4,6}, local sizes
{64,128,256}, strict and fast-math on every layer of every snapshot. Only strict
results may promote.

## Direct comparator

The current QFAST RTX down path is measured on the same snapshot offsets using
the live phase-5 closure: scale-resident state, sparse column gather, masked down,
chunk reduction and route-weight accumulation. This removes the largest remaining
proxy assumption from phase 8.

## Full-bank anti-cache control

A tiny six-expert buffer can produce a false win if those records stay in Arc caches.
One complete real down-expert bank from the middle MoE layer is therefore kept in a
single persistent Intel-GPU buffer. Random six-expert route sets span the whole bank,
with a 128 MiB cache scrub before timed kernels. The measured cold/warm factor is
applied conservatively to the small-snapshot Arc totals during adjudication.

## Stability

- route/cache census: 8192 causal tokens;
- bridge: repeated blocks across the night;
- QFAST/Arc interference: repeated BASE/LOAD/BASE rounds;
- real NVFP4 Arc re-runs continue until the wall deadline;
- first-vs-last performance drift is reported.

## Frozen decision

ADE_GO requires:

1. every strict N=6 real-NVFP4 snapshot/layer correctness gate green;
2. pressure-adjusted Arc N=6 kernel + conservative bridge all-layer sum >=10%
   faster than measured RTX down-only all-layer sum, with no negative snapshot;
3. median QFAST regression under a real full-bank Arc load <=5%;
4. first-vs-last Arc median regression <=10%;
5. at least three independent real-kernel reruns.

ADE_NO_GO if strict correctness fails, Arc is >=10% slower, or sustained QFAST
regression exceeds 15%. Everything else is ADE_BORDERLINE.

No component projection counts as S100. GO opens end-to-end integration.
