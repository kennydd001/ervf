# S100 Phase 16 preregistration

## 16A — local exact-state sensitivity
One routed-bank runtime. Candidate executes one transition from exact state,
then state is restored and the exact transition is replayed.

Scopes: attention all/K/V/O/Q, Mamba in/out, and every BF16 matrix.

Locally safe:
- top1 >= .995
- top5 = 1
- K16 = 1
- mean CE <= .01
- mean KL <= .005

## 16B — cumulative selected subset
Safe matrices ranked by Phase-14 B1 component saving.
Validation subsets N=1/2/4/8/16/all-safe.
Strict: top1 >= .970, top5 >= .999, mean CE <= .025,
mean KL <= .015, p95 KL <= .060.

## 16C — exact-state horizon
attention_all and selected safe subset. H=1/2/4/8.
H4 research-go: first-token >=.95, mean accepted prefix >=1.5,
full H4 match >=.25.

## 16D — selected subset speed accounting
Use measured Phase-14 per-matrix B1 times. Component accounting only.

## 16E — Mamba affine scan
Early/mid/late Mamba layers, H=8 real captured parameters.
Parallel prefix scan versus sequential production recurrence.
Green if all relevant NRMSE <=5e-5.

No Phase-16 result is an S100 claim.
