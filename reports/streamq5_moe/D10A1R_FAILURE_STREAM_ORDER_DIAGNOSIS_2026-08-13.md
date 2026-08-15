# D10A1-R failure diagnosis — CUDA stream ordering

Verdict: **one common stream-order root cause explains all three failed gates**.
This is CPU/source analysis only; the negative D10A1-R result remains immutable.

## Header verifier

Only cases 12, 14, 15 and 28 reported positive byte counts: 151, 156, 73 and
111. Yet all 480 intended and actual header IDs and all 1,440 canary words were
exact in every one of the 40 cases, and candidate/oracle routed Q5 had zero bit
differences. The staging route was therefore correct.

`full_verify()` creates `cp.asarray(header_reference(...))` and
`cp.zeros(mismatches)` on CuPy's current/default stream, then launches
`verify_record_bytes` on the separately supplied nonblocking component stream.
There is no event or wait edge. Intermittent small positive byte counts are
therefore a partially visible header-reference H2D copy, not record corruption.

## GDN convolution state

The recurrent sample agreed exactly. The convolution allocation contains
1,179,648 BF16 words and the kernel addresses exactly 1,179,648 words, so this
is not a size/bounds error. At frozen step 0, the formula must write 292,608
nonzero BF16 words. The observed count was zero because `conv.fill(0)` ran on
the default stream and `gated_deltanet_step` ran on the nonblocking stream
without ordering; the zero-fill could race or overwrite the kernel writes.

## Shared Q5 oracle

96,256 of 98,304 values differed, leaving exactly 2,048 equal—one layer width.
The shared gate/up/down zero-fills were likewise unordered relative to shared
Q5 kernels. CPU hashing additionally verified that all three code+scale
payloads of every one of the 48 shared records equal the layer-0/expert-0
resident reference payload. The broadcast oracle is thus valid for this
invariant bank; the mismatch is scheduling-shaped, not layer payload variation.

## Repair boundary

D10A2 may change only stream ownership/order: all relevant allocations, H2D,
`asarray`, zeros/fills, kernels and readbacks must share one ordered stream or
be connected by explicit event/wait dependencies. It must retain routes,
kernels, math, correctness/performance gates and thresholds unchanged, and it
must verify payload equality before using the resident broadcast. No GPU repair
run is authorized by this diagnosis.
