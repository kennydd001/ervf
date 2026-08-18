
# S100 phase 7 preregistration — isolated heldout recovery and packed downflow

Date frozen: 2026-08-17

## Part A — heldout recovery

The candidate set is exactly the `selected` object written before phase-6
heldout evaluation:

- thr_0003
- thr_0010
- thr_0015
- thr_0020
- k1
- k2
- thr0010_k1
- thr0010_k2
- thr0015_k1

No candidate may be added or removed.

Each candidate runs in a new process. The heldout trace and official gates are
unchanged. A failed process does not share a CUDA context with the next arm.

## Part B — exact packed backend

`PACKED` replaces only the down-code mirror representation.

Legacy order:

- panel chunks are assigned by `panel_list_index % nchunks`;
- mask bits are visited from low to high;
- chunk partials are reduced from chunk 0 upward;
- route contributions are accumulated from route 0 upward.

Packed preserves all four orders. It adds `panel_offset[p]`, the number of
selected columns before panel p, and uses this to address a contiguous
`[n_selected_columns, rowhalf]` byte buffer.

## Packed backend gates

Smoke:

- baseline A/B token parity;
- packed A/B token parity;
- packed equals legacy;
- finite output;
- bad-pick packed control diverges.

Full:

- all smoke gates remain supported;
- >=765 samples per ordinary arm;
- baseline and packed A/B drift <=1.0 ms;
- <=7.8 GiB;
- saving >=0.15 ms/token.

Otherwise legacy remains selected.

## Candidate timing

For every heldout-green candidate, fresh processes run:

1. BASE_A: legacy QFAST;
2. LEGACY_CAND: candidate on legacy;
3. CAND_A: candidate on selected exact backend;
4. CAND_B: deterministic repeat;
5. BASE_B: legacy QFAST.

When PACKED is selected, `LEGACY_CAND` and `CAND_A` must produce identical token
ids. This proves that packed adds no additional model change under threshold or
per-layer K.

## Finish

S100-single requires candidate midpoint <=10.000 ms and heldout fidelity green.
