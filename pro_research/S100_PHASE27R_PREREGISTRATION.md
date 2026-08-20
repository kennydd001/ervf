# S100 Phase27R preregistration

## Frozen source/evidence gates

Required:
- correct 52-layer Nemotron 3.5 Lightning snapshot;
- Phase24 active-parent adoption green;
- Phase27 instrumentation complete;
- Phase27 preflight green;
- Phase27 selected state green;
- Phase27 selected variant exactly:
  - gather_y=4
  - batches=3
  - shared_overlap=true
- Phase27 thermal adoption false;
- Phase27 final candidate A/B drift <=1%;
- Phase27 final parent A/B drift >7%.

No Phase27 source file may have uncommitted local changes.

## Primer

One fresh Phase24 parent process:

```text
context = 1024
warmup  = 0 H4 blocks
measure = 32 H4 blocks
```

The primer result is diagnostic and excluded from adoption statistics.

## Four balanced rounds

```text
R1 Parent    -> Candidate
R2 Candidate -> Parent
R3 Candidate -> Parent
R4 Parent    -> Candidate
```

Each arm is a new Python process.

Per process:

```text
context = 1024
warmup  = 8 H4 blocks
measure = 16 H4 blocks
```

## Statistics

Round gain:

```text
1 - candidate_process_median / parent_process_median
```

Position-paired gain:

```text
1 - candidate_block_ms / parent_block_ms
```

where the two records have the same canonical `pos`.

Robust process-median CV:

```text
1.4826 * MAD(process medians) / median(process medians)
```

## Adoption gate

`PHASE27_PIPELINE_ADOPTED=true` iff:

- all 8 scored runs status=measured;
- all 8 scored runs exact;
- every round has 16 parent and 16 candidate records;
- parent/candidate positions align in every round;
- median round gain >=0.05;
- median over 64 paired-block gains >=0.05;
- at least 3/4 round gains >0;
- parent robust CV <=0.05;
- candidate robust CV <=0.05.

## Promoted contexts

Only after adoption:

- context 128: 4 warmup + 12 measured H4 blocks;
- context 1024: 4 warmup + 12 measured H4 blocks;
- context 4096: 0 warmup + 12 measured H4 blocks.

## Speed gates

`TARGET_100_TARGET_ONLY_OPEN=true` iff every promoted context is
<=10.000 ms/useful token.

`DRAFTER_SHOOTOUT_OPEN=true` iff every promoted context is
<=8.000 ms/useful token.

`S100_SINGLE_ACHIEVED=false` always in Phase27R because no drafter,
accept/reject or fallback cost is included.
