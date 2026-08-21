# S100 Phase84-T1 — authoritative pack/H2D pipeline A/B

## Frozen question

Can a bounded producer plus non-blocking copy-stream pipeline reduce the real
fully-authoritative ctx64 verifier H4 wall time without changing target routes,
LRU52 decisions, miss count, transferred bytes or model arithmetic?

## Arms

- A (`baseline`): main thread copies each mmap expert into its unique pinned
  H4-ring slot and submits its segment H2D copies on the compute stream.
- B (`threaded_copy_stream`): one bounded CPU worker fills the next unique
  pinned slot while the main thread submits the current slot on a dedicated
  non-blocking copy stream. The compute stream waits on one event before any
  routed expert can consume those pages.

Both arms use the same committed 64-token target/reference trace, dynamic
target routers, physical-page LRU52 plans, six compressed expert segments,
global scales and complete integrated target verifier. There is no future
route, oracle prefetch, DFlash signal, speculation or cache-policy change.

## Measurement

Run two fresh unprofiled performance trajectories per arm in a new process.
Select repeat two as the warm host-page sample. Then run one fresh synchronized
validation trajectory outside performance timing. Report every performance
sample; do not use validation wall time as performance.

## Frozen gates

1. Both arms pass the strict Phase84 ctx64 sequence, route, head and finite gates.
2. Every performance/validation trajectory reproduces final-normalized bits,
   ERVF IDs, all 40 route sets and the full persistent-state SHA-256 digest.
3. Both arms have identical per-layer misses, total misses and H2D bytes.
4. The runtime copy ledger records zero D2D promotion bytes in every run.
5. Accept B only if warm primary wall improves by at least 5% or 20 ms over A.

Failure is informative: it means the existing asynchronous submissions already
hide CPU packing, copy-engine serialization dominates, or thread/copy-stream
overhead consumes the overlap. No result is output tok/s.
