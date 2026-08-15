# HET-NEXT-L0-C0-R4 — capability/preflight addendum

Design only. No executable preflight or device action is authorized.

The future C0-R4 Phase-0 preflight inherits C0-R3 design SHA-256 `6d50092b595ff635faa969ab3883617b8c26927819f684f4adebc5c420835035` and must additionally AST-check and TEMP-simulate:

1. cache-line-aligned per-worker `last_command_epoch_i` and `ack_epoch_i`, both initialized zero;
2. active-worker masks and total exclusion of inactive workers from descriptor, command, ready, done, ack and common-start-reset predicates;
3. acquire Interlocked ack check before descriptor/event reuse;
4. descriptor write under exclusive SRWLOCK, then last-command Interlocked publication, then command signal;
5. worker output synchronization and telemetry write, full barrier, ack Interlocked publication and immediately following done signal;
6. coordinator done wait, acquire ack equality check, then and only then output read/merge/reset;
7. common-start reset after exact acks from all and only prior-stage active workers;
8. the schedule prefix `A -> B -> S(Intel) -> S(NVIDIA) -> A`, plus stale ack, wrong epoch, premature reset/output read and 30-second timeout negative cases;
9. independent verifier source must reconstruct this lifecycle without importing runner state-machine helpers.

The later capability sentinel must prove the Interlocked/SRWLOCK/event primitives and frozen LP0/2/4/6 affinities work with exact cleanup, but cannot run until separately authorized after implementation audit.
