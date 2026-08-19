# Phase23R handoff

Do not modify Phase23 kernels.

The only question is whether Phase23's physical byte reduction becomes a real
latency reduction after thermal/order bias is removed.

Frozen Phase23 evidence:
- exact GPU grouping and 24-route LRU;
- 31.16% fewer up weight streams;
- 26.65% fewer sparse-down bytes.

Balanced schedule:
R1 P,G
R2 G,P
R3 G,P
R4 P,G

All runs are fresh Python processes and time the same 16 advancing canonical
H4 positions after 8 warmup H4 blocks.

Promotion requires:
- median round gain >=5%;
- median 64 position-matched block-pair gain >=5%;
- >=3/4 rounds positive;
- robust run-median variability <=5% per arm;
- all correctness green.

Telemetry is diagnostic only.
