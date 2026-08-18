# S100 phase 9 repair findings

The original Phase-9 summary is not a negative result. The 8192-token route trace and five real expert-miss captures are valid, but the cache oracle crashed before producing capacity profiles and the RTX miss benchmark failed before executing.

Repair scope:

- reuse the frozen route trace and miss captures;
- correct and self-test the capacity dynamic program;
- allocate custom cache maps without transient uniform-cache duplication;
- benchmark staged ERVF against CUDA-mapped pinned DirectHost ERVF with >=4x-L2 rotations;
- benchmark the Arc miss engine under cold cache pressure;
- include exact-size CUDA bridge cost plus the independently measured Phase-8 Arc host-pointer touch cost;
- publish only results that pass parity, drift, VRAM and cold-working-set gates.

No original Phase-9 false promotion flag is treated as evidence against cache optimization, DirectHost or Arc miss racing.