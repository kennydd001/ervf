# PH1 Intel execution R0 — implementation preregistration

Status: **closed source implementation; no static preflight, payload read or device call authorized**.

This component executes exactly the already compiled PH1 R2 OpenCL program against the immutable PH1 CPU package for one expert/input. It is correctness-only: no timing or performance claim.

Bindings include compile commit `c9f9ab3838d9d3d4ddd6e16a18f7989c16061901f08e046987081d9d975a152a`, compile result `ac7c90e15c71cf2a481004f78954e9d78631078d3e08893d3f716120345df5cc`, source `f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21`, binary `8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629`/186,352 bytes, and CPU-package commit `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06`.

The runner may read only the CPU package LUT, CPU raw-stage safetensors, exact three official source tensor ranges, and compiled source/binary after all locks/hashes pass. It reconstructs the same three 675,840-byte Q5 records in memory, matching record hashes gate `e3b10ab3...`, up `6da7025a...`, down `bd1a8ef9...`.

The Intel ledger contract is exact: 14 host-USM allocations totaling 2,185,216 bytes; 18 `clSetKernelArgMemPointerINTEL` calls; four kernel launches with `(global,local)` `(4096,256)`, `(4096,256)`, `(512,256)`, `(16384,256)`; one finish; five output plus four counter direct reads; and 21 reverse release attempts: 14 USM, four kernels, program, queue, context. All host-USM allocations require type=host, base=self, exact size and 4096-byte alignment. No `cl_mem`, enqueue read/write/copy, migrate or prefetch call is permitted.

Positive requires exact hashes of gate/up/SiLU/activation/down against the CPU-Q5 freeze, every uint32 counter exactly one, every output canary overwritten, finite BF16 words, clean 21-release ledger and zero live resources. Any mismatch is a valid negative; no retry or retuning. Create-new result/raw/manifest/commit are verified before and after write-through promotion; stale/corrupt/failed attempts are quarantined without changing a valid commit.
