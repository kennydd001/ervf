# PH1 Intel execution R1 — preregistration

Status: closed implementation. No preflight, payload or device call authorized.

R1 supersedes R0 after independent audit SHA-256 `1b701f0f5c4a4aab466688ded74da304e1d240006c37fff55855e6b120f036bf`. It preserves the compiled R2 source/binary and the 14 host-USM, 18 pointerarg, four launch and 21 release scientific contract while closing all eight audit clusters.

Before any OpenCL load R1 validates the complete compile and CPU package manifests/commits, independent PASS artifacts, generator, source/binary/buildlog, exact three records, natural input and LUT. It executes exactly 22 safe controls: truncation, wrong projection, stale CRC, CRC-recomputed code mutation, scale mutation, CRC-recomputed field31, wrong input for each record, plus wrong LUT digest. Each retains zero OpenCL/context/program/kernel/allocation/launch counters.

Authorization precedes every mutating recovery. A valid existing commit is recognized read-only and preserves its positive/negative exit semantics. Start available RAM must be at least 16 GiB; available RAM after payload and final must be at least 2 GiB; peak working set at all retained samples must not exceed 12 GiB; retained artifact bytes must not exceed 16 MiB.

Every release attempt is recorded before its API call with object identity, then code or exception and resulting ownership. All 21 calls are attempted. The final ledger must match exact allocation/argument/launch/finish/read/release tables, device identity, extension call counts, all-one counters and CPU-Q5 stage hashes. Forbidden-call counts are wrapper-owned observed counters.

The independent verifier imports neither runner, backend, common helper nor generator. It rereads official ranges and package artifacts, independently rebuilds the q+15 codec and exact width-8 integer/FMA oracle, reconstructs all five expected stages, and validates exact ordered ledger and provenance. Static preflight executes controls, codec fixtures, transaction/failure simulations, extension ABI checks and independent-verifier mutations. Claim remains one real expert/input Intel correctness component; no timing, full-layer/model or performance claim.
