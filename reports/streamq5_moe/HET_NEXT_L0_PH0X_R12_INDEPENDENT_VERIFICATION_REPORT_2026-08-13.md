# PH0X-R12 independent verification

Date: 2026-08-13  
Mode: CPU-only and read-only for all experimental/model/device evidence

## Verdict

**PASS — 25/25 independent checks.**

The verifier independently rebuilt the selected official BF16 source range into
the 675,840-byte Q5 record, independently evaluated the exact FP32/BF16
projection DAG, and obtained oracle SHA-256
`e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867`.
The stored Intel host-USM output and the stored NVIDIA no-FTZ cubin output are
both byte-for-byte equal to that 512-word oracle.

## Independently reproduced evidence

- Official source range: 2,097,152 bytes, SHA-256 `05bd679b...`.
- Natural input range: 4,096 bytes, SHA-256 `5ce66a20...`.
- Q5 codes/scales/decoded digests: `20399f2c...`, `658d43f3...`,
  `9fd43163...`.
- Complete record: 675,840 bytes, CRC32 `1,976,639,022`, SHA-256
  `e3b10ab3...`.
- Independent strict oracle: 512 BF16 words, SHA-256 `e8a00c17...`.
- Intel evidence: 512/512 equal, 512 counters equal one, one enqueue, zero
  forbidden copies, exact 14-row allocation/release lifecycle, PCI
  `0000:00:02.0`.
- NVIDIA evidence: 512/512 equal, 512 counters equal one, exact 24-row cubin
  load/allocation/copy/launch/release ledger, one RawModule interception, PCI
  `0000:01:00.0`.
- Cubin: 62,319-byte ELF, SHA-256 `660c22ae...`.
- Every dependency recorded by the immutable R12 result still matches its
  recorded SHA-256.

## Development evidence remains immutable

R7 remains formally negative: its FTZ-compiled NVIDIA output differs from the
strict CPU/Intel oracle in 122/512 words. R10 remains a pre-launch failure:
the driver rejected its textual PTX with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
Neither result is overwritten or reclassified by the later R12 cubin success.

## Claim boundary

The supported result is only validation of **one official real Q5 projection
on one known natural activation** across an independently reconstructed CPU
oracle and stored Intel/NVIDIA physical outputs. It does not establish a full
expert, MoE, layer or model; held-out/generalized quality; cohabitation or
concurrency; timing or performance; deployment readiness; novelty; or an
industrial/LLM breakthrough.

Machine-readable verification:
`reports/streamq5_moe/het_next_l0_ph0x_r12_independent_verification.json`.

Independent verifier:
`scripts/streamq5_moe/verify_het_next_l0_ph0x_r12_completed.py`.
