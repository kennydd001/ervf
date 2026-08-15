# PORT80B-D10A1-R resource-budget audit

**Verdict:** the exact explicit route footprint is **14,452 records / 7,153,740 4-KiB pages / 27.289352 GiB**, but **~45.3 GiB is not a safe starting-RAM gate**.

The frozen runner selects exactly 40 correctness cases (all five domains, tokens 0..7) and exactly 32 validation cases (because the limit is reached by `general` tokens 512..543). Eight warm-ups reuse validation cases 0..7. The union also includes both wrong-source negative controls, 48 shared-expert copies and the ten-record layer-0 resident reference. Header/canary checks operate on staged HBM and add no bank pages.

## Exact explicit first-touch union

Each 2,027,520-byte expert record is exactly 495 aligned 4-KiB pages. The execution-order union is:

| source | set records | newly added records |
|---|---:|---:|
| correctness_40_full_record_reads | 11,329 | 11,329 |
| negative_control_overrides | 2 | 0 |
| validation_exact_32_full_record_reads | 6,953 | 3,075 |
| shared_48_full_record_copies | 48 | 48 |
| reference10_full_record_copy | 10 | 0 |


Total: **14,452 unique records = 29,301,719,040 bytes = 27.289352 GiB**. Of these, 14,024 are in the registered prefix and 380 are cold-tail records; the remainder is shared/reference overlap as classified by the exact union.

## Safe starting threshold

The route-only diagnostic formula is:

`29,301,719,040 explicit bank bytes + 1,073,741,824 host/process allowance + 2,147,483,648 post-touch reserve = 32,522,944,512 bytes (30.289352 GiB)`.

That lower number is **not safe to authorize**. D9 observed available RAM fall from 52,887,109,632 bytes before registration to 3,122,561,024 bytes after execution and clean unregister: a 46.346848-GiB delayed drop. The immediate post-registration sample saw only 104.496 MiB. Therefore the explicit route union cannot bound delayed Windows/CUDA registration residency.

The safety formula includes the whole 499-prefix registration footprint plus every explicitly touched record outside that prefix:

`49,430,937,600 bank-page bytes + 1,073,741,824 host/process allowance + 2,147,483,648 post-touch reserve = 52,652,163,072 bytes (49.036148 GiB)`.

The 1-GiB allowance exceeds both the frozen runner's enumerated bulk CPU arrays (116.566 MiB) and D9's immediate registration delta (104.496 MiB). The required 2-GiB post-touch reserve is unchanged.

**Recommendation:** do not lower the gate to 45.3 GiB. Require at least **49.036148 GiB** at the immediate `psutil.available` check; retaining the existing 50-GiB gate is the clean preregistration-compatible choice. Current live availability at this CPU audit was 48.361 GiB. No GPU operation, host registration, bank read or runner edit was performed.

This audit changes only resource interpretation. All correctness, performance, telemetry and cleanup gates remain frozen.
