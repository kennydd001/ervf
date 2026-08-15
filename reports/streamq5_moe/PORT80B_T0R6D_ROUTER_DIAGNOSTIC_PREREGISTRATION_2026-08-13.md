# PORT80B T0-R6-D router diagnostic preregistration

Status before execution: frozen diagnostic-only protocol; no outcome exists.

R6-D preserves the R5 shard, prompt 0, runtime locks, strict thresholds and loader. It performs exactly one official layer-0 forward, retaining the official router hook tuple and its input, followed by one independent call to the same router on that retained input. It builds no Q5 bank and cannot pass R4-REF or T0-P.

Before any adjudication it creates a raw safetensors file and JSON. For all 16 rows the JSON records official-versus-recomputed IDs/weights, FP32 and BF16 sum errors, finite/positive/monotonic/unique/bounds checks, native-BF16 rank-10/rank-11 logits and their uint16 bits, probability margin, the complete rank-10 boundary tie set, and the selected subset. `strict_margin_negative` is emitted iff any independently recomputed margin equals zero; otherwise the exact failed conjunct is named. No threshold may be changed and no retry is allowed.

Claim boundary: diagnostic evidence only. Even perfect internal agreement is `no_failure_reproduced`, never a scientific pass.
