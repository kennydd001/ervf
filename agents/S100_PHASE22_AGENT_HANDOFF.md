# Phase22 handoff

Phase21 proved v6_device_rows is exact and 38.4% faster than the Python grouped
Phase20B scheduler. 71.65% of grouped routed experts are M=1.

Phase22 first removes graph loss and the suspicious 21.4 ms H4 lm_head before
building another grouped MoE scheduler.

FP32 KV graph attention copies the arithmetic of the production kernels:
- <=512 context: attn_decode with t read from device base_pos + row offset.
- >512: attn_decode_warp with identical split/chunk formula; fixed graph grid
  emits neutral inactive partials.

The graph remains fp8_kv=False.

If graph capture technically fails, report technical failure; do not treat the
graph hypothesis as falsified.
