# T0Q5-S0 source-audit checklist

No runner, model load, shard payload read, quantization or output is authorized yet.

1. Inspect the D2-R3 raw schema and name the exact whole-sequence keys available for MLP input, routes, experts/routed aggregate, shared raw, shared-gate linear, layer output, actual residual and complete MLP. Narrow claims/gates where exact reference tensors are absent; forbid subtraction reconstruction.
2. Confirm the selected union is derived only from all four whole routes, sorted unique, and all 32 rows are fixed validation rows.
3. Audit exact official full-16 `torch.where` gather order, fused gate-up, increasing-ID `index_add_`, BF16 casts and gate-first `sigmoid_gate * shared_raw` operand order.
4. Audit biased q+15 group-128 codec/zero/BF16-scale/little packing, independent source reread/requantization per selected matrix, and absence of persistent weight/code artifacts.
5. Confirm exact R3 metric math and unchanged thresholds, but require S0 `validation_positive` rather than pass.
6. Confirm deterministic controls have executable selection, true safe rejection, retained unsafe arrays and independent replay.
7. Confirm CPU/RAM/output-size gates and simple create-new atomic output/failure behavior.

Return GO only for standalone runner and independent verifier implementation. Do not authorize preflight or physical execution.
