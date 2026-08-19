# S100 Phase 20R — KV-scale + independent-reference repair

Target: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
Snapshot: `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`

## Adjudication

- Preflight exact-12 blocker: **True**
- KV-scale semantics: **False**
- Guarded production patch: **None**
- Target consumption: **None**
- Independent full reference: **None**
- Candidate/reference parity: **None**
- `PHASE20A_OFFICIAL_PARITY_GREEN`: **False**
- `PHASE20B_FULL_VERIFIER_OPEN`: **False**
- Next route: `REPAIR_KVSCALE_SEMANTICS`

## Attention KV scales

| layer | k_scale | v_scale | unit NRMSE | scaled NRMSE | pass |
|---:|---:|---:|---:|---:|---|
| 5 | 0.0316685251891613 | 0.0033307757694274187 | 0.03737743044559995 | 0.030547813908757715 | False |
| 12 | 0.0493861623108387 | 0.00847516767680645 | 0.0389123545878543 | 0.048160145928939566 | False |
| 19 | 0.064453125 | 0.01653180830180645 | 0.03869827840793091 | 0.033622332899700125 | False |
| 26 | 0.0770089253783226 | 0.0311104916036129 | 0.04477011541662014 | 0.04557709099406609 | False |
| 33 | 0.3370535671710968 | 0.0398995541036129 | 0.06677764343857706 | 0.11379490369617987 | False |
| 42 | 0.1071428582072258 | 0.0965401753783226 | 0.1118566677491585 | 0.136099869631934 | False |

## Independent reference

- Transformers: `None`
- Config class: `None`
- Model class: `None`
- Full model loaded: `None`
- Technical blocker: `None`

## Parity

- tokens: None
- top1: None
- top5: None
- mean CE delta: None
- mean coarse KL: None
- p95 coarse KL: None

## Claim boundary

20R only repairs and adjudicates Phase20A. It contains no full H=4 block-verifier timing and cannot claim S100.
