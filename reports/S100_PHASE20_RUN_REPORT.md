# S100 Phase 20 — Real Nemotron 3.5 Lightning

Status: **phase20a_blocked_target_consumption**

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
Snapshot: `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`
Architecture: `['NemotronHForCausalLM']` / `nemotron_h`

## 20A identity and schema

- Layers: 52 ({'attention': 6, 'mamba': 23, 'moe': 23})
- Tensor entries: 18487 across 52 shards
- Consumption gate: **False**
- Unknown unused target weights: **12**
- Expected-but-missing weights: **0**

## All 23 Mamba layers

- Status: `green`
- Tested layers: [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]
- Median/min/max speedup: 1.983218201916911 / 1.8127530119767155 / 2.0567680199941
- Maximum output NRMSE: 7.125779396646815e-06
- Maximum state NRMSE: 1.6254716489319885e-06
- Sabotage control: observable=True, max logit delta=11.097311019897461

## Independent reference parity

- Status: `blocked`
- `PHASE20A_OFFICIAL_PARITY_GREEN`: **False**
- Reason: ValueError: The checkpoint you are trying to load has model type `nemotron_h` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`

20B remains closed unless the independent parity gate is green. No S100 claim is made from this Phase 20A result.
