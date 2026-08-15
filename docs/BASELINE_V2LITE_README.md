# MoE Compiler Lab

Reproduceerbaar onderzoek naar activation-space compressie van routed experts.
De eerste teacher is `deepseek-ai/DeepSeek-V2-Lite` (Base). DeepSeek V4 Flash is
pas een schaaltest nadat V2-Lite de vooraf vastgelegde gates haalt.

## Onderzoeksvolgorde

1. Leg hardware, software en vrije opslag vast.
2. Pin de officiële V2-Lite-revisie en download alleen metadata/code.
3. Valideer traceformaat en metrieken met een kleine synthetische MoE.
4. Download V2-Lite BF16 pas na de preflight.
5. Meet eerst één echte MoE-laag; schaal daarna pas naar alle 26 MoE-lagen.

De synthetische test bewijst **niet** dat expertcompressie werkt. Hij bewijst
alleen dat routing, aggregatie, baselines, metrieken en rapportage correct lopen.

## Installatie (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --editable .
```

## Baseline uitvoeren

```powershell
.\.venv\Scripts\python.exe scripts\collect_hardware.py
.\.venv\Scripts\python.exe scripts\probe_runtime.py
.\.venv\Scripts\python.exe scripts\fetch_v2_lite.py --metadata-only
.\.venv\Scripts\python.exe scripts\verify_checkpoint.py
.\.venv\Scripts\python.exe scripts\run_synthetic_baseline.py
.\.venv\Scripts\python.exe scripts\run_synthetic_baseline.py --cpu
.\.venv\Scripts\python.exe scripts\run_real_weight_smoke.py --layer 1 --max-tokens 64
.\.venv\Scripts\python.exe scripts\evaluate_trace_baselines.py data\traces\real_weight_smoke_layer_1.safetensors --name real_weight_smoke_baselines.json
.\.venv\Scripts\python.exe scripts\fetch_wikitext.py
.\.venv\Scripts\python.exe scripts\collect_wikitext_layer1_traces.py
.\.venv\Scripts\python.exe scripts\fit_shared_basis_baseline.py
.\.venv\Scripts\python.exe scripts\train_aggregate_students.py
.\.venv\Scripts\python.exe scripts\train_residual_basis_students.py
.\.venv\Scripts\python.exe scripts\evaluate_weight_quantization.py
.\.venv\Scripts\python.exe scripts\evaluate_mixed_quantization.py
.\.venv\Scripts\python.exe scripts\collect_all_layer_router_calibration.py
.\.venv\Scripts\python.exe scripts\evaluate_streamed_model_effect.py --student mixed_quant_all_layers
.\.venv\Scripts\python.exe scripts\report_storage_accounting.py
.\.venv\Scripts\python.exe -m pytest
```

Modelgewichten worden nooit impliciet gedownload. Gebruik daarvoor expliciet
`scripts\fetch_v2_lite.py --include-weights`, na controle van
`reports/baseline/model.json`.

De gepinde NumPy-versie is functioneel relevant: NumPy 2.5.2 werd op deze
Windows-machine door Application Control geblokkeerd bij het laden van
`_bounded_integers`; 2.2.6 doorstaat de volledige runtimeprobe.

## Beslisgates

- Gate 0: scripts zijn reproduceerbaar en de hardwarepreflight slaagt.
- Gate 1: op één echte V2-Lite-laag minimaal 4× minder expertbytes bij een
  vooraf vastgelegde fout- en kwaliteitsgrens.
- Gate 2: minimaal 8× reductie, inclusief autoregressieve rollout-evaluatie.
- Gate 3: honderden tokens stabiele generatie zonder routerdrift/collapse.
- Pas daarna: DeepSeek V4 Flash als schaaltest.

Zie [docs/BASELINE_PLAN.md](docs/BASELINE_PLAN.md) en
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md). Het actuele cache-routingverdict
staat in
[reports/MASS_BUDGET_EUREKA_2026-08-10.md](reports/MASS_BUDGET_EUREKA_2026-08-10.md).
Het brede behavioral-verdict blijft beschikbaar in
[reports/EUREKA_VERDICT_2026-08-10.md](reports/EUREKA_VERDICT_2026-08-10.md) en
het eerdere compressierapport als
[reports/RESULTS_2026-08-09.md](reports/RESULTS_2026-08-09.md).

De daarna onafhankelijk geteste GhostWeights/RSIV-hypothese is eveneens
terminaal gesloten. Het volledige bewijsoverzicht staat in
[reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md](reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md):
`falsified_rank_working_set`, geen Eureka.
