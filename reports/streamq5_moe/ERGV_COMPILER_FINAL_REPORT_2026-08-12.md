# ERGV compiler — C0/C1 eindrapport

Datum: 2026-08-12  
Status: **restricted prototype bewezen; performanceprogramma nog open**

## Uitkomst

Het handgeschreven ERVF-idee is nu voor het eerst omgezet in een kleine
compilerketen:

```text
Q8/Q5-semantiek
  -> ExactReductionIR
  -> fysieke width-schedule
  -> mechanische ordered-graph-verifier
  -> CUDA-row-reducer
  -> generated-vs-manual GPU-bitcheck
```

Beide vooraf geregistreerde fasen zijn geslaagd:

| Fase | Resultaat |
|---|---|
| C0 CPU-formalisatie | 7/7 tests; 20/20 schedules isomorf; 2.680/2.680 bitchecks |
| C1 GPU-codegen | compile 5,023 s; 115.496 elementen; 0 bitverschillen |

## Wat daadwerkelijk nieuw is gebouwd

- Een immutable `ExactReductionIR` met accumulator-ID's, ordered-add-edges,
  BF16-roundpunt, FMA-policy en volledige Q8/Q5-bronvolgorde.
- Een fysieke schedule voor widths `4/8/16/32/64`, inclusief virtuele
  accumulatoren per lane en expliciete width-64 cross-warpfase.
- Een graph-isomorphism-verifier die operatornamen negeert maar geordende
  operands, leaf-identiteit en semantische metadata strikt bewaakt.
- Negatieve controles voor boom-, source-, cast-, FMA-, lane- en
  cross-warpcorruptie.
- Een deterministische Q8/Q5 CUDA-codegenerator.
- Een representatie van alle bevroren N1C-keuzes.
- Een echte generated-width-16 GPU-replicatie van manual P7.

## Stand tegenover de volledige ERGV-ambitie

| Vereiste uit het next-phase-pack | Stand |
|---|---|
| Manual P7 reproduceren | **gedeeltelijk bewezen**: source audit + width-16 GPU-bitexact |
| Exact op Q8 en Q5 | **bewezen voor generated width 16 op synthetische fysieke data** |
| N1C-keuzes representeren | **bewezen in IR/planning** |
| Manual P7 op snelheid verslaan | niet getest |
| Alle widths op echte modelbank | niet getest |
| Qwen3-Coder-Next-vormen/gewichten | niet getest |
| Tweede GPU-architectuur | niet getest |
| GemLite/CUTLASS/QUICK-baseline | niet getest |
| Predictief prestatiemodel | niet gebouwd |

## Eerlijk verdict

Dit is meer dan een idee of een papieren ontwerp: er bestaat nu een werkende
exactheidscompiler met een mechanisch bewijs en een geslaagde generated CUDA
gate. Het is echter nog geen industriële doorbraak. De doorslaggevende volgende
fase is een vooraf geregistreerde zoek- en timingproef op echte Q8/Q5-banken:
alle gegenereerde widths tegen manual P7/N1C, gevolgd door een ongeopende
testreeks. Alleen als een gegenereerde variant manual P7 op minstens één
matrixfamilie verslaat, wordt de wetenschappelijke compilerclaim sterker dan
reproductie.

Daarna zijn een tweede moderne MoE-shape, tweede GPU-architectuur en equivalente
publieke baselines verplicht voordat een brede of nieuwe-state-of-the-artclaim
verdedigbaar wordt.

## Hoofdartefacten

- `src/moe_lab/ergv_compiler.py`
- `scripts/streamq5_moe/ergv_compiler_cpu_tests.py`
- `scripts/streamq5_moe/ergv_c1_generated_gpu_gate.py`
- `reports/streamq5_moe/ergv_compiler_cpu_tests.json`
- `reports/streamq5_moe/ergv_c1_generated_gpu_gate.json`
- `reports/streamq5_moe/ERGV_COMPILER_CPU_REPORT_2026-08-12.md`
- `reports/streamq5_moe/ERGV_C1_GENERATED_GPU_REPORT_2026-08-12.md`
