# PH1 CPU-freeze R2 — onafhankelijke verificatie

Datum: 2026-08-14  
Methode: CPU-only, read-only ten opzichte van het bevroren pakket; geen import van de base-, R1- of R2-runner; geen device- of compiler-aanroep.

## Verdict

**PASS — 17/17 checks.** De CPU-evidence opent werkelijk de poort om de Intel-validatie-arm te **ontwerpen en implementeren**. Dit autoriseert nog geen Intel-preflight of fysieke device-run.

De onafhankelijke herbouw reproduceert exact:

- de drie allowlisted officiële bronranges en hun canonieke Q5-records;
- de D2R3-input voor prompt 0, token 15;
- de canonieke PyTorch-BF16 SiLU-LUT en de afzonderlijke 100-dps `mpmath`-diagnose;
- de fused officiële source-gate/up graph;
- de integer Q5 gate/up, LUT, BF16-multiply en width-8 down-reductie;
- alle twaalf opgeslagen raw-stage tensors en hun SHA-256-digests.

De herberekende eindkwaliteit is identiek aan het bevroren resultaat:

- `relL2 = 0.040058847132189`;
- `max_abs = 0.00244140625`;
- `different_words = 1970`.

De normatieve LUT heeft SHA-256 `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed`; de onafhankelijke hoogprecisie-diagnose heeft SHA-256 `f2efcbdc3b94b42a24dfe187321ae2a426e7685ab447e05452be994e843693c2`. De verwachte 145 BF16-woordverschillen zijn exact gereproduceerd.

## Package- en provenancecontrole

Het pakket bevat exact de vijf door `manifest.json` gecommitte datafiles. Elke grootte en SHA-256, de manifesthash in `commit.json`, de handoffhash, de base-resultaatbinding, runner/prereg/auth-lock, runtimevelden, denormal witness en RAM-gates zijn opnieuw gecontroleerd. `device_or_compiler_opened` is `false`.

De gerapporteerde peak working set is 659,292,160 bytes, onder de 12-GiB-limiet; de start- en eindreserve blijven boven 2 GiB.

## Claimgrens

Dit bewijst CPU-eligibility voor het implementeren van een Intel-validatie-arm voor **één reeds bekende expert/input-combinatie**. Het is geen Intel-uitvoeringsresultaat en geen bewijs voor een volledige expert, MoE-laag, model, prestatie, generalisatie of industriële doorbraak.

Machineleesbare verificatie: `reports/streamq5_moe/het_next_l0_ph1_cpu_freeze_r2_independent_verification.json`.  
Standalone verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_cpu_freeze_r2.py`.
