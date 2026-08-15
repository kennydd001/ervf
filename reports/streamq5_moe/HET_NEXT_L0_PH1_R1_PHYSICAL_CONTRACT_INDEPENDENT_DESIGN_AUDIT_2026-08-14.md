# PH1-R1 physical contract — onafhankelijke designaudit

Datum: 2026-08-14  
Definitieve onderzochte SHA-256: `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981` (`11,613` bytes). De eerdere SHA `2d83d0d6...` is superseded.

## Verdict

**NO-GO voor implementatie wegens één uitvoeringskritische NVIDIA-contextlacune.** Geen device-actie is uitgevoerd.

Het contract zegt dat de NVIDIA-child de CUDA Driver API en een vooraf gebouwde cubin gebruikt, maar specificeert slechts een “primary-context observation”. `cuModuleLoad*`, streamcreatie, allocaties en launches vereisen in het schone child-proces een geldige current CUDA-context. Het document bevriest niet:

- hoe de primary context wordt retained (`cuDevicePrimaryCtxRetain`) of hoe een alternatief owned context wordt gemaakt;
- hoe de context current wordt gemaakt en welke vooraf bestaande current-contextstatus toegestaan is;
- hoe current-context herstel/pop wordt gecontroleerd;
- hoe de retain exact eenmaal wordt gereleased zonder de primary context te resetten;
- hoe deze extra acquire/cleanup-acties in ledger, failure-cleanup en mutation tests worden opgenomen.

Daardoor zijn “primary context is observed but not owned” en een standalone directe Driver-API execution niet samen uitvoerbaar. Een minimale R2 moet één exact protocol kiezen. Aanbevolen voor het beoogde primary-contextmodel: init/device select; initial-current-context gate; primary-context state query; retain; push/set current; al het devicewerk; reverse resource cleanup; pop/restore met identitycheck; exact één primary-context release; nooit reset. De positieve cleanupcardinaliteit moet dan de 30 bestaande resource releases plus de afzonderlijk benoemde context-pop/restore/release-acties omvatten.

## Gesloten onderdelen

De overige onderzochte ontwerpblokkades zijn afdoende gesloten:

- normatieve 65,536-entry PyTorch-BF16 SiLU-LUT en aparte 100-dps diagnose zijn correct gescheiden;
- de integer BF16-multiply is normatief beschreven; `(7f7f,0001)->3cff` is consistent met onafhankelijke exacte herberekening;
- twaalf stagehashes, runtimeomgeving en de positieve CPU-kwaliteit zijn exact gebonden;
- de 14-buffer-tabel telt exact `2,185,216` bytes;
- Intel: 14 host-USM allocaties, 18 pointerargs, 4 enqueues en 21 normale releasepogingen zijn intern consistent;
- NVIDIA, exclusief de ontbrekende contextlevenscyclus: 14 pinned + 14 device allocaties, 9 memsets, 5 H2D, 4 kernels, 9 D2H, 1 sync en 30 normale buffer/module/stream releases zijn intern consistent;
- de compile/no-FTZ freeze-eisen zijn fail-closed;
- controles zijn exact 21 recordcontroles + 1 globale LUT-controle; de drie numerieke witnesses zijn afzonderlijk en niet outcome-geselecteerd;
- claimgrens en Intel-vóór-NVIDIA volgorde blijven smal en eerlijk.

Na uitsluitend de contextreparatie is een nieuwe designaudit nodig. Pas daarna is implementatie aan de orde; dit rapport autoriseert geen static preflight, compiler- of device-call.
