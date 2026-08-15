# HET-NEXT CAP0X — existing-runner concurrency diagnostic preregistration

## Doel

Dit is een **exploratieve diagnose**, geen confirmatory CAP0-test. Twee eerder fysiek
geslaagde en inhoudelijk ongewijzigde runners worden in afzonderlijke processen zo
dicht mogelijk na elkaar gestart:

- Intel: `run_st2_mini_ergv_w8.py` (host-USM, Q5, width 8);
- NVIDIA: `run_port80b_d7_staged_exact_q5_plane.py` (CUDA staged Q5 plane).

De diagnose beantwoordt slechts of beide bestaande runtimes tijdens dezelfde
procesperiode zonder fout kunnen eindigen en hun eigen reeds bestaande
correctness-gates behouden.

## Bevroren protocol

- Python: `.venv/Scripts/python.exe`.
- Beide child-processen krijgen nieuwe create-new outputpaden.
- NVIDIA wordt eerst gestart; Intel uiterlijk direct daarna.
- Geen retries, geen retuning en geen wijziging van kernel-, warmup- of timingconstanten.
- De coordinator bewaart per child: PID, start-QPC, einde-QPC, exitcode, stdout en stderr.
- Tijdens de gezamenlijke looptijd wordt elke 100 ms vastgelegd of beide processen
  nog leven en, indien beschikbaar, NVIDIA-utilisatie via `nvidia-smi`.
- Procesinterval-overlap is strikt
  `max(start_intel,start_nvidia) < min(end_intel,end_nvidia)`.

## Diagnostische positieve conjuncten

1. Beide processen hebben exitcode 0.
2. De procesintervallen overlappen strikt.
3. Intel-resultaat meldt geen runtimefout en behoudt bitexactheid.
4. NVIDIA-resultaat meldt geen runtimefout, bitexactheid en schone unregister.
5. Geen child blijft na de bounded wait leven.

## Claimgrens

Een positieve diagnose bewijst niet dat kernels werkelijk temporeel overlappen,
niet dat één proces beide runtimes kan laden, niet dat een hybride 80B-laag sneller
is, en niet dat er een productierijpe cleanup- of recoveryharness bestaat. Zij
rechtvaardigt alleen een kleine formele procesgeïsoleerde componenttest rond exact
deze werkende backendcombinatie.

