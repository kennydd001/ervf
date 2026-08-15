# GaugePack P9D-1 — gesloten door ongeldige P9B-premisse

## Uitkomst

P9D-1 krijgt status **`blocked_invalid_p9b_premise`**. De codec is ontworpen en
de fysieke layout werd voor laag 0, experts 0–15 gerealiseerd, maar er kan geen
wetenschappelijk geldige exactheidspass worden afgegeven.

De reden is fundamenteler dan de codec: P9B heeft de expertgewichten nooit
gepruned. De helper gebruikt onder andere `weight[~mask].zero_()` en
`weight[:, ~mask].zero_()`. Boolean advanced indexing maakt hier een tijdelijke
tensor; `zero_()` schrijft niet naar de oorspronkelijke Parameter terug.

## Direct bewijs

- De geïnstalleerde Qwen-forward gebruikt werkelijk de `ModuleList`-experts en
  hun gate/up/down-Parameters.
- Op echte laag-0/expert-0-checkpointgewichten bleven alle drie parameterhashes,
  alle tellingen van niet-nulwaarden en de BF16-expertforward exact gelijk na de
  P9B-helper.
- Een gecorrigeerde `masked_fill_`-mutatie nulde alle bedoelde waarden en
  veranderde 8.181 BF16-elementen van dezelfde vaste forwardprobe.
- P9B en P9E gebruiken nul identieke expert-laagmaskers en verschillen op
  2.018.499 van 2.359.296 indexposities, maar rapporteren exact dezelfde
  candidate CE, top-1, eind-hidden error en alle 48 laag-errors.

Daarmee zijn de eerder gemelde P9B-kwaliteitsscores resultaten van de
ongesnoeide Q5-baseline, niet van 50%-expertpruning. Zij mogen GaugePack niet
autoriseren.

## Wat de codec-dry-run wel en niet zegt

De gerealiseerde 16-expertlayout was 24.395.840 bytes tegenover 48.562.176 bytes
voor dezelfde volledige P1D-records: ratio 0,502363 en een lineaire
full-bankprojectie van 8,725 GiB. Headers, CRC's, maskers en group-ID's waren
intern consistent. Deze maat is uitsluitend een formaatberekening; het
onderliggende vermeende P9B-model bestond niet en de poging is daarom onder
`failed_attempts` bewaard.

Daarnaast bleken CPU en de historische CUDA-run bij gelijke BF16-schalen op
44.317 gate/up-survivorcodes te verschillen. Een toekomstige bitexacte oracle
moet dus dezelfde CUDA-codekeuzes reproduceren.

## Vereiste vervolgstap

Eerst P9B-R uitvoeren met echte in-place `masked_fill_`-pruning: opnieuw
calibreren, daarna validation en alleen bij pass test. Pas wanneer P9B-R de
kwaliteitsgate haalt, mag de CUDA GaugePack-codec/oracle worden heropend.

Dit rapport maakt geen codec-pass-, kwaliteits-, kernel-, full-bank-,
throughput-, 80B- of nieuwheidsclaim.
