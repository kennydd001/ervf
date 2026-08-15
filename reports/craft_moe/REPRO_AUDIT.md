# CRAFT-MoE reproduceerbaarheids- en statistiekaudit

## Uitkomst

**De bewaarde technische uitkomsten zijn intern reproduceerbaar en hun
preregistered gates zijn correct geadjudiceerd.** De onafhankelijke verifier
voerde 751 controles uit: 748 slaagden, nul verplichte controles faalden en
drie beperkingen bleven als waarschuwing zichtbaar.

Dit oordeel betekent dat de positieve tussenoracles, negatieve screens en harde
stops correct uit hun ruwe metriekvelden volgen. Het is geen Eureka-, novelty-
of snelheidsbewijs.

| Auditfamilie | Aantal controles | Uitkomst |
|---|---:|---|
| gateherberekening | 103 | alle geslaagd |
| metriekherberekening | 85 | alle geslaagd |
| bootstrapunit/-metadata | 311 | alle geslaagd |
| artefacthashes | 104 | alle geslaagd |
| schema en provenance | 52 | alle geslaagd |
| verdicts | 16 | alle geslaagd |
| splitdiscipline | 13 | alle geslaagd |
| overige controls/stop-go/retentie | 67 | alle geslaagd |

## Sterkste onafhankelijke controles

- H7 is opnieuw opgebouwd uit de per-token-NNLS-KL-series. De minimum-`k`-
  verdelingen, hogere empirische mediaan/p95 en `k=5`-falsificatiefracties
  reproduceren exact het inconclusief-negatieve verdict.
- H8 is opnieuw geselecteerd uit alle twaalf validatieconfiguraties met de
  vooraf vaste tie-break. `resident_selected_no_cached_bounded` blijft de
  winnaar en is ongewijzigd op test toegepast. Missfracties, zero-fill-uplift,
  kwaliteit, compute-accounting en beide 10.000× sequence-blockbootstraps
  sluiten exact.
- H10 is niet alleen uit de JSON gecontroleerd: de validation-argmin is opnieuw
  rechtstreeks berekend uit het lossless `q3_validation_mse`-sweeptensor van
  `8×720×256`. Dat levert opnieuw schema-index 4 en permutatie-index 466,
  oftewel `bf16_sequential [3,5,1,4,0,2]`. De exacte KL-gapclosures en beide
  gepaarde bootstraps sluiten eveneens.
- H2-audit-v1 blijft als foutief artefact bestaan. Audit-v2 reproduceert alle
  1.280 HiGHS-records binnen de vooraf vaste toleranties en behoudt hetzelfde
  harde inhoudelijke verdict.
- H4-v1 blijft met haar falende componentcontrol bewaard; de trace-anchored
  replicatie heeft sluitende original-/routecontrols en gebruikt exact de vijf
  vooraf vaste seeds.
- Alle gecontroleerde model- en datasetrevisions zijn respectievelijk
  `604d5664dddd88a0433dbae533b7fe9472482de0` en
  `b08601e04326c79dfdd32d625aee71d232d685c3`. De 136-entry artefactmanifest
  sluit met alle eerder gepubliceerde en intern gedeclareerde SHA-256’s.

## Waarschuwingen die claims begrenzen

1. H8- en H4-componentmicrobenchmarks hebben warmup, synchronisatie en zeven
   herhalingen, maar geen temperatuur-/kloktelemetrie. Zij mogen dus geen
   wall-clockspeedup onderbouwen; de bestaande rapporten noemen ze terecht geen
   packed runtime.
2. H10 registreerde dat Q4 met dezelfde orde niet “catastrofaal” mocht
   verslechteren, maar gaf daarvoor geen numerieke grens. Finite outputs zijn
   aantoonbaar en de hoofdgate faalt onafhankelijk zeer ruim; deze term kan
   echter nooit als zelfstandige positieve claim dienen.
3. De repository heeft nog geen commit. Herhaalbaarheid rust daarom op de
   gepinde upstreamrevisions, commands, ruwe JSON en het append-only
   hashmanifest, niet op een lokale Git-commit.

## Reproduceerbaar commando en verdict

De beslissende JSON is `reports/craft_moe/repro_audit.json` (307.504 bytes,
SHA-256
`b68bec1015a470203eff19b4c19ad55f71976c8b2f77a3f7a4bba0902c7d96bd`). Een
latere niet-schrijvende integriteitscontrole is:

```powershell
.\.venv\Scripts\python.exe scripts\craft_moe\verify_all_gates.py --check-only
```

De verifier eindigt niet-nul zodra een gate, control, hash of manifestentry
afwijkt. Eindverdict: `reproducibility_audit_passed_with_declared_warnings`.
De volgende en laatste inhoudelijke audit is de afgescheiden prior-art- en
claimsmatrix; pas daarna mag het projectmasterverdict worden geschreven.

