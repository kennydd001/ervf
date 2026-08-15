# Next Wave — onafhankelijke CPU-verificatie

Datum: 2026-08-12  
Verdict: **90/90 controles PASS**  
Uitvoering: CPU-only; geen GPU-, CUDA- of netwerkgebruik

## Uitkomst

De volledige next-wave-registry `N001` tot en met `N033` en alle nu aanwezige
primaire en companion-resultaten zijn opnieuw onafhankelijk gecontroleerd. Alle
90 auditchecks slagen. Dit bevestigt dat hashes, ruwe timingstatistieken,
rekenkunde, selectieprocedures, poortbesluiten en de expliciete negatieve
bevindingen intern consistent zijn.

De audit bestrijkt:

- 33 aaneengesloten registry-items, zonder `queued` of `in_progress` item;
- 30 primaire en companion-JSON's van N1A tot en met N4BR;
- alle 242 fysieke Q8-bankrecords plus samengestelde bankhash;
- alle 48 P4D-routebestanden;
- 38 ruwe CUDA-eventmeetsets, met herberekende mean/p50/p95/min/max;
- ABBA-volgorde, warmupuitsluiting, statistieken en verdicts voor de gepaarde
  end-to-end-proeven;
- de afzonderlijke N2C-, N2D-, N3A2-, N3A3-, N4B- en N4BR-companionaudits.

## Nieuwe bevestigingen

### N3A2 en N3A3

- N3A2 concat-QKV is bitexact over `245.760` Q/K/V-uitgangen en `49.152`
  KV-elementen. De testverhoudingen zijn opnieuw berekend: p50 `0,881714` en
  p95 `0,889551`. De componentpoort is positief.
- De preregistratietekst noemt abusievelijk `294.912` outputs; de feitelijke en
  volledig geteste vorm is `48 × (4096 + 512 + 512) = 245.760`. De companion
  documenteert dit expliciet.
- N3A3 integreert dezelfde concat-QKV-kern exact in de 128-token P13-runtime.
  Alle zes toestandassen zijn gelijk, maar mean `0,989449` en p50 `0,984025`
  missen beide de vooraf vastgelegde `0,98`-poort. Het end-to-end-verdict blijft
  daarom terecht negatief.
- N3A4 is bitexact over `98.304` O-projectie→residual-states, maar zijn
  validation-p50-ratio `0,999406` mist de `0,98`-openingspoort; de test bleef
  terecht gesloten. Provenance, ratio's en companion `10/10` zijn bevestigd.

### N4B-audit en N4BR-reparatie

- De oorspronkelijke N4B-resultaatrekenkunde klopt: width 8 werd op validation
  geselecteerd, resident expert-p95 is `7,7067 ms`, en de opgeslagen
  conservatieve totaalprojectie is `35,7839 ms`.
- De onafhankelijke N4B-companion vond echter een echte semantische fout: N4B
  rondde SiLU niet eerst naar BF16 af vóór de vermenigvuldiging met `up`.
  Daardoor bewijst de onderlinge width-gelijkheid geen exactheid tegenover het
  canonieke STREAMQ5-contract. Bovendien bond het resultaat geen evaluatorhash
  en archiveerde het geen onafhankelijke outputdigests. De companion sluit N4B
  daarom correct als **numerieke shape/timing-pass, maar geen onafhankelijk
  geverifieerde exacte port**.
- N4BR is een afzonderlijk vooraf geregistreerde fysieke replicatie met de
  canonieke tweestaps-SwiGLU-afronding, evaluator- en inputprovenance en
  SHA256-digests voor width 8/16/32 plus de width-16-referentie. Alle vier
  digests zijn gelijk.
- De CPU-companion reconstrueerde zowel de deterministische inputdigest als de
  volledige 48-laagse outputdigest onafhankelijk en slaagde `34/34` checks.
  N4BR selecteert width 8, meet expert-p95 `8,8689 ms` en projecteert
  conservatief totaal-p95 `36,9461 ms`; alle vastgelegde 50/40/90-ms-poorten
  slagen.

N4BR bewijst daarmee een exact, synthetisch, fysiek Q5-active-expert-
vormresultaat op deze GPU. Het bewijst nog geen echte 80B-checkpointkwaliteit,
routergedrag, DeltaNet-kerneltijd of end-to-end tokens/s.

## Registrytoestand

`registry_all_local_testable_closed=true`: alle 33 items hebben een terminale
status en geldige evidence- of blockerregistratie. De items `N029`–`N032` zijn
correct als `blocked_scope` vastgelegd; de audit interpreteert dit niet als
experimenteel succes.

Een registry-item mag meerdere subproeven bevatten. De verifier eist daarom bij
een positieve componentstatus minstens één positieve primaire evidence en bij
een negatieve status minstens één negatieve primaire evidence, terwijl iedere
nieuwe concrete proef daarnaast zijn eigen strengere inhoudelijke checks krijgt.
Dit voorkomt dat een negatieve subarm een geldige positieve component maskeert,
of omgekeerd.

## Reproduceerbaarheid

- Verifier: `scripts/streamq5_moe/verify_next_wave_2026_08_12.py`
- Machineleesbare uitvoer:
  `reports/streamq5_moe/next_wave_independent_verification_2026-08-12.json`
- Verifier-SHA256:
  `ba1d5c4477370564f4fff78966d44d07d745e9c9ed6eb37372e92cae5aa6b2be`
- Geaudite registry-SHA256:
  `52f0546e06767bb90f16e010a24cda9856efb0419b697a06b81bd5195944edac`

De verifier gebruikt uitsluitend de Python-standaardbibliotheek. Hij voert geen
experimentele kernels uit en creëert geen nieuwe performance-, kwaliteit-,
cross-model-, cross-GPU-, nieuwheids- of SOTA-evidence; hij controleert de
opgeslagen evidence en haar wetenschappelijke claimgrenzen.
