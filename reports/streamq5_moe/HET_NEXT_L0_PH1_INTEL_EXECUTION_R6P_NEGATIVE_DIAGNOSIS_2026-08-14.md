# PH1 Intel execution R6P — 12/15 negative diagnose

Datum: 2026-08-14  
Evidence: `het_next_l0_ph1_intel_execution_r6p_static_preflight.json`, SHA-256 `7542e0ffe248176d2d571941f9fae3f12b54faeb9d83cbdef74d4d471437043b`.  
Methode: read-only bron/resultanalyse; geen preflight-, payload-, compiler- of devicecall uitgevoerd.

## Verdict

De 12/15-uitkomst is een **geldige no-device infrastructuurnegative**, geen Intel-, kernel- of wetenschappelijke negative. R6P mag niet worden herhaald. Een immutable R6P1 preflight-only revision is nodig.

## Failure 1 — `no_device_static`: deterministische self-match

De check zoekt de ruwe tekst `WinDLL("OpenCL.dll")` in zowel de eigen R6P-bron als de oude R6-preflight. Beide bestanden bevatten diezelfde tekst als blacklist-string in de check zelf. Daardoor is `all(... not in source ...)` per definitie false, ook al bestaat er geen uitvoerbare WinDLL-call in die statische modules.

### Exacte reparatie

Vervang raw-substringsearch door AST-semantiek op uitsluitend de statisch uitgevoerde preflight/common/verifiermodules:

- reject echte `ast.Call`-nodes naar `WinDLL`, `CDLL`, `LoadLibrary` en directe OpenCL-entrypoints;
- behoud de forbidden-call-surfacecontrole;
- controleer imports op de bestaande device/modelpakketten;
- laat stringconstanten en de frozen production-backendbron niet als uitvoering tellen.

Voeg een mutation/sentinel toe met één echte synthetische `WinDLL(...)` AST-call die de checker moet afwijzen, plus een stringliteral die juist niet mag falen.

## Failure 2 — codecvector heeft verkeerde vooraf verwachte FP32-quantisatie

Voor bronwaarden `[-1,-.5,-.25,0,.25,.5,.75,1]` en de frozen FP32-operatievolgorde `scale=maxabs/15; rint(value/scale)` is de werkelijke q-vector:

`[-15,-7,-4,0,4,7,11,15]`

De R6P-fixture verwacht nog `[-15,-8,-4,0,4,8,11,15]`. Recordbouw, onafhankelijke codec, FMA en beide legal-width linear sentinels zijn verder consistent.

### Exacte reparatie

Freeze de correcte vector `[-15,-7,-4,0,4,7,11,15]`, gebonden aan de exacte FP32/BF16 bronwoorden en operation order. Dit is een deterministische fixture-oraclecorrectie, geen thresholdtuning. Behoud vervolgens de bestaande common-record/independent-codec/digestchecks.

## Failure 3 — verifierbasis gebruikt records en weights uit verschillende werelden

De drie synthetic records zetten in ieder 5-byte pack veld 0 op `16`, dus q=`+1`, met BF16-schaal `1.0`; alle andere velden zijn q=`0`. De aan `verify_dict(..., prepared=...)` gegeven weightarrays zijn echter volledig nul. Daarmee is de baseline geen intern consistente representatie van de evidence die de verifier geacht wordt te beoordelen. Het nulinput maskeert veel numerieke gevolgen, maar mag de record/decoded-weightbinding niet vervangen.

### Exacte reparatie

- Decodeer de synthetic recordcodes en schalen onafhankelijk naar de prepared BF16-weightarrays, of construeer equivalent exact: ieder 8-slotblok heeft slot 0 `0x3f80`, overige slots `0x0000`.
- Eis vóór `verify_dict` bitwise equality tussen deze independently decoded recordweights en alle drie prepared weights.
- Houd input nul als deterministic control gewenst; outputs blijven dan exact nul, maar de record/weightbron is niet langer inconsistent.
- Pas de schema-assert aan van de tautologie `verify.BUFF == tuple(verify.BUFF)` naar `verify.BUFF == old[0]` én behoud de expliciete exact-length checks.

Een alternatief met volledig q=0-records is niet minimaal: de bestaande safe-code-mutationcontrol vereist minstens één canonical field dat niet 15 is. De q=1/scale1-records behouden en correct decoderen is daarom de veilige kleine reparatie.

## Latente checks vóór P1-uitvoering

- Bind de immutable R6P negative result SHA `7542e0ff…`; schrijf naar een nieuwe R6P1-resultaatnaam en eis die vooraf afwezig.
- Laat de P1-output naast 15/15 ook de positieve verifier-baselinecheckmap of minstens de namen van iedere false baselinecheck bewaren; geen stille bool-collapse.
- Behoud full production shapes, volledige counters, legal 512/2048 reductions, 95-row ownership, promoted attestationcleanup en exact-zero statusmutations.
- Hergebruik de oude R6/R6P-resultaten uitsluitend read-only; geen retry, overwrite of verwijdering.

## Aanbevolen R6P1-scope

Alleen de drie bovenstaande fixture-/static-checkreparaties en nieuwe provenancepaden. Geen wijziging aan production runner/backend/common/verifier, codeccontract, kernelbinary, buffers, launches, resources, thresholds, device-identiteit of claim.
