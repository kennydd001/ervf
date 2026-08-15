# PH1 Intel execution R0 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Auditmethode: uitsluitend bestanden gelezen en gehasht; kandidaat-preflight, payload, compiler en OpenCL/device zijn niet uitgevoerd.

## Verdict

**NO-GO — voer ook de huidige statische preflight niet uit als kwalificerende PH1-preflight.** De centrale host-USM-uitvoeringsvorm is geïmplementeerd, maar verplichte predevice-, provenance-, resource-, onafhankelijke-verifier- en lifecyclepoorten ontbreken. De huidige preflight kan daardoor PASS melden terwijl deze wetenschappelijke blockers blijven bestaan.

De overgedragen hashes zijn exact bevestigd:

- backend `9810341b2cbe1bf4b541dad1392c6db4383316b0073dece386150cce1141b172`;
- runner `6efb5588a06f2760fbb7740e81c22c1175916b4daaf6bec6892cda375c88dc39`;
- verifier `eff98b017533f3f577bf6783345c81151d1b3387460f7d4cdf08c5f206d937c4`;
- preflight `d0f0119037b67c20eb09d36c5ab7c364e2a6b1127c229792c43de6f8fc2231a5`;
- preregistratie `dfdd73f65d24bb2b0a05994b3b4055969ee717031c3ebb06a0cc4deb54d33c62`;
- gesloten lock `aea9fda7267e7601b3df14482631007929467641d5ad856a0cfa491c19e655bf`.

De outputmap en het statische-preflightresultaat waren afwezig. De lock is correct gesloten met `execution_open=false` en `PENDING`.

## Wat wel correct is uitgewerkt

- De compilebron en 186.352-byte binary zijn aan de bewezen R2A-digests gebonden.
- De bufferconstante bevat 14 allocaties en telt exact `2.185.216` bytes; de argumenttabel telt 18 pointerargs; de vier launches hebben de vastgelegde globale/lokale groottes.
- De backend vraagt host-USM aan, controleert type/base/size/alignment, initialiseert vijf outputcanaries en vier counterarrays, gebruikt een in-order queue, doet één finish en leest pas daarna direct vanaf host-USM.
- De normale success cleanup loopt door alle 14 allocaties, vier kernels, program, queue en context: 21 beoogde pogingen.
- De vijf vastgelegde CPU-Q5-stagehashes en all-one-countergate zijn aanwezig. De bron bevat geen expliciete buffer/read/write/copy/migrate/prefetch-call.

Dit maakt R0 een bruikbaar implementatieskelet, maar nog geen uitvoerbare frozen test.

## Fatale blockers

### 1. De 22 veilige predevice-controls bestaan niet

De fysieke contracttekst vereist zeven checker-controls per Q5-record plus één verkeerde LUT-digest, elk verworpen vóór compile/alloc/launch. `payload()` bouwt alleen de drie positieve records; runner en preflight bevatten geen enkele control-loop, verwachte reject-stage of tellerbewijs. Zie runnerregels 55–60 en preflightregels 9–15.

**Reparatie:** implementeer exact de 22 frozen controls en bewijs per control dat OpenCL-load/context/program/kernel/alloc/launch allemaal nul bleven. Houd de drie outcome-onafhankelijke q-step-witnesses afzonderlijk als onveilige CPU-diagnostiek.

### 2. Payload- en compileprovenance zijn onvoldoende fail-closed

`authorize()` bindt compile commit/result/source/binary en CPU commit/verificatie, maar valideert niet de compilemanifest/buildlog/independent R2A-verificatie en valideert het CPU-pakket niet als pakket. `payload()` importeert bovendien de veranderbare CPU-generator zonder zijn hash te binden. De drie source-records worden alleen tegen de op dat moment geïmporteerde generatorconstanten gecontroleerd; R0 vergelijkt hun uiteindelijke digests niet zelf met `e3b10ab3…`, `6da7025a…`, `bd1a8ef9…`. De natuurlijke input krijgt zelfs geen SHA-controle en de LUT wordt zonder controle gelezen. Zie runnerregels 50–60.

**Reparatie:** bind en verifieer vóór payload alle pakketmanifests/commits en de onafhankelijke PASS; bind de generator of implementeer een standalone reader/codec; controleer de drie recorddigests, inputdigest `5ce66a20…` en LUT-digest `a3cbc779…` expliciet vóór de eerste OpenCL-call. Bind eveneens buildlog `6e340b9c…` en R2A independent-verification.

### 3. Resourcepoorten ontbreken volledig

Er is geen meting/gate voor start available RAM `>=16 GiB`, postfase available RAM `>=2 GiB`, peak working set `<=12 GiB` of retained artifacts `<=16 MiB`.

**Reparatie:** leg metingen en exacte meetmomenten vooraf vast, stop vóór payload/device bij onvoldoende start-RAM, retain ruwe telemetry en laat de verifier alle grenzen herberekenen.

### 4. De onafhankelijke verifier replayt het bewijs niet onafhankelijk

De verifier vertrouwt de in `result.json` opgenomen outputhex en ledger grotendeels. Hij reconstrueert geen Q5-records/input/LUT of CPU-orakel en controleert niet: authorization/provenance, exact device identity, record/input/LUT-writehashes, geordende CPU writes/initialisaties/reads, exact één finish, allocatietype, buffernaam/volgorde/grootte, exacte argnaam/pointermapping, releasevolgorde of totale ledgerordening. De allocationcheck accepteert bijvoorbeeld elke set van 14 unieke pointers met dezelfde totaalgrootte; de argcheck alleen 18 unieke `(kernel,index)`-paren. Zie verifierregels 11–16.

**Reparatie:** maak een standalone verifier zonder runner/backend/generator-import die de inputs en codec zelf herleidt, alle evidencevelden en de volledige geordende ledger tegen exacte tabellen controleert, en niet-vacuous mutaties van elk kritisch veld verwerpt.

### 5. De statische preflight is vacuüm voor de cruciale risico's

De preflight doet alleen hashes, AST-functienamen en tekstaanwezigheid. Hij voert geen 22 controls, codec/orakeltest, backend-emulator, failure-/cleanup-/transaction-simulatie of verifiermutaties uit. De OpenCL-callsetinspectie ziet bovendien de vier dynamisch verkregen USM-functies niet als calls en controleert hun functienamen/signatures niet exact. Zie preflightregels 9–15.

**Reparatie:** laat een nieuwe no-device preflight echte control- en codecfixtures uitvoeren, exact de vier extensionnamen/signatures statisch valideren, runner-transacties in een tijdelijke map uitvoeren en voor elke gate minimaal één gerichte resultaat/ledger-mutatie door de onafhankelijke verifier laten afwijzen.

### 6. Autorisatie volgt na muterende recovery

`main()` roept `recover()` aan vóór `authorize()` (runnerregels 67–69). Met de publiek zichtbare ACK kan een gesloten of ongeldige revisie daardoor stale/corrupt mappen verplaatsen vóór de lock is gevalideerd.

**Reparatie:** een geldige bestaande commit mag read-only als eerste worden herkend; iedere muterende stale/corrupt recovery moet pas na exacte autorisatie plaatsvinden. Een reeds gecommitteerde negatieve uitkomst moet bij heraanroep negatief blijven; de huidige `already_complete`-tak retourneert altijd exitcode 0.

### 7. Cleanup-evidence telt alleen teruggekeerde calls, niet alle pogingen

Een release-returncode wordt vóór `check()` vastgelegd en is goed afgedekt. Maar als de ctypes-call zelf een uitzondering geeft, wordt geen release-attempt-row toegevoegd. Daardoor is niet gegarandeerd dat failure evidence exact 21 pogingen en hun exception/resultaat bevat. De finale `live_owned_resources` wordt bij iedere cleanupfout `None`, zonder resterende ownership per object te rapporteren. Zie backendregels 87–98.

**Reparatie:** registreer iedere poging vóór de call en vul daarna code of exception in; probeer altijd alle 21 objecten, rapporteer per object ownership/release-uitkomst en bereken live resources uit objectstatus.

### 8. Runnergates zijn niet volledig genoeg voor onafhankelijke adjudicatie

De runner controleert niet de exacte allocatienamen/volgorde/groottes of het gerapporteerde host-USM-type, niet de exacte pointermapping van de 18 args, niet exact één finish/read-order en niet de fysieke device-identiteit in zijn eindgates. `forbidden_calls` is een achteraf hardgecodeerde nul-dictionary in plaats van een observeerbare callcounter/ledger. Zie backendregel 136 en runnerregels 61–63.

**Reparatie:** gate de volledige exacte tabellen en volgorde; maak verboden API-counters onderdeel van de werkelijke wrapper/callledger en laat bron-audit én onafhankelijke verifier deze nulclaims bevestigen.

## Besluit

R0 sluit de rekenkundige host-USM-hoofdstructuur grotendeels, maar niet de experimentele integriteit. Vereist is een immutable R1 met bovenstaande acht reparaties, daarna een nieuwe onafhankelijke source-audit. Pas bij GO mag de verbeterde no-device preflight worden uitgevoerd; een PASS daarvan vereist nog een afzonderlijke authorization-only revision vóór exact één fysieke Intel-poging.
