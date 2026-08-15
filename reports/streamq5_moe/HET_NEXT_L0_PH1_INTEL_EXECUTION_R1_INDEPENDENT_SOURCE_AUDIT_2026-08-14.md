# PH1 Intel execution R1 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron/lock-audit; preflight, payload, compiler en device zijn niet uitgevoerd.

## Verdict

**NO-GO voor de huidige statische preflight.** R1 sluit het merendeel van de acht R0-clusters inhoudelijk, maar heeft vier blokkerende integriteitsgaten en één formele provenance-onvolledigheid. De huidige preflight kan daarop ten onrechte PASS geven.

Alle overgedragen R1-hashes zijn bevestigd; iedere in de gesloten lock genoemde artefacthash matcht. `execution_open=false`, token `PENDING`, outputmap afwezig en preflightresultaat afwezig zijn bevestigd.

## Correct gesloten ten opzichte van R0

- Exact 22 controles zijn als executable synthetische controls aanwezig met verwachte reject-stage en nul predevice-tellers.
- Compilemanifest/log/binary/source, CPU-manifest/commit, beide onafhankelijke verificaties, rapporten en generator worden in `authorize()` gehasht.
- Autorisatie gebeurt vóór de muterende recovery; bestaande positieve en negatieve commits behouden exitsemantiek.
- Start-, reserve-, peak-working-set- en artifactgates zijn toegevoegd.
- De backend registreert releasepogingen vóór de call en probeert de volledige 21-objectenlijst door; faultinjectie controleert dit.
- Runnergates gebruiken exacte allocation-, write-, initialization-, arg-, launch-, read- en release-tabellen.
- De verifier implementeert zelfstandig codec en exact integer/FMA-orakel zonder kandidaatrunner/backend/common/generator te importeren.

## Blokker 1 — twee transitief uitgevoerde R0-bronnen zijn niet gebonden

De R1-backend importeert en erft `het_next_l0_ph1_intel_execution_r0_backend` (backendregels 6–17). De R1-runner importeert `run_het_next_l0_ph1_intel_execution_r0` en gebruikt diens `recover`, `archive`, `canon`, `write` en `move` (runnerregels 9, 21, 36–49). Deze twee bestanden bevatten dus de daadwerkelijke OpenCL-uitvoeringskern en transactielogica, maar hun hashes staan niet in R1-lock, `EXPECTED`, `authorize()` of preflight.

Actuele maar ongebonden hashes:

- R0-backend `9810341b2cbe1bf4b541dad1392c6db4383316b0073dece386150cce1141b172`;
- R0-runner `6efb5588a06f2760fbb7740e81c22c1175916b4daaf6bec6892cda375c88dc39`.

Een wijziging aan een van deze bestanden passeert alle R1-lockchecks. **Reparatie:** maak R1 standalone of bind beide dependencyhashes in prereg, lock, authorization en self-binding preflight.

## Blokker 2 — CPU/compile package-validatie controleert commitsemantiek niet

`package_exact()` controleert bestandsset en manifestrijen, maar leest `commit.json` alleen in en gebruikt `c` vervolgens niet (runnerregels 22–23). Een pakket kan dus intern een ongeldige commitinhoud hebben terwijl de functie `True` retourneert; alleen de bekende commit-filedigest voorkomt dit in de huidige snapshot, maar de geclaimde volledige pakketvalidatie is niet werkelijk geïmplementeerd. Ook worden `kind` en exacte manifestschema's niet gecontroleerd.

**Reparatie:** eis exact package-kind, unieke/nauwkeurige manifestrijen en dat commit exact de manifest- en resultdigest bindt. Laat preflight gerichte manifest/commit-mutaties verwerpen.

## Blokker 3 — de preflight test niet de echte codec of het integer-orakel

De preflight bouwt synthetische records direct uit reeds geconstrueerde codes/scales en roept alleen `controls()` aan. Zij test `common.record()` niet, en voert geen enkele vector of stage uit door de zelfstandige `codec`, `fma`, `mul` of `linear` implementatie van de verifier. Daarmee kunnen codec- of rekenbugs bestaan terwijl `synthetic_22_controls` PASS blijft. Dit sluit het R0-verzoek om executable codec/orakelfixtures niet.

**Reparatie:** voeg kleine, vooraf bevroren source→Q5-code/scale/recordvectors en BF16/FMA/width-8 reductievectors toe; voer ze werkelijk door beide onafhankelijke paden en laat gerichte codec/reductiemutaties falen.

## Blokker 4 — `verifier_mutations` muteert niet de echte verifier

De preflight noemt de test `verifier_mutations`, maar muteert alleen een kleine kunstmatige dictionary voor `static_contract_fixture()` (preflightregels 47–50). `verify_dict()` wordt nooit op een volledig positief dummyresultaat aangeroepen. Geen ledger-, output-, counter-, identity-, provenance-, resource-, commit- of cleanupmutatie raakt dus de echte verifiercode.

**Reparatie:** construeer een compleet positief verifierfixture of een compacte representatieve schemareplay en eis dat `verify_dict()` iedere vooraf geregistreerde kritieke mutatie afwijst. Test daarnaast manifest/commit via een tijdelijke bundle.

## Blokker 5 — resource- en failure-evidence zijn formeel onvolledig

- `peak_wset` wordt slechts rond payload en na device gemeten; er is geen retained sampling tijdens de kernel/backendfase. Dit ondersteunt de claim “peak working set at all retained samples”, maar niet een werkelijke fasepiek.
- `artifact_cap` wordt pas getest nadat result/manifest/commit al in de tijdelijke map staan. Bij overschrijding wordt de map correct naar failure verplaatst, maar de failurebundle zelf is niet tegen de 16 MiB-cap of een manifest/commit gecontroleerd.
- De eindverifier controleert artifactgrootte op de uiteindelijke drie bestanden, maar vergelijkt `resource` niet met een onafhankelijk vastgelegde OS-telemetrybron; de waarden blijven runner-self-report.

**Reparatie:** preregistreer de exacte meetmomenten als steekproeven (zonder “peak” als ononderbroken maximum te claimen), neem een meting direct vóór/na elke fysieke fase op, en definieer failure-artifactcap en failurebundle-integriteit. De verifier moet alle retained samples en schema's controleren.

## Extra aandachtspunt — verboden-callcounter is niet instrumenterend

`forbidden_calls` wordt berekend uit ledgerregels met een `api`-veld, maar geen wrapper genereert zulke regels voor de verboden APIs; de backend exposeert of bindt deze functies simpelweg niet. Dit is bruikbaar als bronoppervlakbewijs, maar geen runtime-callcounter. Formuleer de claim als “API niet gebonden en niet aanwezig in geaudite callgraph” of implementeer een echte dispatch/interceptielaag. De huidige naam `wrapper-owned observed counters` overschrijdt het bewijs.

## Vereiste R2-route

Bind of elimineer de twee R0-dependencies; maak commit/package-validatie werkelijk exact; vervang de oppervlakkige preflightfixtures door executable codec/orakel- en echte `verify_dict()`-mutaties; herstel resource/failure-schema en maak de forbidden-API-claim bewijsconform. Daarna is een nieuwe frozen source-audit nodig. Alleen bij GO mag exact de nieuwe no-device preflight worden uitgevoerd; een PASS opent nog uitsluitend een aparte authorization-only revision voor één fysieke Intel-correctnesspoging.
