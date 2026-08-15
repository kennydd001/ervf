# PH1 Intel execution R2 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: uitsluitend read-only bron-, lock- en callgraph-audit; preflight, payload, compiler en device zijn niet uitgevoerd.

## Verdict

**NO-GO — voer de huidige no-device preflight niet uit.** R2 sluit de twee transitieve R0-hashbindingen en verbetert package-, resource- en fixture-evidence, maar de preflight heeft een deterministische crash en meerdere formele gates blijven vacuüm of onvolledig.

De zeven overgedragen hashes matchen exact. Ook de gebonden R1-audit, R0-backend en R0-runner matchen hun lockwaarden. Lock is gesloten (`execution_open=false`, `PENDING`); output en preflightresultaat zijn afwezig.

## Gesloten R1-punten

- R0-backend `9810341b…` en R0-runner `6efb5588…` zijn nu gebonden in lock, autorisatie en preflight.
- `package_exact()` controleert unieke manifestrijen, bestandsset/hashes, package-kind en exacte commitinhoud voor compile- en CPU-pakket.
- Resourcebewijs is eerlijk hernoemd tot retained phase samples; backend-entry, pre/post vier launches, pre/post finish en post-cleanup worden vastgelegd.
- De preflight bevat positieve codec/FMA/BF16/reductiefixtures en package-commitmutaties.
- Failure-artifactgrootte en disposition zijn gedeeltelijk toegevoegd; de forbidden-API-claim is nauwer geformuleerd als statisch call-oppervlak plus ontbrekende ledgerrows.

## Blokker 1 — huidige preflight crasht deterministisch

`Backend.close()` schrijft eerst de cleanup-row en daarna `resource_sample('post_cleanup')` (backendregels 42–43). `cleanup_faults()` verwacht na `b.close()` echter `b.ledger[-1]['release_attempts']` (preflightregel 30). De laatste rij is `resource_sample` en bevat die sleutel niet. De eerste faultinjectie eindigt daarom in `KeyError`, voordat een preflight-JSON kan worden geproduceerd.

**Reparatie:** zoek de unieke cleanup-row op `op=='cleanup'` en controleer daarnaast expliciet de ene post-cleanup sample; test alle 21 exceptionposities en niet-nul-returncodes.

## Blokker 2 — nonzero codecpad wordt niet werkelijk getest

`codec_oracle_fixtures()` voert `common.record()` alleen uit op een volledig nul-sourcepad. Dat bewijst header/padding en de triviale q=0-code. Het nonzero deel berekent daarna alleen handmatig een `q`-array en vergelijkt acht waarden; het voert die nonzero source niet door `common.record()` of de onafhankelijke `verify.codec()`. Q5-sign, q+15-packing, scale-BF16, CRC en decode voor niet-nulwaarden kunnen dus breken terwijl de fixture PASS blijft. Er zijn ook geen gerichte codec- of reductiemutaties.

**Reparatie:** leg een kleine niet-nul sourcefixture vast met exacte codes/scales/CRC/record/decode en voer hem door beide implementaties; laat vooraf bepaalde packing-, scale-, tie- en width-8-DAG-mutaties afwijzen.

## Blokker 3 — “actual verifier mutations” gebruikt nog steeds een alternatieve speelgoedverifier

`verify_dict()` dispatcht bij kind `ph1_intel_execution_r2_preflight_fixture` onmiddellijk naar `verify_mutation_fixture()` (verifierregels 97–104). Die vergelijkt een kleine synthetische dictionary en doorloopt niet de echte evidence-parser, codec/orakelreplay, exacte ledgertabellen, authorization, package- of resourcechecks van het productiepad. De preflightmutaties op regels 52–59 testen daardoor opnieuw niet de werkelijke verifier.

**Reparatie:** bouw een volledig positief productie-schemafixture (met compacte/mocked onafhankelijke arithmetic inputs waar nodig) dat het normale `verify_dict()`-pad doorloopt, of factoriseer exact dezelfde schema/ledger/provenancefuncties zodat preflight en main aantoonbaar dezelfde functies gebruiken. Muteer iedere kritieke echte rij/waarde afzonderlijk.

## Blokker 4 — onafhankelijke verifier herberekent provenance niet

De productie-verifier accepteert authorization wanneer de zelfgerapporteerde `auth['observed']` gelijk is aan waarden in de lock. Hij hasht niet zelf runner, backend, common, de twee R0-dependencies, prereg, audit, preflight, generator, compile manifest/result/commit/buildlog/verificatie/rapport of CPU manifest/commit/verificatie/rapport. Voor compile controleert hij alleen source en binary. Dit is geen onafhankelijke volledige provenance-replay.

**Reparatie:** definieer een eigen padtabel in de verifier, hash ieder gebonden bestand rechtstreeks, controleer beide package-commits/manifests volledig en vergelijk die herberekening met lock én result.

## Blokker 5 — host-USM-uniciteit is nog niet gated

Contract en R0-audit vereisen 14 unieke pointers. Runner en verifier vergelijken naam/grootte/type/base/size/alignment, maar geen van beide vereist `len(set(pointer))==14`. Ook backend weigert aliasing niet na iedere allocatie. Exacte outputhashes maken veel aliasgevallen fysiek onwaarschijnlijk, maar vervangen de preregistreerde allocation-integriteitsgate niet.

**Reparatie:** fail onmiddellijk bij nul of reeds gebruikte pointer; gate onafhankelijk 14 unieke pointers en bind iedere arg aan de juiste unieke allocation.

## Blokker 6 — preflight self-binding en lifecycle zijn nog onvolledig

- Preflight `observed` bevat alleen de vijf R2-bronnen/prereg en twee R0-bronnen. Audit, compileketen, CPU-keten en generator worden niet tegen de lock gehasht; een drift kan preflight-PASS geven en pas later door autorisatie worden ontdekt.
- `transaction_sim()` roept geërfde helpers direct aan, niet productie-`main()`, en test geen authorization-before-mutation, negatieve already-complete exitcode, productie-failure-schema of artifactcap.
- Autorisatie, start-RAM en `common.package()` staan vóór de productie-`try`; fouten daar leveren geen vastgelegde failurebundle, ondanks de brede preregclaim.
- Als een reeds verplaatste failure-attempt groter dan 16 MiB blijkt, gooit regel 57 een nieuwe fout maar laat de oversized archive bestaan. De cap is dus een detectie, geen fail-closed cap.

**Reparatie:** bind de volledige lockketen in preflight; voer productielevenscyclus via tijdelijke paden en geïnjecteerde dependencies uit; definieer welke predevice stops bewust geen attempt zijn; maak failurepromotion atomisch en garandeer vóór promotion dat de volledige doelbundle binnen de cap valt.

## Blokker 7 — forbidden runtimecounter blijft niet-observerend

`forbidden_calls` telt ledgerrows met `api==name`, maar geen wrapper kan zulke rows voor verboden APIs produceren. De statische AST-check is legitiem; de runtime-dictionary is afgeleid van afwezigheid in een ledger die deze calls niet instrumenteert. Behoud uitsluitend de statische/hashgebonden claim, of implementeer echte dispatchinstrumentatie. Laat runner/verifier niet doen alsof dit een onafhankelijke runtimecounter is.

## Volgende stap

Maak een immutable R3 die eerst de deterministische preflight-crash herstelt en daarna bovenstaande zes bewijsgrenzen sluit. Vervolgens is opnieuw frozen source-audit vereist. Alleen een GO daarop opent exact de verbeterde no-device preflight; ook een preflight-PASS autoriseert nog geen fysieke devicepoging zonder afzonderlijke authorization-only revision.
