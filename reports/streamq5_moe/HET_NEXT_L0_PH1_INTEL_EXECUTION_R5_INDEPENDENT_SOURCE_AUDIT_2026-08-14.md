# PH1 Intel execution R5 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron-, lock- en callgraph-audit; geen preflight, payload, compiler of device uitgevoerd.

## Verdict

**NO-GO voor uitvoering van de huidige statische preflight.** R5 sluit het merendeel van de vier R4-clusters, zonder wetenschappelijke regressie, maar twee exacte evidence-/fault-fixturegaten blijven open.

Beoordeelde freeze:

- runner `d67038f67398d6410cc10e13157f353c694194e65f1a6addb18ca150c550a07e`;
- backend `48f39e31fd726040a1a095997e215d40e7e91525afab0ee614acecf97a240f53`;
- common `27809db51ba43171cb0f69cd10d64688f8c871532571e042b6e95760f06b8189`;
- verifier `e3e781a6d5197c454bd1e10ddfd3793f0b7cb70339ab27293f0480044cb9836a`;
- preflight `544332351fb683ddd9c7c7a092e2fed64bb3f63e2ee19a13a11558a44fbf9736`;
- preregistratie `6272a39c9ce4fb6cb0d0977d93ce84beec617ff5770246ff65a838afe9e30b23`;
- lock `7d8ca7ab4ec62e52444940dac29981f47981e1c6fa5fe041701c0f5b177bee3c`.

Lock is gesloten/PENDING; output en preflightresultaat zijn afwezig. De R4-audit `29d418b3…` is exact gebonden. R5-common verschilt van R4-common alleen door trailing lege regels; Q5-codec, input, LUT, kernelbinary, shapes, launches, thresholds en claim zijn niet gewijzigd.

## Werkelijk gesloten

- Runner en onafhankelijke verifier koppelen de zeven create-returns en veertien allocreturns aan main-ledgerhandles/pointers; veertien frees worden aan de omgekeerde allocationvolgorde gekoppeld.
- `alignment==4096`, viermaal `event_requested is False` en 21 maal `owned_before is True` zijn nu zowel runner- als verifiergates en hebben gerichte mutationfixtures.
- De ownershipfixture muteert missing, duplicate, returned, pending en object-pointer evidence.
- TEMP-lifecyclepaden bereiken via de echte runner auth-, start-RAM-, payload-, post-device-, telemetry-, oversize/quarantine- en already-complete branches.
- De 16-MiB-claim is nu eerlijk beperkt tot `OUT` en `FAILED`; `QUAR` is expliciet forensisch bewijs buiten die cap.

## Blokker 1 — type/base/size-faultfixture test het verkeerde ownershippad

In productie (`r0` regels 117–119) wordt een non-null host-USM pointer na de alloc-statuscheck aan `self.allocations` toegevoegd. Pas daarna lopen de drie `clGetMemAllocInfoINTEL`-calls en de type/base/size-attestation. Een fout in type/base/size moet dus het **promoted allocationpad** en release-identiteit `usm:<buffernaam>` testen.

`atomic_ownership_faults()` regels 63–68 roept alleen de hostalloc-wrapper aan, voegt niets aan `self.allocations` toe en raiset daarna handmatig voor alle vier labels `status/type/base/size`. Daardoor testen alle vier gevallen hetzelfde **pending-USM-pad** (`pending_usm:4096`). De preregistreerde post-allocation type/base/size-failure en diens promoted cleanup/release-identiteit worden niet nagebootst.

Minimale reparatie: statusfailure blijft pending; type/base/size-fixtures moeten eerst dezelfde named allocation tuple promoveren als productie, vervolgens ieder specifiek attestationpunt laten falen, en exact `usm:<name>`, één freepoging, correct ownershipbewijs en nul live resources eisen. Eén generieke naam volstaat voor padsemantiek; als “every host allocation” letterlijk blijft staan, herhaal dit voor alle 14 bufferposities.

## Blokker 2 — zestig succesvolle statusreturns mogen nog willekeurige integers zijn

De 95-row ownershipgate eist voor iedere return alleen `isinstance(..., int)`. Exacte equality is toegevoegd voor 7 creates, 14 hostallocs en 14 frees, maar niet voor:

- 42 `clGetMemAllocInfoINTEL`-returns;
- 18 `clSetKernelArgMemPointerINTEL`-returns.

Alle zestig moeten op een positief pad exact `0` zijn. Een opgeslagen ownershipledger waarin één van deze returns post-hoc naar bijvoorbeeld `5` is gewijzigd, blijft nu door runner en onafhankelijke verifier geaccepteerd. De mutationlijst raakt alleen een create-return en sluit dit niet.

Minimale reparatie: eis exact nul voor alle get-info-, set-pointer-arg- en free-statusreturns in runner én verifier; voeg minstens één mutation per status-APIklasse toe.

## Niet-fatale resterende teststerkte

De lifecyclefixtures bereiken de bedoelde productiebranches, maar meerdere branches controleren slechts exitcode/mapaanwezigheid. Exacte kind/stage/disposition, `device_opened`, cap, OUT-afwezigheid en immutabilityhash worden niet in elke branch afzonderlijk geassert. Dit is geen derde mechanismefout, maar aanscherping is aanbevolen bij dezelfde kleine preflightrevision.

## Conclusie

R5 is dicht bij autoriseerbaar en heeft geen gevonden wetenschappelijke regressie. De huidige preflight mag nog niet worden uitgevoerd onder het frozen R5-contract: eerst de promoted attestationfaultfixture en de zestig zero-statusgates/mutaties sluiten. Daarna volstaat een beperkte R5A source/preflight-herfreeze; er is geen reden om codec, kernels, payload of fysieke backend opnieuw te ontwerpen.
