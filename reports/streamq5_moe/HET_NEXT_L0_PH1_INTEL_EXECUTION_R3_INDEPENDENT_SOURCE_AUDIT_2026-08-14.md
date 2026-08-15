# PH1 Intel execution R3 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron/lock/callgraph-audit; preflight, payload, compiler en device zijn niet uitgevoerd.

## Verdict

**NO-GO voor de huidige no-device preflight.** R3 herstelt de deterministische R2-preflightcrash en sluit meerdere eerdere gaten werkelijk, maar vier productieblockers en twee niet-vacuous verifier/preflightgaten blijven bestaan.

Alle zeven overgedragen hashes matchen exact. Ook alle `PROVENANCE`-paden matchen de gesloten lock, inclusief R2-audit en beide transitieve R0-bronnen. `execution_open=false`, token `PENDING`, output afwezig en preflightresultaat afwezig zijn bevestigd.

## Correct gesloten sinds R2

- `cleanup_faults()` zoekt nu de cleanup-row op in plaats van foutief `ledger[-1]` te gebruiken; zowel 21 exceptionposities als 21 niet-nul-codeposities worden doorlopen.
- Het nonzero Q5-pad loopt door zowel `common.record()` als de onafhankelijke `verify.codec()`; FMA-, BF16-multiply- en reductiefixtures zijn executable.
- De preflight gebruikt het normale `verify_dict()`-pad met prepared compacte arithmetic data, niet langer de aparte speelgoeddispatch.
- De verifier herhasht alle in `PROVENANCE` genoemde bronnen en artefacten; R0-dependencies zijn inbegrepen.
- Backend, runner en verifier vereisen 14 niet-nul, onderling verschillende host-USM-pointers.
- Package-kind/manifest/commit en retained phase-sampling zijn aangescherpt.

## Blokker 1 — gedeeltelijk succesvolle createcalls kunnen buiten ownership vallen

De werkelijke create-/allocatielogica blijft geërfd uit de R0-backend:

- `clCreateKernel` retourneert eerst `k`, daarna wordt de foutcode gecontroleerd en pas daarna wordt `k` aan `self.kernels` toegevoegd.
- `clHostMemAllocINTEL` retourneert eerst een pointer; de R3-wrapper registreert hem alleen in `host_pointers`, vervolgens controleert R0 de foutcode en voegt hem pas daarna aan `self.allocations` toe.

Bij een non-null handle/pointer samen met een foutcode — of een uitzondering in de wrappercontrole — ontbreekt het object in de cleanup ownershiplijst. `close()` kan het dan niet als poging registreren of vrijgeven. De preflight test uitsluitend reeds volledig geregistreerde 14+4+3 resources; zij injecteert geen fout op iedere create-/attestationstap.

**Reparatie:** registreer ieder teruggegeven non-null object onmiddellijk als `pending_owned`, vóór foutcontrole/attestatie. Promote daarna naar volledig owned of release in dezelfde failure-finally. Voeg faultinjectie toe voor context, queue, program, vier kernels, iedere allocatie en iedere drieledige alloc-attestatie.

## Blokker 2 — de failure-artifactcap blijft detectie ná promotion

Bij een fout nadat de attemptmap bestaat, wordt de volledige attempt eerst door `base.archive(..., attempt)` naar de failuredirectory verplaatst. Pas daarna telt regel 57 de bytes en werpt `failure_artifact_cap`. Als de attempt groter dan 16 MiB is, blijft de oversized failuredirectory dus op schijf staan. Dit voldoet niet aan “success and failure artifacts are capped at 16 MiB”.

**Reparatie:** bepaal vóór promotion de totale attempt+failuregrootte. Bij overschrijding behoud alleen een vooraf begrensde failure-samenvatting plus digests/bytecounts en geef de oversized temp een expliciete fail-closed disposition buiten de geclaimde retained bundle, of definieer en gate een grotere vooraf geregistreerde failurecap.

## Blokker 3 — ongeldige autorisatie kan toch filesystemmutaties veroorzaken

`main()` vangt een fout uit `authorize()` en schrijft daarna via `base.archive()` een nieuwe failuredirectory. Daardoor kan een gesloten, foutieve of gedrifte lock met het publieke CLI-ACK toch persistent state muteren. Dit botst met de autorisatiegrens en met de formulering dat autorisatie vóór iedere mutatie komt. De preflight `lifecycle_sim()` bevestigt juist dit gedrag als positief.

**Reparatie:** een authorization- of ACK-fout moet read-only stoppen met nonzero exit. Pas na geldige autorisatie mag create-new failure evidence ontstaan. Start-RAM- en payloadfouten kunnen daarna wel volgens een gebonden schema worden gearchiveerd.

## Blokker 4 — resource-sampling kan de fysieke fout maskeren

`close()` schrijft eerst cleanup en roept daarna onbeveiligd `self.sample('post_cleanup')` aan. Een psutil-/telemetryfout daar ontsnapt uit `close()` en kan de oorspronkelijke execution-/cleanupfout maskeren. Ook de pre/post-launch wrappers laten een pre-samplefout de launch verhinderen en een post-samplefout een succesvol uitgevoerde launch als generieke failure verschijnen zonder aparte telemetry-disposition.

**Reparatie:** maak telemetry expliciet gated maar exception-safe: bewaar de oorspronkelijke devicefout, registreer telemetryfout afzonderlijk en voer cleanup altijd volledig uit. Preregistreer of telemetryfailure een geldige capability-negative dan wel invalid attempt is.

## Blokker 5 — verifierresourcegate controleert de 12 samples niet inhoudelijk genoeg

De onafhankelijke verifier eist alleen `len(samples)==12`, gebruikt alle `available` waarden in een minimum en vergelijkt één runner-summary `peak_retained_wset` met 12 GiB. Hij eist niet de exacte 12 stagevolgorde, monotone/tijdige QPC-evidence, noch dat iedere sample-`peak_wset <=12 GiB` en de summary exact het maximum van de samples en boundarymetingen is. Runner `ledger_gates()` controleert dit sterker, maar de verifier vertrouwt vervolgens alleen `all(rrr['gates'].values())` in plaats van iedere runnergate te herberekenen.

**Reparatie:** controleer zelfstandig de exacte stagelijst, samplevelden/typen, QPC-order, ieder available/peakveld en `peak_retained_wset == max(boundaries,samples)`. Voeg mutaties toe voor samplevolgorde, samplepeak en samenvattingsinconsistentie.

## Blokker 6 — preflightmutaties dekken nog niet de echte gateoppervlakte

De productiepadfixture is een duidelijke verbetering, maar de negen mutaties raken slechts één allocationpointer, cleanupboolean, één provenancewaarde, één output, één stagehash, controlcardinaliteit, identity, summarypeak en één forbidden teller. Zij testen niet: pointeraliasing, allocation type/base/size/alignment, arg mapping, launch geometry/event, finish/read-order, extension counts, release order/code/exception/ownership, resource-samplestages, manifest/commit van de uiteindelijke resultbundle of negatieve existing-commit lifecycle. `transaction_sim()` gebruikt nog rechtstreeks geërfde helpers in plaats van alle productiebranches.

**Reparatie:** registreer minstens één gerichte mutatie per onafhankelijke check en eis dat precies de bedoelde gate faalt. Test result/manifest/commit en failurepromotion via tijdelijke productiepaths.

## Overige formele begrenzing

De static-AST-afwezigheid van verboden APIs is het geldige bewijs. De runtime `forbidden_calls`-dictionary blijft niet-instrumenterend, omdat verboden calls geen wrapper hebben die ledgerrows kan produceren. Behoud dit uitsluitend als secundaire schema-evidence, zoals de R3-prereg nu terecht aangeeft.

## Vereiste R4-route

Maak resource ownership fail-closed vanaf iedere createcall; bereken failurecap vóór promotion; laat authorization-fouten read-only; maak sampling exception-safe; versterk onafhankelijke resource-replay en mutationcoverage. Daarna is opnieuw frozen source-audit nodig. Alleen een GO opent exact de nieuwe no-device preflight; een PASS daarvan vereist nog een aparte authorization-only revision vóór één fysieke Intel-poging.
