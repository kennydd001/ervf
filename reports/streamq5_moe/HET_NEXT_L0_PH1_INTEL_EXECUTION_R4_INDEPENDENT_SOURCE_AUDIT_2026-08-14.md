# PH1 Intel execution R4 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron-, lock- en callgraph-audit. Preflight, payload, compiler en device zijn niet uitgevoerd.

## Verdict

**NO-GO voor uitvoering van de huidige no-device preflight.** De uiteindelijke ownership-adjudicatie sluit het eerdere hoofdgat in de positieve result/verifier, maar de bevroren preflight bewijst de preregistreerde post-return ownership- en failurepaden nog niet non-vacuous. Er blijven bovendien enkele independently unchecked succesvelden.

De exact beoordeelde freeze is intern hashconsistent:

- runner `06cf9f08ca63ca160a4b947ce8eb457c2da54ec1b3766d58def72125d9dee260`;
- backend `5113a7905c38038be1285635e255b4e18f8efd78646a4c7152d9e159d802c6e4`;
- common `d6abe5792e3069c15cef87f8b8550bb8d9893f992fd7bb93a71e0264d34890e1`;
- verifier `85681a7ae8841c5814047523d4288853e844247beee04d587b90fb528f6123c7`;
- preflight `68c10d6ab90fe4006c00869c40e8037e56174e06bec540feb3c0dbe57db17f2c`;
- preregistratie `589d120c702bcd087264c22be6bbc83dd755c5e23dac2a840f0a3517f30280b2`;
- lock `b9c0fafeb9b2fbde0b40b1f613e98154b0b20370d05bfd592e274fcedc5c18c9`.

Lock is gesloten (`execution_open=false`, token `PENDING`); zowel output als preflightresultaat is afwezig.

## Werkelijk gesloten

- Context, queue en program worden direct na een non-null return owned; kernels en host-USM worden vóór geërfde status-/attestatiechecks pending-owned.
- Cleanup probeert pending en promoted resources door, ook na een releasefout, en bewaart code/exception/ownership.
- Runner en onafhankelijke verifier eisen nu exact 95 ownership-rijen: 3 creates, 4 kernel-creates, 14 hostallocs, 42 alloc-info calls, 18 pointer-args en 14 frees; API-volgorde, attempted/exception/returntype en pending-classificatie zijn gated.
- Authorization failure blijft read-only. Telemetrie is nonthrowing en secundaire errors overschrijven de primaire devicefout niet.
- Attemptgrootte wordt vóór failurepromotion gemeten; een oversized attempt gaat naar quarantine en `FAILED` houdt alleen bounded summary-evidence.
- De verifier herberekent de 12 samplevolgorde, QPC-monotoniciteit, veldschema en de exacte retained-peak-summary.

## Blokker 1 — post-return ownershipfouten zijn niet getest zoals gepreregistreerd

De preregistratie vereist statische tests voor een non-null create/alloc-resultaat dat daarna door een geïnjecteerde statusfout of mislukte attestation faalt. `atomic_ownership_faults()` test dit niet:

- regels 47–53: alleen normale non-null returns;
- regels 54–57: een createcall die vóór return zelf werpt;
- regels 58–66: normale pending-USM/context-cleanup.

Geen test laat context, queue, program of kernel non-null retourneren en laat vervolgens de geërfde error/statuscheck falen. Geen host-USM-test injecteert na een non-null pointer een type-, base- of size-attestationfout. Daardoor is de centrale R4-reparatie production-plausible maar niet conform het frozen fault-testcontract bewezen.

Minimale reparatie: executeer productiegetrouwe faultfixtures voor context/queue/program/elke kernelpositie en host-USM met non-null+statusfailure; voor USM ook type/base/size-failure. Eis exacte pending/promoted ownership, alle toepasselijke releasepogingen en nul live resources.

## Blokker 2 — ownership-returnidentiteiten zijn niet gekruist of gemuteerd

De nieuwe 95-row gate controleert voor `returned` alleen `isinstance(int)`. Hij koppelt create-returns niet aan context/queue/program/kernelhandles en hostalloc-returns niet aan de 14 geattesteerde allocationpointers. Een ownership-row met een verkeerd maar integer returnhandle kan dus onafhankelijk PASS blijven.

Minimale reparatie: bind iedere returned handle/pointer aan de corresponderende hoofdledger-resource; eis `0` voor succesvolle statuscalls; voeg wrong-return, duplicate, reordered en wrong-`registered_pending` verifiermutaties toe.

## Blokker 3 — enkele verplichte succesvelden blijven niet onafhankelijk gated

De onafhankelijke verifier mist nog:

- `event_requested is False` op alle vier launches;
- het vastgelegde allocationveld `alignment == 4096` (pointermodulo alleen is niet hetzelfde bewijs);
- `owned_before is True` op alle 21 releases.

De preflight muteert geen van deze velden en muteert ook de ownershipledger zelf niet. De runner controleert `event_requested`, maar dat vervangt geen onafhankelijke herberekening.

## Blokker 4 — failure/telemetry/cap-paden blijven vacuüm in de preflight

`lifecycle_sim()` voert alleen authorization-failure en een reeds gecommitteerde negative uit. Er is geen productiepadtest voor start-RAM failure, payload failure, telemetryfailure naast een primaire devicefailure, gewone post-device failure, oversized-attempt quarantine of de bounded summary/cap. Dit zijn expliciete R4-lifecycleclaims.

Minimale reparatie: TEMP-routeer alle roots en injecteer deze paden via `execute_authorized()`/`main()`, met exacte exit/disposition, behoud van primaire fout, bounded `FAILED`, quarantine-summary en geen wijziging van een bestaand commit.

## Claimgrens

De 16-MiB-grens geldt volgens de preregistratie voor wat onder de `failed-attempts`-root blijft; een grote quarantinekopie valt buiten die artifactcap. Dat is verdedigbaar, maar is geen totale disk-cap.

## Conclusie

De finale R4-productiecode is substantieel sterker en de nominale ownershipledger is nu positief geadjudiceerd. De bevroren statische preflight is echter nog niet voldoende om juist de gerepareerde post-return ownership- en failuresemantiek te bewijzen. Eerst een beperkte preflight/verifier-revisie met bovenstaande concrete fixtures en cross-checks; daarna opnieuw frozen source-audit. Dit verdict opent noch de huidige preflight noch een fysieke run.
