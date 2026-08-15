# PH1 Intel execution R6 — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron-, lock- en diff-audit. Geen preflight, payload, compiler of device uitgevoerd.

## Verdict

**GO voor exact één uitvoering van de bevroren no-device statische preflight, en uitsluitend die preflight.** Dit is nog geen toestemming voor payload- of device-uitvoering. Beide R5-blockers zijn exact en niet-vacuüm gesloten; er is geen wetenschappelijke of lifecycle-regressie gevonden.

Exact beoordeelde freeze:

- runner `ebd4444b254c597ec4b07ff847871a9ad6561530020f6d453cf2c7e534b2f025`;
- backend `8bbfa1a69caef5bb78f0a320f3f9093d2e778fa7f8ed67f8e67026ed0b87861f`;
- common `d6abe5792e3069c15cef87f8b8550bb8d9893f992fd7bb93a71e0264d34890e1`;
- verifier `b6e4909fcaf4a9113b3682bb2a2c6efbe1ca744f9de1bf480412dbac9f81d041`;
- preflight `363682edd5d9ccfba380dd6f3e887e04c64e951568c8b943af0ba82ac72a46ba`;
- preregistratie `69ff4293e8c600cce2fdd765f1fbbd53f4922a7eca5baf62cceead1722ccb954`;
- lock `fb31ae3071168483dab5792f0817c91643e75a7b357c74bbb188f56caaaf4657`.

De lock is gesloten/PENDING en bindt de R5-audit `df59ed6d…`; R6-output en statisch preflightresultaat zijn afwezig.

## Reparatie 1 — promoted attestationcleanup: PASS

De fixture behoudt het productieonderscheid:

- `status`: pointer blijft pending, release exact `pending_usm:4096`;
- `type`, `base`, `size`: pointer wordt vóór de geïnjecteerde fout als named allocation gepromoveerd, release exact eenmaal als `usm:attest_<field>`;
- promoted gevallen veroorzaken geen tweede pending-free;
- iedere case eist exact één free van pointer `4096` en nul live resources.

Dit volgt de R0-volgorde waarin allocstatus vóór `self.allocations.append(...)` wordt gecontroleerd en alloc-info/attestation erna.

## Reparatie 2 — zero-status ownershipbewijs: PASS

Runner en onafhankelijke verifier selecteren de 42 get-info- plus 18 set-pointer-rows, eisen exact 60 rijen en `returned == 0` voor iedere rij. De exacte 95-API-volgorde en extensioncounts behouden daarnaast de afzonderlijke cardinaliteiten 42/18. De statische verifierfixture muteert één return in elke APIklasse naar `-5`; beide moeten verification laten falen.

## Regressiecontrole

- Common is exact de eerdere frozen hash; codec, bronnen, input, LUT, buffers, kernels, launches, thresholds, identiteit en claim zijn ongewijzigd.
- Backendwijziging is naam-/revisioneel; het production ownership- en cleanupmechanisme is onveranderd.
- R5-crosslinks, alignment/event/releasegates, lifecyclebranches, resources, bundletransactie, capgrens, controls en onafhankelijke numerieke replay blijven aanwezig.
- Autorisatie blijft gesloten; de huidige audit opent alleen het statische, no-device preflightpad.

## Toegestane volgende stap

Voer exact `preflight_het_next_l0_ph1_intel_execution_r6.py` eenmaal uit terwijl de freeze gesloten blijft. Alleen een volledig PASS-resultaat met onveranderde hashes mag naar een aparte authorization-only revision leiden. Geen payload-, compiler- of OpenCL/devicecall is door dit rapport geautoriseerd.
