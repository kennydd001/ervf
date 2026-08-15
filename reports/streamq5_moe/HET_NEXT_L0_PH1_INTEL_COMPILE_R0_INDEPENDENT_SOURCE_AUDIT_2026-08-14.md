# PH1 Intel compile-only R0 — onafhankelijke source-audit

Datum: 2026-08-14  
Methode: strikt statisch/read-only ten opzichte van de frozen kandidaat; geen import, preflight, OpenCL- of device-call.

## Frozen bindings

- backend `1c70d4248bdf64404589916a6be624594e8343442a64c57e926e52926f51ceac`;
- runner `b21da93f0d401e7e3ef0909df8db7140c84d9d4c6380471d6b95a8560a6001cc`;
- preflight `9c82847031dee20c69acc347130b52157b4cb814fd20ab0e605a28af6660b834`;
- prereg `ef88e57f3d66b70de3cf45f83460f1be66f0cf712601af1f4010c56e9471bfa9`;
- source literal `98415386a96e2cac5de6f2792ab9b89b51e4ae76a7551dda03b9454de7f52394`;
- closed compile-lock `883794b8350f19d104884e4a37bb87a7992d8829c296790465f7046bb5f6f819`, `execution_open=false`, token `PENDING`.

De outputdirectory is afwezig.

## Verdict

**NO-GO voor de static preflight/open-revision route.** Vier blokkers moeten in een nieuwe immutable revisie worden gesloten.

### 1. OpenCL extension-pragmas zijn niet canoniek

Backendregels 54–55 gebruiken `cl_intel_required_sub_group_size` en `cl_khr_int64`. Het Intel-extensiontoken bij `intel_reqd_sub_group_size` is `cl_intel_required_subgroup_size` (zonder `_sub_group_`), terwijl `cl_khr_int64` geen passend desktop-OpenCL-extensiontoken is. Een compiler mag onbekende extension-pragmas negeren of diagnosticeren. Een enige fysieke compilepoging mag niet met deze bekende bronambiguïteit starten. Corrigeer het Intel-token en verwijder het overbodige/ongeldige int64-pragma; bind daarna de nieuwe sourcehash.

### 2. De static preflight is methodologisch vacuüm voor de kernclaims

De prereg vereist negatieve source-mutations en bewijs van entrypoint/cardinaliteit/buffer/launch/compile-only-eigenschappen. De preflight:

- berekent de elf BF16-vectorantwoorden alleen met zijn eigen `Fraction`-oracle, maar voert of emuleert `multiply_bf16_exact` uit de OpenCL-source niet;
- voert geen enkele negatieve source-mutatie uit;
- zoekt slechts enkele strings voor geometrie;
- inspecteert de `compile_only`-callgraph niet op afwezigheid van queue-, kernel-, event-, USM-, buffer-, payload- en launchcalls;
- laat `no_payload_import` alleen over de preflight zelf lopen, niet over backend en runner;
- controleert zijn eigen SHA niet tegen `lock.preflight_sha256`.

Hierdoor zou een betekenisvol defect in de source of een verboden call in `compile_only` toch groen kunnen blijven. R1 heeft een onafhankelijke source-routine-emulator/testvectorharnas, AST/callgraph-allowlist en gerichte mutatietests nodig.

### 3. Een lege program binary kan formeel positief worden

`compile_only` leest `CL_PROGRAM_BINARY_SIZES` en de bytes, maar vereist nergens `sizes[0] > 0`. De runner zet na iedere succesvolle return onvoorwaardelijk `status=compile_positive` en `positive=true`. Dit schendt de expliciete prereg-gate “nonempty binary”. Vereis vóór success exact één binary met grootte >0, consistente querygrootte en digest over precies die bytes.

### 4. De transactie heeft geen robuust post-device failure/recovery-pad

`FAILED` is gedeclareerd maar ongebruikt. Fouten tijdens artifactwrites, manifest/commit of promotion vallen buiten de runner-`try`, kunnen een `.inprogress`-directory achterlaten en produceren geen immutable failure/quarantine-evidence. Er is evenmin voorafgaand stale-temp recovery. De open revision moet stale/corrupt state vóór de device-open behandelen en elke post-device write/promotion failure fail-closed archiveren zonder een valid commit te wijzigen.

## Wat wel sluit

- `compile_only` bevat in zijn nominale pad werkelijk geen queuecreatie, kernelcreatie, USM/bufferallocatie, payloadread of launch; het opent alleen DLL/enumeratie/context/program/build/binary en sluit program/context.
- De 2048- en 512-koloms width-8 reductievolgorde, row mapping, launchgeometrie en counterwrites zijn intern consistent.
- De bit-level BF16-multiply, inclusief subnormalen, signed zero en `(7f7f,0001)->3cff`, is consistent met het PH1-R2-contract.
- Build-log en binary worden als raw bytes behouden; program- en contextrelease worden na een normale compilepoging beide geprobeerd.
- De gesloten lock, runner/backend/prereg/source/CPU-evidence/R1/R2-bindings zijn coherent.

Een herstelde revision vereist opnieuw source-audit. Pas daarna mag exact de verbeterde static preflight worden uitgevoerd; deze audit autoriseert geen OpenCL- of device-call.
