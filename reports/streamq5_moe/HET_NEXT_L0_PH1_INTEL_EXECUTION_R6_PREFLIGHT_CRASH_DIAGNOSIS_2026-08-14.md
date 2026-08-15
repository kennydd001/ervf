# PH1 Intel execution R6 — preflightcrash diagnose

Datum: 2026-08-14  
Methode: read-only bronanalyse; geen preflight, payload, compiler of device uitgevoerd.

## Root cause

De crash is een fixturevormfout, geen Intel-, kernel- of wetenschappelijke failure.

`verify.linear(w, x)` implementeert uitsluitend de twee production-reducties:

- `c == 2048`: `vc=32`, boom `16,8,4,2,1`;
- `c == 512`: `vc=8`, boom `4,2,1`.

De frozen R6-preflight geeft in `verifier_mutations()` gate/up `(64,64)` en down `(1,64)`. Dan is `vc=1`, maar de else-boom probeert meteen `old[i+4]`; dat veroorzaakt deterministisch `IndexError`.

## Tweede latente vormfout

Ook `codec_oracle_fixtures()` gebruikt `w.shape == (1,64)` en `x.shape == (64,)` en roept dezelfde vaste `verify.linear()` aan. Deze sentinel is eveneens ongeldig. Afhankelijk van evaluatie-/tracecontext kan dit de eerste crash zijn; beide 64-wide callsites moeten in dezelfde R6P-revisie worden gerepareerd.

## Exact aanbevolen R6P-reparatie

1. Laat production `verify.linear()` ongemoeid.
2. Maak de standalone width-8 sentinel legaal:
   - down-branch: `w=(1,512)`, `x=(512,)`, alleen de eerste acht waarden `0x3f80`; verwachte BF16-som blijft `0x4100`;
   - aanbevolen extra gate/up-branch: identieke `w=(1,2048)`, `x=(2048,)`, zelfde eerste acht waarden en verwachte `0x4100`.
3. Maak `verifier_mutations()` intern production-shape-consistent:
   - gate/up weights `(512,2048)`;
   - down weights `(2048,512)`;
   - input 2048 BF16 woorden / 4096 bytes;
   - gate/up/silu/activation elk 512 BF16 woorden / 1024 bytes;
   - down 2048 BF16 woorden / 4096 bytes;
   - gate/up/activation counters elk 512 little-endian `uint32(1)` / 2048 bytes;
   - down counters 2048 `uint32(1)` / 8192 bytes;
   - gebruik de exacte frozen 14-row `BUFF`, 18 args en vier launches.
4. Houd weights/input nul zodat outputs deterministisch nul zijn en de fixture geen resultaat-afgeleide tuning introduceert. Laat de echte verifier de full-shape outputs zelf herberekenen; monkeypatch `verify.linear` niet.
5. Herbereken fixture-STAGE-hashes uit exact deze volledige outputbytes en behoud alle bestaande ownership/resource/control/provenance mutations.

## Aanvullende latente checks vóór uitvoering

- De huidige vier counteroutputs bevatten slechts één `uint32(1)` per stage, terwijl hun ledgerbytes production-size claimen. Bij full-shape repair moeten de counterpayloads werkelijk 2048/2048/2048/8192 bytes zijn.
- `mini_buff` claimt nu `natural_input=128`, gate/up/silu/activation `128`, down `2`, terwijl de production launch/cardinaliteit en fixed verifier-oracle 4096/1024/1024/1024/1024/4096 vereisen. Vervang de hele tabel; pas niet alleen de weightshape aan.
- Behoud recordbuffers en specs op de bestaande 675840-byte productionvorm; anders breken de hard-coded record checker offsets `64:655424:671808`.
- Voeg in de R6P-preflight een expliciete fixture-schema-assert toe vóór `verify_dict`: exact outputlengtes, counteraantallen, inputlengte, weightshapes en BUFF-som. Zo wordt een volgende mismatch een gerichte preflight-negative in plaats van een exception.
- De eerder geaudite R6 two-point repairs zelf blijven methodologisch geldig; alleen de synthetische verifierfixture is niet uitvoerbaar.

## Verdict

R6-preflight blijft **NO-GO**. Maak een immutable preflight-only R6P-revisie met beide 64-wide callsites en alle afhankelijke buffer/countervormen samen gerepareerd. Daarna opnieuw source-audit; geen fysieke code of wetenschappelijke gate hoeft te veranderen.
