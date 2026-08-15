# Onderzoekslogboek

Eén blok per fase, **nieuwste bovenaan**. Schrijf hier ook wat er *niet* werkte
en waarom — dat is meestal het bruikbaarste deel. Formaat:

```
## <datum> — <fase> — <verdict in één zin>
**Vraag** · **Opzet** (armen, één variabele) · **Uitkomst** (getallen) ·
**Poorten** · **Wat dit sluit of opent** · **Artefacten**
```

---

## 2026-08-15 — A1 — de bewezen stack staat nu default aan, en het anker is opnieuw bevroren

**Vraag.** E6 mat dat ERVF + v4 + D1 sneller én exact is. Mag dat de default worden?

**Waarom niet meteen.** E6 vergeleek twee armen binnen één proces met dezelfde
cachegeschiedenis — precies het regime waarin de exactheid vóór D1 óók léék te
kloppen. Vier fasen (NERVF-3, NERVF-4, E4, S11) haalden hun pariteitspoort over
2×64 tokens terwijl de runtime niet deterministisch was. Een adoptie mag niet op
een blinde test rusten.

**Opzet.** Verander de **cachecapaciteit** (72 vs 56). Dat verandert het
hit/miss-patroon op vrijwel elke laag van elke token, dus de optelvolgorde, zonder
een gewicht/route/kernel aan te raken. Plus een **controle-arm die moet falen**:
dezelfde vergelijking zonder D1.

**Uitkomst.** Met D1: identiek over 2 × 256 tokens. Zonder D1: divergeert
(expository, token 224; narrative niet). De test heeft dus vermogen — maar smal,
en dat verklaart waarom de fout vier fasen lang onzichtbaar bleef.

**Poorten.** G-A1-CAP ✅ · G-A1-CTL ✅ (faalde zoals vereist) · G-A1B-DEFAULT ✅
(runtime zonder enige vlag reproduceert bit-identiek wat A1 mat) · G-A1B-FLAGS ✅.

**Wat dit opent.** Defaults om: `use_ervf=True`, `rt.attn=attention_fp8_gqa4`,
`deterministic_accum=True`. De attentiekernel wordt nu via `rt.attn` gekozen;
oude scripts die `rt.k.attention_fp8_gqa` overschrijven meten voortaan een
nul-verschil in plaats van stil iets verkeerds — bewust zo gekozen.

**Het anker.** V35 wordt niet meer gereproduceerd, divergentie al bij token 1.
Dat komt van D1, niet van v4 (E4 reproduceerde V35 wél). Het anker legde een
ordeafhankelijk artefact vast. Vooraf vastgelegde regel gevolgd: `V36_DETERMINISTIC_ANCHOR.json`
bevroren, V35 ongewijzigd bewaard, niet-vergelijkbaarheid opgeschreven.

**Artefacten.** `A1_ADOPTION_PREREGISTRATION_2026-08-15.md` ·
`A1_ADOPTION_REPORT_2026-08-15.md` · `A1_ADOPTION_PRECONDITION.json` ·
`A1B_ADOPTION_VERIFY.json` · `V36_DETERMINISTIC_ANCHOR.json`

---

## 2026-08-15 — E6 — geïntegreerd 41,98 → 37,49 ms per token, exact

**Opzet.** Drie armen `base_a / integrated / base_b`, 3 domeinen × 512 causale
tokens, D1 in **alle** armen (zonder D1 zijn twee armen niet eens vergelijkbaar).
Eén variabele: ERVF + v4-attention.

**Uitkomst.** +4,169 (expository) / +4,443 (narrative) / +4,858 (code) ms, elk
boven zijn eigen drift. Bit-identieke uitvoer. VRAM ongewijzigd.

**Poorten.** Exactheid ✅ · latency ✅ · VRAM ✅ · eindpoort ≥50 tok/s ❌ (26,7).

**Wat dit sluit.** Niets — maar het laat zien dat de resterende afstand tot 50
tok/s niet uit de gebouwde componenten kan komen. E2 is weerlegd en E1 fase 2 is
ongebouwd; dat zijn de twee posten die het plan ervoor had ingeboekt.

**Artefacten.** `E6_INTEGRATED_REPORT_2026-08-15.md` · `E6_INTEGRATED_RUN.json`

---

## 2026-08-15 — D1 — de runtime was niet deterministisch, en dat is nu opgelost

**Vondst.** `_moe_cached` accumuleert de zes routed experts in
**hit-dan-miss-volgorde**. Welke expert een hit is hangt van de LRU-staat af, dus
twee runs met andere cachegeschiedenis tellen in andere volgorde op. FP-optelling
is niet associatief → twee armen met **identieke configuratie** divergeerden over
512 tokens. Ontdekt doordat NERVF-5 `base_b` tegen `base_a` zette.

**Ingreep.** Rekenvolgorde en optelvolgorde gescheiden: rekenen blijft hit-eerst
(latencywinst blijft), bijdragen naar aparte slotbuffers, na de lus optellen in
routevolgorde `s = 0..5`. Kosten: `top_k × hidden` floats (64 KB), nul extra
kernels.

**Uitkomst.** `base_b` nu identiek aan `base_a` over 3 × 512 tokens; NERVF-5
slaagt alsnog op alle drie zijn poorten. ERVF-winst met D1 aan: +2,771 / +5,008 /
+5,395 ms.

**De les.** Vier eerdere exactheidspoorten waren waar voor hun eigen run maar
bewezen minder dan ze leken. Vandaar werkregel 8: bouw een controle-arm die moet
falen.

**Artefacten.** `D1_DETERMINISM_REPORT_2026-08-15.md` · `d1_determinism.json`

---

## 2026-08-15 — NERVF-0 t/m 5 — ERVF gerepliceerd op een tweede model

**Vraag.** Reproduceert de Qwen3-30B ERVF-doorbraak op een architecturaal andere
NVFP4 hybrid-Mamba MoE?

**Antwoord: ja.** 1,936× bitexact op het projectievlak, bij **dezelfde** gekozen
subwarp-breedte 16 die Qwen selecteerde, op een ander model, een andere
quantisatie (NVFP4 vs Q5/Q8) en een andere shape.

**Het mechanisme.** w-lane subwarps per rij, 256/w rijen per 256-thread block,
per lane aparte virtuele accumulatoren, en een reductie die de DAG van de
referentiekernel **exact** reconstrueert — de eerste butterfly-stap (offset 16)
wordt een lane-lokale optelling, offsets 8/4/2/1 blijven shuffles binnen de
subwarp, en de acht warp-sommen vouwen in registers in exact de volgorde
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

**Fout onderweg.** w=4 en w=8 gaven eerst 72/72 mismatches doordat ik de virtuele
accumulatoren *sequentieel* vouwde in plaats van in butterfly-volgorde. Na de fix
alle vier breedtes bitexact.

**NERVF-1 valstrik.** RAW_SCAN sprong van 9,77 naar 51,67 µs tussen runs: het
2,81 MiB record past in L2. Opgelost door alle armen door een 254 MiB pool van 95
replica's te cyclen → spreiding 0,1%. Elke bandbreedtemeting hierna doet dit.

**NERVF-4.** Weerlegd, zie E2.

**Poorten.** Doorbraakladder LEVEL 2 gehaald (≥1,35× exact). LEVEL 3 niet: het
volledige expertpad bevat de down-projectie, die ERVF niet raakt en waar NERVF-4
de voor de hand liggende route sloot. LEVEL 4 (≥35 tok/s) niet: 29,45.

**Niet geclaimd.** Geen nieuwheidsclaim — geen prior-art-audit, geen stock
llama.cpp-vergelijking, geen tweede GPU.

**Artefacten.** `NERVF_NEMOTRON_FINAL_REPORT.md` · verifier 66/66 in
`nervf_independent_verification.json`
