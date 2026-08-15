# E4 in-lus — v4 adoptiemeting (G-E4-T1)

Datum: 2026-08-15
Verdict: **v4 is in de echte lus conclusief sneller: attention 18,211 → 14,554 ms bij ctx 262100 (−3,658 ms, drift 0,544) en het token 55,915 → 51,164 ms (−4,751 ms, drift 3,014). De generatie blijft bit-identiek in alle drie de armen. G-E4-T1 faalt niettemin: hij eist ≤ 6,0 ms attention en meet 14,554. De ankerclausule faalt om een reden die niets met v4 te maken heeft — het S5-anker dateert van vóór de v35-checkpointwissel.**
Terminal state: `e4_inloop_v4_conclusively_faster_absolute_gate_failed`
Preregistratie: `E4_ATTENTION_ROOFLINE_PREREGISTRATION_2026-08-15.md` (G-E4-T1, bevroren)
Voorafgaand: `E4_ATTENTION_ROOFLINE_REPORT_2026-08-15.md`, `HANDOFF_E4_EN_VERDER_2026-08-15.md` punt 3.1

## 1. Methode

CUDA-events zetten tijdstempels op de stream rond elk van de zes
attention-lagen **binnen** de echte decodelus, zoals `s14_moe_layer_timeline.py`,
zodat er geen host-synchronisatie bijkomt en de overlap van de lus intact blijft.
De kernelwissel is één monkeypatch — de wrappers hebben identieke signaturen.

Drie armen, **v1 / v4 / v1**, zodat de herhaling van v1 de drift begrenst. Twee
rondes van 16 gemeten stappen per context per arm.

## 2. Uitkomst

| context | attention v1 | attention v4 | winst | drift | token v1 | token v4 | winst | drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,474 | 2,430 | +0,044 | 0,022 | 38,147 | 37,520 | +0,628 | 2,264 |
| 131.072 | 10,771 | 9,028 | **+1,743** | 0,258 | 46,791 | 45,509 | +1,281 | 1,115 |
| 262.100 | **18,211** | **14,554** | **+3,658** | 0,544 | 55,915 | 51,164 | **+4,751** | 3,014 |

De attention-winst overschrijdt op alle drie de diepten haar eigen drift, dus
alle drie zijn conclusief. Bij het token is dat zo bij 262100 (4,751 tegen
3,014) en bij 131072 marginaal (1,281 tegen 1,115); bij ctx 0 niet (0,628 tegen
2,264) — daar is de winst dan ook klein, want attention is er maar 2,4 ms.

**De in-lus meting bevestigt E4's geïsoleerde cijfers.** Geïsoleerd mat E4
2,803 ms/laag voor v1 en 2,304 voor v4; maal zes is dat 16,82 en 13,82 ms, tegen
in-lus 18,21 en 14,55. Beide binnen 10%, en de voorspelde besparing van ~3,0 ms
komt uit op **3,658 ms** — de voorspelling was conservatief.

## 3. Poorten

| poort | eis | gemeten | |
|---|---|---:|:--:|
| **G-E4-T1 drempel** | attention in de lus ≤ 6,0 ms bij 262100 | **14,554 ms** | ❌ |
| G-E4-T1 stretch | ≤ 4,8 ms | 14,554 | ❌ |
| **G-E4-T1 pariteit** (arm tegen arm) | 2 × 64 tokens bit-identiek | identiek in v1/v4/v1 | ✅ |
| G-E4-T1 pariteit (S5-anker) | eerste 33 tokens gelijk aan het anker | **niet gelijk** | ❌ zie §4 |

**De drempel wordt niet verruimd.** G-E4-T1 vroeg of attention in de lus goedkoop
genoeg is geworden — 6,0 ms is 1,0 ms per laag, hetzelfde niveau als S1. Het
antwoord is nee, en dat was na E4's kernelfase al te verwachten: die stelde de
exacte fp32-vloer op ~1,2–1,5 ms/laag.

Wat de poort **niet** vraagt en wat wel vaststaat: v4 is bit-identiek en
conclusief sneller. De onafhankelijke verifier herbevestigt de bitwise-identiteit
los van de runner, op willekeurige FP8-KV bij t = 64/1024/4096: **3 van 3
identiek**.

## 4. De ankerclausule faalt op een verouderd artefact

Het anker `reports/lightningstream_nemotron/s5_baseline_generation.json` is
bevroren op **2026-08-14T20:02:42Z**. De v35-layoutopname
(`n2r_v35_layout.json`) staat op **20:52:30Z**. Het anker is dus gegenereerd met
**Nemotron 3 Nano**, niet met 3.5 Lightning, en kan per constructie niet matchen.

Zichtbaar in de tekst zelf:

- anker: `' Paris." No extra punctuation? The sentence includes a period at the e'`
- v35: `' Paris.  \nThe capital of Germany is Berlin.  \nThe capital of Italy is '`

Dit is geen v4-defect en ook geen regressie; het is een gate die naar een
artefact van het verkeerde model wijst. De pariteit die er wél toe doet — v1
tegen v4 op het huidige model, 2 × 64 tokens — is gehaald.

**Opgelost voor volgende fasen:** het ontbrekende anker is nu bevroren als
`reports/treesweep200/V35_GENERATION_ANCHOR.json`, uit de v1-arm van deze run,
2 × 64 tokens, met runtime- en kernel-hashes. E2, E5, E1 en E6 kunnen daar tegen
gaten.

## 5. Wat dit betekent voor de adoptiebeslissing

v4 is **strikt beter dan v1**: bit-identieke uitvoer, geen extra geheugen, en
−3,658 ms attention per token bij 262100. Dat is 6,7% van het token daar.

De preregistratie zegt over adoptie: bij gefaalde S1 wordt het profiel gebruikt
om de volgende kandidaat te kiezen. Dat is E4's kernelfase al gedaan (v4 ís die
volgende kandidaat, en de analyse toont dat verder gaan fp16/tensor-cores vraagt,
wat de exactheid breekt). Er is geen gemeten nadeel aan adoptie, en de poort die
faalt is een **absolute roofline-poort**, geen regressiepoort.

Ik zet de default niet om — dat is een productiebeslissing en hoort bij E6's
integratie, waar v4 samen met E1/E2/E5 gemeten wordt. v4 blijft geregistreerd als
bevroren beste exacte kandidaat, nu mét in-lus bewijs.

## 6. Claim boundary

In-lus meting: CUDA-events zetten tijdstempels op de stream rond de zes
attention-lagen binnen de echte decodelus, dus er is geen host-synchronisatie
toegevoegd en de overlap blijft intact. Het attention-getal is een **component**
van het token en is geen doorvoerresultaat; het tokengetal ernaast is end-to-end
wandtijd op deze GPU bij capacity 72. v4 is per constructie bit-identiek aan v1
(E4 G-E4-C1); de pariteitscheck hier bevestigt dat die eigenschap in de lus
overeind blijft over 2 × 64 tokens, en de verifier herbevestigt het los van de
runner op willekeurige FP8-KV. Drie armen v1/v4/v1 begrenzen de drift; een winst
telt alleen waar zij haar eigen drift overschrijdt. Geen kwaliteitsclaim.

## 7. Artefacten

`scripts/treesweep200/e4_inloop.py` · `E4_INLOOP_RESULTS.json` ·
`scripts/treesweep200/e4_inloop_independent_verify.py` (52/52, `VERIFIED`) ·
`e4_inloop_independent_verification.json` · `V35_GENERATION_ANCHOR.json` ·
`reports/lightningstream_nemotron/protected_verification_after_e4inloop.json`
(0 modified / 0 removed)
