# S14 — GPU-event-tijdslijn van de MoE-laag in de lus: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.** Geschreven ná S13 (bouwen weerlegd), S12
(attributie dekt 15,53 van 39,52 ms) en S8 (geïsoleerd timen telt 16,9 ms te
veel door het sync-artefact).

## 1. Vraag

Van de MoE-term bij 262K (39,523 ms/token, S8) dekken de schone in-lus
marginalen van S12-R1 samen 15,53 ms (`down` 7,478 + `up` 4,756 + `shared`
3,298). De rest — ~24 ms — heeft bewust geen naam gekregen. S12 heeft uitgesloten
dat hij op één plek zit die een kernelherschrijving weghaalt, maar niet *waar*
hij dan wel zit. Kandidaten die de marginale methode principieel niet kan zien:

- **wachttijd van de compute-stream** op de copy-stream (`wait_event` bij
  misses) — repliceerbaar werk is er niet, dus de probe meet het niet;
- **host-tijd tussen kernel-launches** (route-readback, LRU-boekhouding,
  Python-lus overhead) terwijl de GPU niets te doen heeft;
- **trager lopen van dezelfde kernel in de lus** dan in de microbenchmark
  (koude L2, copy-stream die HBM deelt).

## 2. Methode — timestamps, geen syncs

Een subklasse van `LightningRuntime` in het runnerscript overschrijft
`_moe_cached` met een kopie van dezelfde code waarin **CUDA-events met timing**
tussen de fasen staan. Een event op een stream is een timestamp, geen
synchronisatie: de overlap die S8's geïsoleerde meting vernietigde blijft
intact. `runtime.py` wordt niet aangeraakt; de verifier controleert de hash.

Segmenten per MoE-laag (elke microseconde van de laag valt in precies één
segment, want segmenten zijn opeenvolgende event-paren op dezelfde stream):

1. `route` — router-GEMV + sigmoid + argsort + pack;
2. `shared_up`, `shared_dn` — de shared expert (gelanceerd vóór de readback,
   dat is de bestaande overlap);
3. `readback_host` — host-wandtijd van de `cp_asnumpy`-readback (dit is de
   échte sync van de lus; hij wordt niet extra gesynchroniseerd, alleen
   omsloten door `perf_counter`);
4. `pre_first_expert` — stream-tijd van einde `shared_dn` tot einde van de
   eerste expert-`up`-GEMV: bevat de readback, de LRU-boekhouding, het
   issuen van miss-copies en de eerste GEMV zelf;
5. per expert `up`, `down_masked`, `accum` — stream-spans;
6. `miss_copy_batch` — span op de **copy-stream** van het issuen van de
   miss-transfers (gemeten op de copy-stream, dus zichtbaar naast en niet in
   de compute-stream);
7. `post_last_expert` — einde laatste `accum` tot einde van de laag.

Contexten 0 en 262.100 via dezelfde pos-sprong als N7-B/S12 (64 echte tokens,
`rt.pos = target`, 32 warm, 16 gemeten). Dezelfde `varied`-tokenstroom
(seed 11). Capacity 72 (productieconfiguratie; de probe heeft geen
kladruimte nodig, alleen events).

## 3. Poorten

- **G-S14-C1 — semantiek.** De geïnstrumenteerde subklasse genereert
  bit-identiek aan dezelfde runtime zonder instrumentatie (2 prompts × 32
  tokens, de S12-prompts).
- **G-S14-P1 — probelast.** `p50(probed) − ½(p50(base₀)+p50(base₁))` per
  context wordt gerapporteerd. Is die > 20% van de basislijn-p50, dan wordt de
  fase **niet-conclusief** afgesloten in plaats van geherinterpreteerd —
  dezelfde discipline als S12's driftpoort.
- **G-S14-S1 — boekhouding.** Per context: de som van alle segmentgemiddelden
  over 23 MoE-lagen ≤ de gemeten token-p50 van de probed arm; en die som ≥ de
  helft van de S8-MoE-term (39,523/2 bij 262K, resp. de ctx-0-MoE-term uit de
  eigen unprobed meting is niet bekend — de ondergrens geldt alleen voor 262K).
  Negatieve segmenten (< −0,01 ms) betekenen kapotte event-ordening → geen
  resultaat.

Er is **geen prestatiepoort**: dit is een attributiefase. Zij bouwt niets en
stelt niets voor. Haar uitkomst is een tabel die zegt waar de ~24 ms zit, met
per segment een onder- of bovengrens-etiket.

## 4. Wat de segmenten wel en niet betekenen (vooraf gelegd vast)

- Stream-spans zijn wandtijd op de stream: een span bevat zowel rekenwerk als
  wachten op events. `pre_first_expert` is daarom bewust grof — hij is de
  kandidaat-container voor de host-gebonden tijd die de marginale methode niet
  zag.
- Per-expert `down_masked` in `up_only`-modus bevat de sparse host-gather over
  mapped geheugen: PCIe-wachten zit ín die span, zoals ontworpen.
- Segmenten zijn geen onafhankelijke componentkosten en worden niet naar tok/s
  omgerekend. De som is per constructie ≤ de tokentijd.

## 5. Verificatie

De onafhankelijke verifier importeert de runner niet. Hij herberekent alle
segmentgemiddelden uit de ruwe per-token segmentlijsten met een eigen
mediaan/gemiddelde, her-evalueert de drie poorten, controleert dat
`runtime.py` op de input-lock-hash staat, en controleert dat de ruwe data per
context uit 16 tokens komen.

## 6. Artefacten (te produceren)

`scripts/lightningstream_nemotron/s14_moe_layer_timeline.py` ·
`s14_moe_layer_timeline.json` ·
`scripts/lightningstream_nemotron/s14_independent_verify.py` ·
`s14_independent_verification.json` ·
`S14_MOE_LAYER_TIMELINE_REPORT_2026-08-15.md` · `s14_input_lock.json`
