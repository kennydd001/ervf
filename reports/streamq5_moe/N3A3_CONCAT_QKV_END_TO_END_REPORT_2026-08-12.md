# N3A3 — concat-QKV same-runtime integratie

Datum: 2026-08-12. Status: **exact, directioneel positief, formele speedgate
gefaald**.

## Uitkomst

De N3A2 `concat_qkv`-kernel veranderde geen enkel gecontroleerd semantisch bit
over 128 gepaarde fysieke P13-decodes. De volledige runtime werd licht sneller,
maar niet genoeg voor de vooraf vastgelegde 2%-drempel op mean én p50.

| Metriek | baseline | kandidaat | ratio | vereiste | besluit |
|---|---:|---:|---:|---:|---|
| mean | 56,2199 ms | 55,6267 ms | 0,98945 | ≤0,98 | faalt |
| p50 | 55,8209 ms | 54,9292 ms | 0,98402 | ≤0,98 | faalt |
| p95 | 76,6981 ms | 75,8622 ms | 0,98910 | ≤1,00 | slaagt |

De gemiddelde absolute gepaarde delta was −0,5932 ms per token. De kandidaat
was sneller in 60,7% van de 112 getimede paren. De mean delta bleef negatief in
beide uitvoervolgordegroepen: −0,4459 ms voor even en −0,7404 ms voor oneven
stappen. Dat ondersteunt een kleine directionele winst, maar verandert de
gesloten passdrempel niet.

## Exactheid

Alle zes poorten slaagden voor ieder van de 128 paren:

- prediction;
- fysiek aantal expertcachemisses;
- volledige KV-digest;
- dynamische LRU-toestand;
- SHA256 van alle logits;
- SHA256 van de uiteindelijke state.

Baseline en kandidaat startten per token uit dezelfde runtime-snapshot. Even
stappen draaiden baseline→kandidaat, oneven stappen kandidaat→baseline. Alleen
een derde canonieke baseline bestuurde de volgende tokeninput.

## Besluit

`concat_qkv` blijft een bewezen fysieke componentoptimalisatie: N3A2 mat een
13,42% p50-winst op precies de attention-inputflow. In de volledige P13-decoder
verdunt dat tot ongeveer 1,1% mean en 1,6% p50. Het is dus geschikt als kleine,
exacte kernelverbetering, maar N3A3 bewijst geen afzonderlijke end-to-end
doorbraak onder de vooraf gekozen materialiteitsgrens.

Er is niet opnieuw gemeten, hertuned of een zwakkere drempel gekozen.

Auditspoor:

- `N3A3_CONCAT_QKV_END_TO_END_PREREGISTRATION.md`;
- `scripts/streamq5_moe/run_n3a3_concat_qkv_end_to_end.py`;
- `n3a3_concat_qkv_end_to_end.json`;
- `scripts/streamq5_moe/verify_n3a3_concat_qkv_end_to_end.py`;
- `n3a3_concat_qkv_end_to_end_verification.json`.

Claimgrens: 128-token same-runtime P13-integratie op één GPU en één domein; geen
10K-endurance, tweede model/GPU, energie-, nieuwheids- of SOTA-claim.
