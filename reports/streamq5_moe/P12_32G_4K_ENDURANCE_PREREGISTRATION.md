# P12 preregistratie — 32 GiB, 4K-context en 10K-token-duurtest

Datum: 2026-08-12. Status bij vastlegging: geen P12-output geopend.

## Hypothese

De geselecteerde P7 ERVF-16-runtime kan in één proces onder een echte 32-GiB
process-commitlimiet een volledig 4.096-token KV-context vullen en vervolgens
10.000 totale feedbacktokens afwerken zonder OOM, paginginstorting of thermische
staartregressie.

## Protocol

- Windows Job Object met `JOB_OBJECT_LIMIT_PROCESS_MEMORY = 32 GiB`, toegewezen
  vóór bank- en CUDA-allocatie.
- Exacte P7C-runtime, Q5-bank, Q8-trunk, cachebeleid en greedy feedback.
- Eerste segment: 4.096 opeenvolgende posities en volledige KV-digest.
- Daarna contextreset op de bestaande 4K-grens en doorgaan tot 10.000 decode-
  calls (de `10K tokens`-tak van “60 minuten of 10K”, wat eerst komt).
- Log proces-RSS/commit/pagefaults en GPU-temperatuur/vermogen per 250 tokens.

## Gates

- Joblimiet daadwerkelijk toegewezen; peak commit `<= 32 GiB`; geen OOM.
- Exact 4.096 posities × 48 KV-laagschrijvingen, niet-lege 4K-KV-digest.
- Gehele run mean `<= 100 ms`, p95 `<= 150 ms`, p99 `< 110 ms`, `>= 10 tok/s`.
- Laatste 1.000-token mean en p95 maximaal 110% van de eerste 1.000 (na de
  eerste 16 cold tokens); geen pagefile-groei van meer dan 256 MiB.

