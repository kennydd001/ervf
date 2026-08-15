# PORT80B-D1 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_negative`  
**GPU-context geopend:** nee  
**Alle replaybare checks:** PASS

## Onafhankelijk herberekend

| Arm | n | mean ms | p50 ms | p95 ms | p99 ms | min–max ms |
|---|---:|---:|---:|---:|---:|---:|
| record480 | 120 | 40.194713 | 40.172337 | 40.330388 | 40.483605 | 40.103294–40.725506 |
| layer48 | 120 | 39.002156 | 38.966064 | 39.158491 | 39.198457 | 38.937664–39.277184 |
| token1 | 120 | 38.867812 | 38.839024 | 39.041290 | 39.082947 | 38.809696–39.136223 |
| mmap→pinned staging | 32 | 70.528278 | 69.042400 | 88.474880 | 91.610749 | 48.514700–92.160100 |

- `token1/record480` p50-ratio: **0.966810171**.
- `token1/record480` p95-ratio: **0.968036550**.
- Ideale overlap-p95: **88.474880 ms**.
- Volledig seriële p95-projectie: **127.516170 ms**.

## Poorten

| Poort | Herberekend |
|---|---|
| alle opgeslagen full-bufferclaims gelijk | True |
| 120 eindige samples per H2D-arm | True |
| token1 p95 ≤45 ms | True |
| token1/record480 p50 ≤0,80 | False |
| token1/record480 p95 ≤0,90 | False |
| ideale overlap-p95 ≤45 ms | False |

Gefaalde poorten: `token1_p50_ratio_le_0_80, token1_p95_ratio_le_0_90, ideal_overlap_p95_le_45ms`. De onafhankelijke status is daarom `directpath_closed`, gelijk aan het bronresultaat.

## Provenance, volgorde en hashes

- Preregistratie-, runner- en manifest-SHA's matchen de in D1 opgeslagen waarden.
- Het geaudite bronresultaat heeft SHA-256 `8361b082d22bbb017bdffa6ccf800d059978c73ff0d4f998667aef3add137f8f`; de verifier zelf `c1a00e858d3997483ed0fc85fdd4090cdb01b7ef94c7d675a2a99e89550ad884`.
- De fysieke bank is opnieuw volledig CPU-side gehasht: `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`; dit matcht het bevroren manifest.
- De 480 layer-major/top-10-records van token 10.000 zijn onafhankelijk uit SplitMix64 gereconstrueerd. Hun geordende bron-SHA is `7e9423a1863cb698decbe8ec8dea98f834162e5bf0bbd26b9a0e36765225cc6a` en matcht D1.
- De edge-digest van exact tokens 10.001–10.032 is `727d0cbcd5b7466094ed59f0d59be140e42cb42f61faee6580542a0298d82ff3` en matcht D1.
- Alle 120 meetorders matchen het rotatie/omkeerprotocol. Elk van de zes permutaties komt 20 maal voor; elke arm staat 40 maal op iedere positie.
- De fysieke contracten zijn 48×10 records, 2.027.520 bytes per record en 973.209.600 bytes per token.

## Bewijsgrens

Het opgeslagen resultaat bevat per arm alleen `byte_equal: true` en geen devicebufferhash. Een CPU-only audit kan de tijdelijke GPU-buffers daarom niet post-hoc opnieuw vergelijken. De aggregatie van die claims is intern correct en alle bronhashes zijn onafhankelijk gereconstrueerd. Deze beperking verandert het negatieve verdict niet: de p50-/p95-ratiopoorten en de stagingpoort falen onafhankelijk van de correctness-pass.

Er is geen expertcompute, echte 80B-router, kwaliteit, dense shell, werkelijke staging/H2D-overlap of end-to-end tokens/s geverifieerd.
