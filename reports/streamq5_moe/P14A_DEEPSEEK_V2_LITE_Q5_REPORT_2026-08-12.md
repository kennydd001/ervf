# P14A — DeepSeek-V2-Lite Q5-replicatie

## Uitkomst

De vooraf vastgelegde full-depth kwaliteitstest passeert op beide splits van
de lokaal aanwezige DeepSeek-V2-Lite-checkpoint:

| split | relatieve CE-toename | top-1 token | mediane route-overlap |
|---|---:|---:|---:|
| validation | +0,716% | 95,703% | 96,875% |
| test | +1,493% | 94,922% | 96,745% |

Alle 26 MoE-lagen zijn uitgevoerd. Dit is betekenisvolle ondersteuning buiten
Qwen: top-6-routing, shared experts en een andere MoE-familie behouden de
vastgelegde Q5-kwaliteitsgrens.

## Niet bewezen

Er is voor DeepSeek nog geen fysieke Q5-bank, cachepolicy, kernel of volledige
decode-timing gebouwd. De generalisatie is daarom een kwaliteitsreplicatie,
geen tweede end-to-end systemsreplicatie.
