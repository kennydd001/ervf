# P7 ERVF — definitief lokaal resultaat

Datum: 2026-08-12  
Status: **48/48 onafhankelijke verificatiepoorten geslaagd**

## Resultaat

P7 levert een bewezen, bitexacte versnelling van de volledige custom
Qwen3-30B-A3B decode op de NVIDIA RTX PRO 2000 Blackwell Laptop GPU met 8 GB
VRAM. De nieuwe techniek heet voorlopig **Exact-Reduction Virtual Fusion
(ERVF)**.

| Meting | P6B | P7C ERVF | Verandering |
|---|---:|---:|---:|
| Afgesloten test, mean | 49,927 ms | 33,208 ms | −33,49% |
| Afgesloten test, p95 | 58,187 ms | 43,488 ms | −25,26% |
| Afgesloten test, tok/s | 20,029 | 30,113 | +50,35% |
| 512-tokenrollout, mean | 63,024 ms | 47,813 ms | −24,13% |
| 512-tokenrollout, p95 | 74,936 ms | 59,682 ms | −20,36% |
| 512-tokenrollout, tok/s | 15,867 | 20,915 | +31,81% |

Geïsoleerd daalde de Q8-projectievlaktijd van 15,213 naar 8,819 ms (1,725×)
en de Q5-expertprojectievlaktijd van 18,167 naar 7,614 ms (2,386×).

## Wat ERVF doet

P6B gaf iedere outputrij een volledig block van 256 fysieke threads. ERVF laat
een 16-lane subwarp één rij uitvoeren en plaatst zestien rijen in hetzelfde
256-thread block. Om numerieke drift te vermijden houdt iedere lane de
accumulatoren van meerdere oorspronkelijke virtuele threads apart. De
lane-lokale folds en daarna subwarp-shuffles reconstrueren exact dezelfde
binaire reductieboom als P6B.

Dit verwijdert blockbrede shared-memoryreducties en synchronisatie, verhoogt het
aantal gelijktijdige rijen en verandert geen gewicht, schaal, MAC-volgorde per
virtuele thread of BF16-outputbit.

## Correctheidsbewijs

- P7B: alle breedtes 8, 16 en 32 waren bitgelijk; de gekozen breedte 16 had
  0 verschillende bits over 502.144 Q8- en 1.376.256 Q5-uitvoerelementen.
- P7D: baseline en ERVF hadden 0 verschillende individuele CE-waarden over
  1.270 validation- plus 1.270 testlabels.
- Validation en test: voorspellingen, expertmissers en alle KV-digests exact
  gelijk.
- Rollout: dezelfde prompt-, feedback- en 512 gegenereerde token-ID's en
  dezelfde eind-KV-digest.
- Test-CE bleef 2,260959 tegenover BF16-teacher 2,259874: relatieve toename
  +0,0480%, exact gelijk aan P6B.
- Fysieke residentie bleef gelijk: 4,978 GB expertcache, 1,249 GB device-trunk,
  402,65 MB KV en circa 749,7 MB vrije scratch na allocatie.
- Regressietests: 156/156 geslaagd.

## Betekenis

Dit is een echte **lokale Eureka**: niet een simulatie, maar een volledige
48-laagse fysieke decoder met echte routing, cachemissers, H2D-kopieën,
attention, KV-mutatie, LM-head en autoregressieve feedback. De winst komt uit
een nieuwe exact-reduction launcharchitectuur en niet uit minder modelwerk of
een kwaliteitsoffer.

Het is nog geen verdedigbare “beste ter wereld”- of algemene LLM-doorbraak.
De test betreft één model, één GPU, één custom runtime en korte contexten. Een
wereldclaim vereist minimaal dezelfde-hardware externe runtimebaselines,
langdurige thermische runs, langere contexten, een tweede MoE-model en een
prior-artonderzoek naar exact-tree virtual-thread fusion.

## Volgende gesloten experimenten

1. P8A projection-adaptive breedtes, vooraf gekozen op validation en éénmalig
   getest.
2. P8B bitexacte scale-broadcast binnen ERVF.
3. P8C dezelfde-hardware runtimevergelijking met publiek reproduceerbare
   instellingen.
4. P8D 60-minuten thermische run plus lange-context KV-sweep.
5. P8E tweede moderne MoE-checkpointreplicatie.
