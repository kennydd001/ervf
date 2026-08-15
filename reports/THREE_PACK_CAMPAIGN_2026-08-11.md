# BITFLOW + CORETAIL + offload-roofline — geconsolideerd testrapport

## Samenvatting

| Richting | Uitkomst | Hardste bewijs |
|---|---|---|
| BITFLOW C1/Q4 | negatief, gesloten | −645,95% validation- en −395,02% test-recovery; 23/23 verifierchecks |
| CORETAIL locked16 | mechanica bevestigd, full P0 geblokkeerd | 75.497.472 codes en BF16-scales exact; 28/28 checks; slechts 16/6.144 bronnen |
| P-A Qwen wall-clock | geblokkeerd | geen full-bank GPTQ of packed offloadruntime |
| P-B LFU residency | negatief | gemiddelden pass, p99 18/31 en slechts 3/20 switches pass; 14/14 checks |
| P-C K3-roofline | hardwareleg conditioneel positief | 26,159 GB/s → 0,959 tok/s bij externe T=27,28 GB; 15/15 checks |
| P-D speculative | union gemeten, acceptatie geblokkeerd | Qwen U(8)=23,9–30,4; geen drafter/K3-acceptatiemeting; audit 15/15 |
| P-E permuted tile-64 | negatief | 5,014×/8,322× versus gate 1,20×; niet bit-exact; 18/18 checks |

## Wetenschappelijke conclusie

Er is geen bewezen Eureka. BITFLOW’s geregistreerde lineaire reparatieroute en
de voorgestelde cumulatieve-LFU- en spectrale-permutatieroutes zijn overtuigend
negatief. De K3-busbandbreedte ondersteunt de roofline alleen conditioneel,
omdat de dominante 27,28-GB-trunkinput nog extern en ongemeten is.

CORETAIL is de enige duidelijk veelbelovende mechanische vondst: de werkelijke
codec is exact en de lineaire geheugenprojectie haalt alle gates. De universele
claim mag echter pas promoveren na fysieke encoding van alle 6.144 canonieke
GPTQ-experts. De huidige coverage van 16 experts is 0,2604% en kan geen
full-bank-tailverdeling of worst-case-indexoverhead bewijzen.

## Verantwoorde volgende stap

Produceer eerst de volledige canonieke Qwen-GPTQ-bank één keer, zonder RTN-
substitutie. Die ene asset ontsluit zowel de echte CORETAIL P0 als P-A’s eerste
wall-clock. Daarna is de beslisvolgorde:

1. encodeer alle 6.144 experts en controleer werkelijke core/tail/residentbytes;
2. alleen bij een echte CORETAIL-P0-pass: bouw de fused exact decoder;
3. integreer vervolgens de full-bank 4-bit baseline en meet tok/s/CE/VRAM/RSS;
4. download of bouw geen K3/drafter vóór deze Qwen-systeembaseline bestaat.

Alle onafhankelijke verifiers slagen en de volledige regressiesuite eindigt op
153/153.
