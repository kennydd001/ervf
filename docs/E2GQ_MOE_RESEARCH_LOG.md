# E2GQ-MoE onderzoekslog

## 2026-08-11 — onafhankelijke representatieprecondition bevestigd

De codehistogrammen zijn rechtstreeks opnieuw afgeleid uit de 16 originele
FLEQ GPTQ-safetensors, niet overgenomen uit het aangeleverde audit-JSON. Exact
dezelfde telling van 75.497.472 codes werd gevonden: `{-2: 4.713.974, -1:
17.846.753, 0: 31.599.966, +1: 21.336.779}`. De code-entropie is
1,782864891374 bpp en met raw BF16 group-128 scales 1,907864891374 bpp. Alle
code/scale-reconstructies zijn bit-exact; alle 16 experts en 48 matrices zitten
ideaal onder 2 bpp.

Dit corrigeert terecht de eerdere interpretatie dat 2,125 bpp een universele
ondergrens was. Het getal blijft correct voor fixed-width uint2 plus raw
scales, maar niet voor een lossless entropyrepresentatie.

## 2026-08-11 — nieuwe onafhankelijke P0 geregistreerd

FLEQ/GSQ blijft gesloten. E2GQ test een andere mechaniek. Het aangeleverde
agentpack liet de full-bank GPTQ-calibratieregel impliciet; zonder die regel
zijn codes van nooit of zelden gerouteerde experts niet eenduidig. P0 bevriest
daarom 32.768 WikiText-train-tokens, uitsluitend echte routed activaties en
minimaal 128 rijen per expert. Onvoldoende dekking is een negatieve uitkomst,
geen aanleiding voor een stille fallback of nieuwe corpuskeuze.

## 2026-08-11 — P0 sluit als coverage-negative

De 32.768 locked trainings­tokens zijn modelbreed doorgestuurd. Alle 48
capture-artifacts en routercounts sluiten bit-exact met de officiële
routeruitvoer. De run bleef ruim binnen de resourcegrenzen: 2,50 GB peak CUDA
allocation, 3,65 GB proces-RSS en 139,85 seconden wandtijd.

De calibratiedekking faalt hard. Alle 48 lagen bevatten ondergedekte experts;
1.695 van 6.144 laag-expertparen hebben minder dan 128 invocaties en 196 hebben
exact nul invocaties. De maximale expertcount is 23.480, wat tegelijk een zeer
scheve routingverdeling aantoont. De onafhankelijke verifier sluit 12/12.

Conform de preregistratie zijn geen GPTQ-codes voor de ondergedekte experts
verzonnen en is P1 niet geopend. De 16-expert representatieprecondition blijft
bevestigd, maar de full-bank entropyhypothese is nog ongetest. Een geldig
vervolg vereist een nieuwe vooraf vastgelegde calibratiemechaniek; deze
registry mag niet post-hoc overschakelen naar ongerouteerde activaties of een
nieuw corpus.

## 2026-08-11 — post-P0 coder-sanity, uitsluitend diagnostisch

Na het negatieve coveragebesluit zijn de 16 bestaande artifacts in-memory met
drie generieke lossless codecs verkend. De codes zijn eerst echt als uint2
gepakt; raw BF16-scales en 64 headerbytes per matrix zijn meegerekend. Zlib-9
projecteerde 1,930709 bpp aggregate en maximaal 1,944009 bpp per expert; LZMA-9
1,938004 bpp; BZ2 miste de grens met 2,022630 bpp.

Dit is relevante fysieke plausibiliteit bovenop Shannon-accounting, maar geen
formeel P1-resultaat: de codecvergelijking was niet vooraf geregistreerd, er is
geen bestand geschreven, decode-identiteit is niet getest en slechts
matrix-level random access is verondersteld. De meting blijft daarom als
exploratory poging 001 bewaard en wijzigt het coverage-negative verdict niet.
