# P9B — 50% statische expertpruning

## Uitkomst

Per expert en laag bleven exact 384 van 768 SwiGLU-kanalen over, geselecteerd
met de vooraf vastgelegde activatie×down-normscore. Daarna zijn experts Q5 en
trunk/head INT8 gekwantiseerd.

| split | relatieve CE-toename | top-1-overeenkomst |
|---|---:|---:|
| validation | +1,478% | 92,362% |
| test | −0,478% | 92,598% |

De full-depth kwaliteitspoorten passeren. De theoretische expertgewichtfractie
is 50%.

## Belangrijke fysieke nuance

P9C testte vervolgens de naïeve implementatie: de 384 down-kolommen dicht op
elkaar zetten en opnieuw in Q5-groepen van 128 verdelen. Die variant faalde hard
op validation (+48,03% relatieve CE, 60,47% top-1). P9B rechtvaardigt dus niet
zomaar een gewone dichte `[2048,384]`-Q5-matrix. Een fysieke variant moet de
oorspronkelijke down-groepschalen behouden of de compacte gewichten opnieuw
kalibreren en opnieuw full-depth valideren.
