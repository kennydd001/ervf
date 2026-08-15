# P10D — GPU-router/top-k negatief gesloten

## Uitkomst

Op 480 echte Qwen-routerlogitvectoren gaf de CUDA-kandidaat exact dezelfde
top-8-volgorde en bitexact dezelfde BF16-gewichten als de bestaande CPU-route.
De timing faalde echter overtuigend:

| routebarrière | p50 | p95 |
|---|---:|---:|
| logits D2H + NumPy-route | 0,0296 ms | 0,0816 ms |
| GPU top-k + 64 B D2H | 0,1300 ms | 0,1684 ms |
| verhouding | 4,392× | 2,064× |

De geïsoleerde GPU-kernel had zelf p50 0,05165 ms. Zolang de host de expertcache
en kopieplanning uitvoert, blijft een synchronisatie nodig en wint de kleinere
retourkopie niet van kernel- en eventoverhead. Deze vaste GPU-routervariant is
daarom negatief gesloten.
