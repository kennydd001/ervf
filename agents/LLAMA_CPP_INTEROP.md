# llama.cpp-interop: wat kan, wat niet, en de snelste weg naar een chat-UI

Datum: 2026-08-16 · alle feiten geverifieerd tegen upstream, niet uit het geheugen

**Vraag.** Kunnen we ons onderzoek in llama.cpp draaien, zodat we na een
token-barrière kunnen blijven testen zonder zelf een chat te bouwen?

**Kort antwoord.** Twee losse dingen, en ze moeten niet door elkaar lopen:
1. **Chat-UI zonder iets te porten** — opgelost, vandaag gebouwd. Onze runtime
   spreekt nu de OpenAI-API, dus élke bestaande client werkt.
2. **Ons model in llama.cpp draaien** — nu **geblokkeerd door één upstream
   assertie die dimensioneel niet klopt**. Dat is een concreet, bijdraagbaar
   bugrapport, geen maandenlange port.

---

## 1. Feiten over llama.cpp (geverifieerd)

| | status |
|---|---|
| `nemotron_h` / `nemotron_h_moe` architectuur | **aanwezig**, herkent onze modelklasse en laadt de metadata |
| `GGML_TYPE_NVFP4` | **aanwezig**; `convert_hf_to_gguf.py` detecteert ModelOpt-NVFP4 en herpakt naar GGUF-blokformaat |
| CUDA NVFP4 W4A4 | aanwezig, met per-channel amax-schaling |
| Nemotron-3-Nano-30B-A3B GGUF (onze zustermodel) | **laadt niet** — zie hieronder |

### De blokkade, precies

`ggml-org/llama.cpp` issue #20570, bij het laden van Nemotron-3-Nano-30B-A3B:

```
GGML_ASSERT(d_inner % (n_group*n_embd) == 0)   // mamba-base.cpp:173
ssm_d_inner = 4096   ssm_n_group = 8   n_embd = 2688
```

**Onze Lightning-checkpoint heeft exact dezelfde afmetingen**, nagerekend uit
`config.json`:

```
hidden_size (n_embd)                = 2688
mamba_num_heads 64 x head_dim 64    -> d_inner = 4096
n_groups = 8, ssm_state_size = 128

assert zoals hij is:  4096 % (8*2688 = 21504) = 4096   -> FAIL
d_inner % (n_group*head_dim):  4096 % (8*64 = 512) = 0 -> PASS
d_inner % n_group:             4096 % 8         = 0    -> PASS
```

**De assertie lijkt een upstream-typefout.** `n_embd` is de modeldimensie en
heeft dimensioneel niets te maken met hoe `d_inner` over SSM-groepen verdeeld
wordt; de zinnige eis is `d_inner % n_group == 0` (of `n_group * head_dim`).
Beide slagen bij ons.

**Onafhankelijke bevestiging dat wij de Mamba-layout goed lezen** — de
`in_proj`-uitvoerdimensie reconstrueert exact:

```
z 4096 + x 4096 + BC (2*8*128 = 2048) + dt 64 = 10304
```

en 10304 x 2688 is precies de shape die we all sessie gemeten hebben. Onze
lezing van de layout is dus niet de bron van het conflict.

**Actie:** dit is het waard om upstream te melden met bovenstaande rekensom.
Als de assertie gecorrigeerd wordt, kan onze checkpoint via
`convert_hf_to_gguf.py` (die NVFP4 al aankan) naar GGUF.

### Wat llama.cpp ons *niet* geeft

Ook ná die fix draait daar **niet** onze 51 tok/s. Onze snelheid komt uit een
eigen CUDA/CuPy-runtime: NVFP4-expertstreaming uit pinned host, ERVF-GEMV's,
device-residente LRU-expertcache, CUDA-graphcapture, H-SCALE en B3-overlap.
ggml heeft die architectuur niet. llama.cpp is voor ons dus een
**correctheids- en UX-referentie**, en een externe tok/s-vergelijking — geen
snelheidsdoel en geen plek om ons onderzoek naartoe te porten.

---

## 2. De chat-UI: opgelost zonder port

`scripts/lightningstream_nemotron/serve_openai.py` (nieuw) zet een
OpenAI-compatibel endpoint voor onze runtime:

```
.venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/serve_openai.py
```

- `GET /v1/models`, `GET /health`
- `POST /v1/chat/completions`, streaming (SSE) en niet-streaming
- alleen stdlib, geen nieuwe dependency
- `--stack v18` (default, de recordweg) of `--stack v6` ter vergelijking

Daarmee werkt élke bestaande client: llama.cpp's eigen web-UI, Open WebUI,
LM Studio, Continue, of gewoon `curl`. Getest en werkend, streaming en
niet-streaming, met `usage` en een `x_tokens_per_second`-veld erbij.

**Kant en klaar, één commando.** De server serveert op `/` ook een ingebouwde
web-UI, dus er hoeft niets geïnstalleerd te worden:

```powershell
.\CHAT.ps1
```

Dat start de V18-stack, opent de browser en je chat. `CHAT.ps1` ruimt eerst
achtergebleven serverprocessen op (anders faalt de volgende start met
`cudaErrorAlreadyMapped`, zie sectie 4) en waarschuwt als de GPU al bezet is.
Opties: `-Port`, `-Stack v6`, `-Capacity`, `-NoBrowser`.

Geverifieerd op een schone GPU: `GET /` geeft de UI (200, 4185 bytes),
`/health` meldt `stack: v18`, en een streaming-request levert 63 SSE-chunks op
**49,04 tok/s**.

Wie liever een andere UI gebruikt, richt die gewoon op
`http://127.0.0.1:8080/v1` — llama.cpp's web-UI, Open WebUI, LM Studio.

Er was al een `chat_lightning.py` (CLI). Die stond **untracked** en bouwde
alleen de kale runtime; beide zijn nu vastgelegd.

**Grenzen van de server, expliciet:** één GPU, één runtime, één sequentie.
Requests staan achter een lock en elke request reset de modelstate. Geen
KV-slots, geen parallelle decode.

---

## 3. De app haalt nu 48-49 tok/s — opgelost, met de oorzaak

Eerst gemeten: de server deed 24,7-24,9 tok/s waar de kale runtime 38,7 haalde.
Slechter dan kaal is een fout, geen ruis. `scripts/.../bench_stacks.py` strookt
de HTTP-laag weg en timet de drie stacks door dezelfde decodelus, één stack per
proces:

| stack | sync tok/s | queued tok/s | graph extra VRAM |
|---|---:|---:|---:|
| bare (runtime + graph) | 35,99 | 35,98 | 6 MiB |
| v6 (+ selectieve ERVF + gebatchte MoE) | 46,35 | 46,64 | 0 MiB |
| **v18 (+ H-SCALE + B3)** | **50,10** | **50,29** | **524 MiB** |

**De V18-stack was dus nooit stuk — mijn serverbouw wel.** Het verschil was één
ontbrekende regel: `cp.get_default_memory_pool().free_all_blocks()` vóór het
installeren van H-SCALE.

Die regel is **dragend, geen opruiming**. H-SCALE heeft 492,4 MiB aaneengesloten
nodig voor de schaalvlakken tegen ~607 MiB vrij. CuPy's pool houdt élk blok vast
dat hij ooit heeft aangevraagd, dus zonder teruggave heeft de driver veel minder
beschikbaar en zakt de allocatie weg in pool-thrash in plaats van netjes te
falen. Het symptoom was zichtbaar en werd eerst over het hoofd gezien:
`setup_graph` meldde **0 MiB** extra graph-VRAM in de server tegen **524 MiB** in
de bench.

Na de fix, drie warme runs van 200 tokens over de HTTP-API:

| | tok/s |
|---|---:|
| V18-stack kaal (bench_stacks) | 50,10 |
| **via de OpenAI-server, echte chat** | **48,08 / 49,12 / 48,95** |

Het resterende verschil van ~1-2 tok/s is echte app-overhead: per-token
`tok.decode` plus SSE/HTTP. Dat is eerlijk en klein.

De server logt nu de VRAM-rekening en valt expliciet terug op v6 als de
schaalvlakken niet passen, in plaats van stilletjes traag te worden.

**Methodische les, dezelfde als eerder deze sessie:** vergelijk een kandidaat
altijd met wat er werkelijk draait, en let op de secundaire meter — die 0 MiB
tegen 524 MiB stond er van meet af aan en wees direct naar de oorzaak.

## 4. Bijvangst: `cudaErrorAlreadyMapped` verklaard

Kimi's K2-harness liep vast op `cudaErrorAlreadyMapped: resource already mapped`
en dat werd toegeschreven aan een fout in de graph/capture-harness. Vandaag
reproduceerde dat hier met een duidelijke oorzaak: **een achtergebleven
Python-proces hield de pinned/mapped hostbuffers nog vast.** `nvidia-smi` toonde
7775 MiB in gebruik terwijl er geen run liep; `pkill -f` had het proces onder
Git Bash niet gedood.

Betrouwbaar opruimen op Windows:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
Stop-Process -Id <pid> -Force
```

Daarna `nvidia-smi` controleren tot het weer bij ~0 MiB staat. Dit hoort in elke
runner vóór een timing-arm — de V18-runner van Kimi doet dat inmiddels al
(weigert te starten bij >1 GB in gebruik), en dat blijkt terecht.
