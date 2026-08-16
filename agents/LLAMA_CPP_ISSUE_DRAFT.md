# Ready-to-post upstream issue: the Mamba-2 assert that blocks nemotron_h_moe

Datum: 2026-08-16 · alles hieronder is nagerekend uit `config.json`, geen schatting

Dit is een kant-en-klare tekst voor `ggml-org/llama.cpp`. Plak hem als comment
onder issue **#20570** (Nemotron 3 Nano 30B-A3B faalt op `mamba-base.cpp:173`),
of als nieuwe issue als die gesloten is. Er zit een concrete fix-suggestie in,
dus het is bruikbaar voor een maintainer in plaats van "het werkt niet bij mij".

**Waarom wij dit melden:** `nemotron_h_moe` en `GGML_TYPE_NVFP4` zitten er allebei
al in, en onze Lightning-checkpoint heeft **exact dezelfde afmetingen** als het
model in #20570. Eén assertie blokkeert de hele familie. Zie
`agents/LLAMA_CPP_INTEROP.md` voor de bredere context.

---

## Voorgestelde tekst

**Title:** `nemotron_h_moe: d_inner % (n_group*n_embd) assert rejects valid Mamba-2 configs`

**Body:**

Loading Nemotron-3-Nano-30B-A3B (and NVIDIA-Nemotron-3.5-Lightning-30B-A3B,
same dimensions) fails at `mamba-base.cpp:173`:

```
GGML_ASSERT(d_inner % (n_group*n_embd) == 0)
ssm_d_inner = 4096   ssm_n_group = 8   n_embd = 2688
```

`4096 % (8 * 2688 = 21504) = 4096`, so the assert fires.

I believe the assert is dimensionally wrong rather than the model being
unsupported. `n_embd` is the model width. It has no relationship to how
`d_inner` is partitioned across SSM groups, so multiplying by it produces a
quantity with no meaning in the Mamba-2 layout. The constraints that do hold for
this config, and that I would expect to be the intended ones:

```
d_inner % n_group             = 4096 % 8         = 0    OK
d_inner % (n_group * d_head)  = 4096 % (8 * 64)  = 0    OK
```

Independent check that the config is internally consistent and that this is a
genuine Mamba-2 layout rather than something exotic: the `in_proj` output width
reconstructs exactly from the config.

```
z            d_inner                    = 4096
x            d_inner                    = 4096
B, C         2 * n_group * d_state      = 2 * 8 * 128 = 2048
dt           n_heads                    =   64
                                          ------
                                          10304
```

and `in_proj.weight` in the checkpoint is `[10304, 2688]`. So `d_inner = 4096`,
`n_group = 8`, `d_state = 128`, `d_head = 64`, `n_heads = 64` are all consistent
with the stored tensor shapes.

Relevant config fields:

```json
{
  "hidden_size": 2688,
  "mamba_num_heads": 64,
  "mamba_head_dim": 64,
  "n_groups": 8,
  "ssm_state_size": 128,
  "num_hidden_layers": 52
}
```

Architecture loads fine up to this point — metadata is parsed and the model is
correctly identified as Mamba-based; it is the SSM validation during context
init that rejects it.

Happy to test a patch on the real checkpoint (RTX PRO 2000 Blackwell, SM120,
Windows/CUDA 13.2) if that helps.

---

## Wat wij zelf nog niet weten

Eerlijk erbij, zodat niemand dit als volledig geverifieerd leest:

1. **Wij hebben de fix niet gedraaid.** De redenering dat `n_embd` er niet in
   hoort is dimensioneel sterk, maar wij hebben niet gecontroleerd of het
   Mamba-2-pad in ggml verderop nog ándere aannames maakt die deze config wél
   breken. Het kan zijn dat de assertie een symptoom is en niet de oorzaak.
2. **Wij hebben geen GGUF van ons eigen model.** `convert_hf_to_gguf.py` kan
   NVFP4 aan, maar wij hebben de conversie niet uitgevoerd, dus of ónze
   checkpoint schoon converteert is ongetest.
3. **Snelheid is hier geen argument.** Zelfs met een werkende fix draait daar
   niet onze 51 tok/s — die komt uit expertstreaming, ERVF, de device-LRU,
   graphcapture, H-SCALE en B3, en ggml heeft die architectuur niet. llama.cpp
   is voor ons een correctheids- en UX-referentie.
