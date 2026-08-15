# HET-NEXT L0 PH0X — exploratory real-projection preregistration

## Scope

Eén vooraf vastgelegde officiële Qwen3-Coder-Next-projectie wordt sequentieel op
CPU, Intel host-USM en NVIDIA CUDA uitgevoerd:

- `model.layers.0.mlp.experts.50.gate_proj.weight`, BF16 `[512,2048]`;
- officiële D2-R3 `p0_whole_post_norm[15]`, BF16 `[2048]`;
- STREAMQ5 q+15-codec, group 128, exact wire 675.840 bytes;
- exact width-8-reductiecontract uit PH0-R3;
- 512 BF16-outputwoorden en uint32-rowcounters worden volledig bewaard.

## Bevroren uitvoering

1. Bron/input worden uitsluitend uit hun vooraf vastgelegde ranges gelezen en
   gehasht.
2. De bestaande PH0-R3 packer en bit-level software-FP32-orakel worden gebruikt.
3. De Intel PH0-kernel krijgt syntactisch geldige pragma's en gebruikt vier
   host-USM-buffers; geen expliciete OpenCL-copycalls.
4. De NVIDIA-kernel gebruikt `cooperative_groups::tiled_partition<8>`, pinned
   hostbuffers, twee H2D-copies, één kernel, twee D2H-copies.
5. Iedere backend wordt exact één keer geprobeerd; geen tuning, timinggate of retry.
6. Positief vereist: CPU-controls groen, Intel/NVIDIA 512/512 bitexact tegen CPU,
   counters overal één, outputcanary volledig overschreven en cleanup zonder fout.

## Claimgrens

Dit is exploratieve componentevidence. Een positieve uitkomst bewijst uitsluitend
dat deze ene echte Q5-projectie/input onder de drie bevroren reductiepaden dezelfde
BF16-output geeft. Geen volledige expert, MoE-laag, modelkwaliteit, hybrid speedup,
tokens/s, deployment, novelty of breakthrough.

