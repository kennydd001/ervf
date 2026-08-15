# N0 onafhankelijk dekkingsaudit — `intel1`, `intel2` en zes hypotheses

Datum: 2026-08-12  
Status: conceptaudit; de centrale `NEXT_WAVE_REGISTRY_2026-08-12.yaml` is niet gewijzigd.

## Auditregel

Een idee is alleen **tested** wanneer naast een preregistratie ook een fysiek of
empirisch resultaatbestand bestaat. **Superseded** betekent dat een sterkere
latere proef de premisse verving; het betekent niet automatisch dat iedere
variant van het oude idee is uitgevoerd. **Open** betekent lokaal toetsbaar maar
niet gesloten. **Blocked** betekent dat een artifact, tweede GPU of omvangrijke
nieuwe implementatie ontbreekt.

De actuele `NEXT_WAVE_REGISTRY` loopt achter op de bestanden: N001, N005 en N008
staan daar nog als `in_progress`/`queued`, terwijl er al resultaten voor
shared-activation, de eerste temporal-oracle en de LM-head-oracle bestaan.

## Idee-voor-idee

| Bronidee | Auditstatus | Wat daadwerkelijk bewezen of gefalsificeerd is | Precieze evidence |
|---|---|---|---|
| ERVF-basis op Q5/Q8 | **tested — pass** | De originele width-16-transformatie is bitexact en versnelt de volledige Qwen-runtime. | `reports/streamq5_moe/P7_ERVF_FINAL_REPORT_2026-08-12.md`; `reports/streamq5_moe/P7_ERVF_INDEPENDENT_VERIFICATION.md` |
| Projection-adaptive width 8/16/32 | **tested — negative als brede/full-runtimeclaim** | Q5 width-8 won geïsoleerd, Q8 faalde de gezamenlijke poort en Q5-only gaf end-to-end geen winst. | `reports/streamq5_moe/P8AB_KERNEL_FOLLOWUPS_REPORT_2026-08-12.md`; `reports/streamq5_moe/P8A2_Q5_WIDTH8_END_TO_END_REPORT_2026-08-12.md` |
| Algemene EVRM/ERGV-autotuner, widths 4/8/16/32/64 | **open** | Alleen een handmatige 8/16/32-search voor de bestaande Q5/Q8-projecties is gedaan. Geen width 4/64, geen automatische zoeker en geen voorspellend occupancy/registermodel. | Open item N003 in `reports/streamq5_moe/NEXT_WAVE_REGISTRY_2026-08-12.yaml`; gedeeltelijke basis in `reports/streamq5_moe/p8a_projection_adaptive_ervf.json` |
| EVRM op INT4, BF16, RMSNorm, softmax en andere reductions | **open** | P13 bewijst een specifieke exacte attentionvirtualisatie, niet de gevraagde generieke reduction-graphtransformatie over al deze kernelfamilies. | `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md`; N003 in de next-wave-registry |
| Shared-activation ERVF, gewone shared-memory staging | **tested — negative als brede/eindclaim** | Q8 was trager (`1.2244×` p50). Q5 won geïsoleerd (`0.9016×` p50), maar de kandidaat-eerst-replicatie keerde de full-runtimewinst om (`1.1133×` mean, `1.1705×` p95). De 10K-test bleef dicht. | `reports/streamq5_moe/n1a_shared_activation_ervf.json`; `reports/streamq5_moe/n1a2_q5_staging_end_to_end.json`; `reports/streamq5_moe/n1a2r_q5_staging_reverse.json` |
| Vectorized async/double-buffered activation staging | **open** | N1A testte coöperatieve shared staging, niet afzonderlijk de in `intel2` genoemde async- en double-buffered varianten. | `reports/streamq5_moe/N1A_SHARED_ACTIVATION_ERVF_PREREGISTRATION.md` |
| Q5 vectorized/widened pack loads (`uint4`) | **open** | Registry-item bestaat, maar er is geen preregistratie of resultaat. | N002 in `reports/streamq5_moe/NEXT_WAVE_REGISTRY_2026-08-12.yaml` |
| Meerdere outputrijen per lane / extra ILP | **open** | ERVF verwerkt meerdere rijen per block; de specifiek voorgestelde meerdere rijen per lane zijn niet als aparte variant gemeten. | Geen resultaatbestand; alleen bronsectie 3 van `info/pro/NA_ERVF_ZES_HYPOTHESES_2026-08-12.md` |
| `cp.async`/TMA weightstreaming | **open** | Geen afzonderlijke kandidaat, preregistratie of fysieke meting gevonden. | Geen evidence-item in `reports/streamq5_moe/NEXT_WAVE_REGISTRY_2026-08-12.yaml` |
| Q5 scale-broadcast | **tested — negative** | Bitexact, maar trager (`1.0375×` p50). | `reports/streamq5_moe/P8AB_KERNEL_FOLLOWUPS_REPORT_2026-08-12.md`; `reports/streamq5_moe/p8b_scale_broadcast_ervf.json` |
| ERVA / exact virtual GQA-attention | **superseded door sterkere tested-pass** | EVT-PM is bitexact over 48 lagen bij context 128/512/1024/4096; bij 4K daalde p50 naar `0.1338×`; de volledige 10K-run bij 4K haalde 14.235 tok/s onder 32 GiB. | `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md`; `reports/streamq5_moe/p13b_explicit_add_attention.json`; `reports/streamq5_moe/p13c_evt_pm_32g_endurance.json` |
| Attention/contextcurves bij 8K/16K/32K | **open** | De sterkste fysieke attentiontest stopt bij 4096. Geen 8K/16K/32K curve. | Claimgrens in `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md`; N010 in de next-wave-registry |
| Temporal ERVF, zelfde expertset, S=2/4/8 | **tested — component pass** | Alle Q8/Q5-uitgangen bitexact. S=4 test gaf gecombineerd `0.7179×` p50 en `0.6599×` p95 tegenover vier losse calls. Dit is alleen een optimistische same-expert target-oracle. | `reports/streamq5_moe/n2a_temporal_ervf_oracle.json`; `reports/streamq5_moe/N2A_TEMPORAL_ERVF_ORACLE_PREREGISTRATION.md` |
| Echte S=4 route-unie | **tested — statistische pass** | Over vijf domeinen, 48 lagen en 1024 tokens was de mean-unie 18.340 en p95 25; de uniepoort slaagde. De pessimistische byte-lineaire projectie was echter `1.121×` de sequential baseline. | `reports/streamq5_moe/n2au_route_union.json`; `reports/streamq5_moe/N2AU_ROUTE_UNION_PREREGISTRATION.md` |
| Sparse temporal Q5 op echte routepatronen | **tested — negative** | Bitexact over 27.525.120 waarden, maar trager dan vier losse calls: validation-p50 `1.0457×` en p95 `1.0706×`. De testpartition bleef conform preregistratie dicht. Hiermee is de echte-routekernel negatief gesloten; alleen de optimistische same-expert-oracle blijft positief. | `reports/streamq5_moe/n2as_sparse_temporal_q5.json`; `reports/streamq5_moe/N2AS_SPARSE_TEMPORAL_Q5_PREREGISTRATION.md` |
| Drafter/MTP acceptance en end-to-end speculative decoding | **blocked artifact** | Zonder passend drafter/MTP-head kan acceptance niet worden gemeten. N2A en N2AU bewijzen dit niet. | `reports/streamq5_moe/P13D_SPECULATIVE_DECODING_CLOSURE_2026-08-12.md`; N006 in de next-wave-registry |
| Projection-flow/layer-fused glue | **open, met gedeeltelijk negatief/superseded voorwerk** | De oude glue-only graphpremisse is door P13 gesupersedeerd. P3A groepeerde gate/up al; een P4C gate+up+SwiGLU-overlapvariant faalde validation. De nieuwe exacte norm→QKV→RoPE→KV-, O→residual→norm→router- en MoE-outputketens zijn niet gebouwd. | `reports/streamq5_moe/P8C_GLUE_GRAPH_CLOSURE_2026-08-12.md`; `reports/streamq5_moe/DATAPLANE_ONTLEDING_VERDICT_2026-08-12.md`; N007 in de next-wave-registry |
| Layer-fused megakernel (H-A) | **open** | Geen fysieke megakernel of gemeten `G <= 6 ms`. Een CUDA graph voor de oude expertketen en P13-attention zijn geen uitvoering van deze kandidaat. | N007 in `reports/streamq5_moe/NEXT_WAVE_REGISTRY_2026-08-12.yaml` |
| Exacte LM-head cluster/bound search | **tested — negative voor de vaste signcluster-variant** | 1270/1270 tokens waren exact gecertificeerd, maar nul vocabulariumrijen konden worden overgeslagen; de 60%-poort faalde en test bleef dicht. | `reports/streamq5_moe/n2b_certified_lm_head_oracle_validation.json`; `reports/streamq5_moe/N2B_CERTIFIED_LM_HEAD_ORACLE_PREREGISTRATION.md` |
| Exacte LM-head top-k fusion zonder full-logit write | **open** | Dit is een andere kandidaat dan cluster-pruning en is nog niet fysiek gemeten. | Gecombineerd maar nog niet afgedekt door N008 in de next-wave-registry |
| VRAM cache/KV-allocatie-Pareto (H-B) | **open** | P10 testte cachepolicies en P13 bewees één 4K/32-GiB-punt; er is geen fysieke curve over cache-slots, maximale context en tok/s. | `reports/streamq5_moe/p10_cache_family.json`; `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md`; N009 in de next-wave-registry |
| INT8/INT4-KV bij 8K en 32K (H-C) | **open** | Exact EVT-PM maakte lossy KV voor de gemeten 4K-poort overbodig, maar falsificeert of bewijst de expliciete 8K/32K kwaliteits- en capaciteitshypothese niet. | N010 in de next-wave-registry; grens `context > 4096` in `reports/streamq5_moe/FINAL_VERDICT.md` |
| TTFT en custom-runtime prefill-GEMM (H-D) | **open** | De CPU-only llama.cpp-baseline rapporteert prompt-tok/s, maar geen TTFT/prefillmeting of GEMM-kernel van STREAMQ5. | `reports/streamq5_moe/P15A_LLAMA_CPP_CPU_BASELINE_REPORT_2026-08-12.md`; N011 in de next-wave-registry |
| Exactheidsverifier + mechanische autotuner (H-E) | **open, bouwstenen bewezen** | Bitexacte checks en onafhankelijke verifiers bestaan; een automatische geometriezoeker die minimaal P7 en één snellere kandidaat vindt bestaat niet. | `reports/streamq5_moe/p7_ervf_independent_verification.json`; `reports/streamq5_moe/p8_p13_independent_verification.json`; N003 in de next-wave-registry |
| Batch 2/4/8/16 route-unie en aggregate throughput (H-F) | **open** | N2AU meet temporele blokunies, niet fysieke multi-requestbatchdoorvoer; alle gerapporteerde systeemclaims blijven batch 1. | N012 in de next-wave-registry; claimgrens in `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md` |
| Synthetic Qwen3-Coder-Next-80B shape gate | **open** | Nog geen fysieke synthetic shape-timing voor top-10/intermediate-512/shared expert. | N013 in de next-wave-registry |
| Volledige Qwen3-Coder-Next-80B-replicatie | **blocked artifact** | Checkpoint en fysieke bank ontbreken; synthetic gate moet eerst slagen. | N016 in de next-wave-registry; I036 in `reports/streamq5_moe/ALL_IDEAS_CLOSURE_REGISTRY_2026-08-12.yaml` |
| Tweede MoE-familie | **tested — kwaliteitspass; fysieke runtime open** | DeepSeek-V2-Lite passeert full-depth Q5-kwaliteit over 26 MoE-lagen, maar er is geen fysieke bank/cache/kernel/decode. | `reports/streamq5_moe/P14A_DEEPSEEK_V2_LITE_Q5_REPORT_2026-08-12.md` |
| Tweede GPU / cross-architectuur | **blocked hardware** | Geen tweede NVIDIA-architectuur beschikbaar. | N015 in de next-wave-registry |
| Publieke baseline | **tested — CPU-anker; sterke GPU-equivalenten open/blocked scope** | llama.cpp CPU-only is 63.22× trager, maar dat is expliciet geen vergelijking met GemLite/CUTLASS/QUICK of de beste hybride runtime onder gelijke semantiek. | `reports/streamq5_moe/P15A_LLAMA_CPP_CPU_BASELINE_REPORT_2026-08-12.md`; N014 in de next-wave-registry |
| Harde 32-GiB- en 10K-duurproef | **tested — pass** | 10.000 tokens, 4K context, 14.235 tok/s, identieke prediction/miss/KV-digests. Dit is circa 702.5 s, niet een afzonderlijke 60-minutenrun. | `reports/streamq5_moe/P13_EVT_PM_EUREKA_REPORT_2026-08-12.md`; `reports/streamq5_moe/p13c_evt_pm_32g_endurance.json` |
| CPU miss-compute als `beta*m -> 0` | **tested — negative** | De expliciete CPU-Q5-misscomputeproef sloot negatief. | `reports/streamq5_moe/p11a_cpu_q5_miss_compute.json`; `reports/streamq5_moe/P11A_CPU_Q5_MISS_COMPUTE_PREREGISTRATION.md` |

## Dekkingsgaten in de huidige next-wave-registry

De 19 registry-items vatten de hoofdrichtingen goed samen, maar “alles uit de
drie documenten” is nog niet letterlijk afgedekt. Vier varianten verdienen een
eigen item of expliciete substatus, zodat een negatief resultaat op een verwante
variant ze niet per ongeluk sluit:

1. vectorized async en double-buffered **activation** staging (niet hetzelfde
   als N001 gewone shared staging of N002 Q5-weightloads);
2. meerdere outputrijen **per lane** en `cp.async`/TMA-weightpipelining;
3. LM-head **top-k write-elision/fusion** (niet gesloten door de negatieve
   Euclidische signcluster-oracle);
4. volledige fysieke DeepSeek-V2-Lite-bank/cache/runtimereplicatie; tot nu toe is
   alleen de kwaliteit gerepliceerd.

Daarnaast moet N005 intern vier grenzen bewaren: same-expert target-oracle is
positief, route-unie-statistiek is positief, de echte sparse-routekernel is
negatief en drafteracceptatie is artifact-geblokkeerd. Eén status op N005 zou
die nuances verliezen.

## Conclusie

Niet alles is getest. Het sterkste nieuwe positieve bewijs is Temporal ERVF op
de geïsoleerde same-expert targetkernel; de beslissende echte-routevariant is
bitexact maar fysiek trager en dus negatief gesloten. De sterkste nieuwe
falsificaties zijn shared-activation als brede/full-runtimewinst, sparse
Temporal ERVF op echte routepatronen en de vaste gecertificeerde
LM-head-signclusterzoeker.
De grootste volledig open blokken zijn generalized autotuning/load-pipelining,
layer/projection-flow fusion, lange-context KV/TTFT, physical batch en de
synthetic 80B-shapegate. Externe generalisatie blijft door artifact, scope of
hardware geblokkeerd.
