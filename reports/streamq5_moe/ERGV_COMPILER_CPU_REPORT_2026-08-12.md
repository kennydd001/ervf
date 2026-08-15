# ERGV-C0 — restricted exact-reduction compiler

Datum: 2026-08-12  
Verdict: **CPU-formalisatie PASS (7/7); GPU-gate nog niet uitgevoerd**

## Resultaat

Het P6B/P7/N1C-reductiepatroon is omgezet in een restricted
`ExactReductionIR` en een mechanische verifier. Het prototype kan de logische
256-accumulatorboom onafhankelijk van de fysieke CUDA-schedule representeren,
de schedule opnieuw tot een geordende operatorboom materialiseren en beide
bomen mechanisch vergelijken.

De CPU-gate is volledig geslaagd:

| Poort | Resultaat |
|---|---:|
| Testgroepen | 7/7 |
| Gecontroleerde shapes | Q8 2048/4096; Q5 2048/768 |
| Fysieke breedtes per shape | 4/8/16/32/64 |
| Isomorfe shape×width-schedules | 20/20 |
| Nodes per geordende graph | 512 |
| Random/adversariële inputvectoren | 134 |
| Bitvergelijkingen | 2.680/2.680 |
| Negatieve semantische mutaties | 6/6 afgewezen |
| Gegenereerde CUDA-source | 14.793 bytes, deterministisch |

## Wat nu mechanisch wordt bewaakt

De IR bevat:

- 256 stabiele logische accumulator-ID's;
- de 255 geordende add-nodes van strides
  `128,64,32,16,8,4,2,1`;
- de uiteindelijke BF16-round;
- de FMA-policy;
- iedere Q8-kolom of Q5-pack in de oorspronkelijke laadvolgorde;
- voor Q5 ook de seriële volgorde van de acht codes binnen ieder 40-bit pack;
- de mapping `logical_id -> (physical_lane, virtual_slot)`;
- de fase van iedere add: lane-local, warp-shuffle of cross-warp shared.

Voor breedte 64 eist de verifier exact 32 cross-warp-adds op stride 32. Voor de
kleinere breedtes zijn cross-warp-adds verboden. De geordende graphvergelijking
behandelt verwisselde kinderen niet als equivalent.

## Reproductie van bestaand handwerk

De bronaudit vond en valideerde:

- P6B: de oorspronkelijke 256-thread shared-memoryboom;
- P7: breedtes `8,16,32`, met lane-lokale folds en begrensde shuffles;
- N1C: breedtes `4,8,16,32,64`, inclusief de expliciete width-64-stap.

De bevroren N1C-keuzes zijn zonder uitzonderingscode representeerbaar:

```text
Q8: head=16, k=64, o=16, q=16, router=64, v=64
Q5: gate_up=8, down=8
```

De codegenerator levert Q8- en Q5-row-reducers voor alle P7-breedtes en een
width-64 pre-reducer/completion-contract. Shape-onafhankelijke helpers worden
op family×width gededupliceerd. De gegenereerde bron heeft SHA-256
`b313eee9070a672a9dcfde2bf73c54ab0d9deb13dfc0a4ef2d0722b40006d532`.

## Negatieve controles

De verifier wees alle vooraf vereiste corrupte kandidaten af:

1. verwisselde kinderen in de geordende root-add;
2. omgekeerde bronlaadvolgorde;
3. gewijzigde eindcast;
4. gewijzigde FMA-policy;
5. verwisselde virtuele lane-mapping;
6. verwijderde width-64 cross-warp-node.

Daarmee is de pass niet alleen afkomstig van het feit dat generator en
verifier dezelfde nominale configuratie delen; relevante semantische fouten
worden daadwerkelijk gedetecteerd.

## Bewijsgrens en vervolgstap

Dit is een echt formeel prototype-resultaat, maar nog geen kernelprestatie. De
CPU-test begint bij de 256 reeds opgebouwde FP32-deelaccumulatoren. Hij bewijst
dus de reductievolgorde en de gemodelleerde bronordening, niet zelfstandig de
CUDA-compilersemantiek van iedere MAC.

De eerstvolgende toegestane stap is een korte GPU-gate waarin gegenereerde
width-16 Q8/Q5-code naast de handgeschreven P7-code wordt gecompileerd en op
synthetische inputs bit-voor-bit wordt vergeleken. Die gate is bewust nog niet
uitgevoerd zolang de gedeelde GPU bezet is. Er wordt in C0 geen snelheids-,
tweede-GPU-, externe-baseline- of nieuwheidsclaim gemaakt.

## Artefacten en provenance

- Preregistratie:
  `reports/streamq5_moe/ERGV_COMPILER_PREREGISTRATION.md`
- Compiler:
  `src/moe_lab/ergv_compiler.py`
- CPU-runner:
  `scripts/streamq5_moe/ergv_compiler_cpu_tests.py`
- Machineleesbaar resultaat:
  `reports/streamq5_moe/ergv_compiler_cpu_tests.json`
- Preregistratie-SHA-256:
  `d52316ad8e2fddd6a42e95b39dab39fb3b429abc93dc06e3821ab64acd985cf3`
- Compiler-SHA-256:
  `6b87e672f392cee34b729726c69a75ab58520440bd5ab20e90bac5be015555c2`
- CPU-runner-SHA-256:
  `4fc4c9f12d06a8483083437b81b7a39fe0bcf09a1529bf5e896bde88cfee38ab`
