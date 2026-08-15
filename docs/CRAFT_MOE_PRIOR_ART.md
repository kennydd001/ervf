# CRAFT-MoE: onafhankelijke prior-artaudit

Zoekcutoff: 2026-08-10. Dit document scheidt literatuurclaims van onze eigen metingen. Een niet-gevonden exacte combinatie is geen bewijs van nieuwheid. De patentpass was beperkt en levert geen juridische conclusie.

## Samenvatting

De brede bouwstenen van CRAFT-MoE zijn bezet: alternatieve routes, cache-aware routing, blockbrede expert-unies, dynamische bit-slices, neuronfijne sparsity en custom kernels. Twee smalle doorsneden werden in de gerichte search niet exact gevonden, maar beide faalden hun technische CRAFT-gates. Er is daarom geen verdedigbare brede nieuwheidsclaim.

## Verplichte vergelijkingsfamilies

### F1_CACHE_CONDITIONAL — Cache-Conditional Experts / Max Rank / Cache-Prior

Label: `clearly prior art`.

Cache-aware promotion, residency-aware route selection, partial expert residency, and prefetch overlap are established mechanisms. CRAFT cannot claim cache-conditioned routing or cache-aware expert selection broadly.

Primaire bronnen: [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference](https://arxiv.org/abs/2412.00099), [MoE-ERAS: Expert Residency Aware Selection](https://openreview.net/forum?id=o43eHjPEMO), [US20250356164A1 — MoE inference with full and partial hot expert buffers](https://patents.google.com/patent/US20250356164A1/en).

### F2_COUNTERFACTUAL_ROUTING — Counterfactual Routing Analysis

Label: `clearly prior art`.

Evaluating frozen-model alternative equal-compute routes and showing that non-natural routes can improve downstream utility is direct prior art. CRAFT's exhaustive top-12-choose-6 KL protocol is a narrower diagnostic distinction.

Primaire bronnen: [When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models](https://arxiv.org/abs/2605.07260).

### F3_REROUTING_AND_RESIDENCY — BuddyMoE, SERE, MoE-ERAS and ReMoE

Label: `close/overlapping`.

Resident substitution, similarity rerouting, residency-aware routing, dynamic allocation, and custom rerouting kernels overlap strongly. None performs the exact frozen-model joint route-and-bit oracle tested in CRAFT.

Primaire bronnen: [BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference](https://arxiv.org/abs/2511.10054), [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](https://arxiv.org/abs/2602.07616), [Official SERE implementation](https://github.com/JL-Cheng/SERE), [MoE-ERAS: Expert Residency Aware Selection](https://openreview.net/forum?id=o43eHjPEMO), [ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing](https://arxiv.org/abs/2412.14711).

### F4_DYNAMIC_BITS_AND_ROUTING_PTQ — D²MoE, SliceMoE, MoBiQuant and routing-consistent PTQ

Label: `close/overlapping`.

Dual expert/precision routing, residual bit slices, on-demand precision, bit-sliced caching, and routing-aware PTQ are occupied. The exact selection of a different expert subset jointly with per-expert Q3/Q4 masks was not found.

Primaire bronnen: [D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving](https://doi.org/10.1145/3680207.3723493), [Delta Decompression for MoE-based LLMs Compression](https://arxiv.org/abs/2502.17298), [Official D²-MoE delta-decompression implementation](https://github.com/lliai/D2MoE), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM](https://arxiv.org/abs/2602.20191), [Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models](https://arxiv.org/abs/2606.05688).

### F5_INTRA_EXPERT_SPARSITY — FloE, intra-expert activation sparsity, MoE-Prism and Mixture of Neuron Experts

Label: `clearly prior art`.

Neuron-level expert decomposition, activation-based neuron skipping, sub-expert scheduling, sparse MVM execution, and associated kernels are all prior art.

Primaire bronnen: [FloE: On-the-Fly MoE Inference on Memory-constrained GPU](https://arxiv.org/abs/2505.05950), [Uncovering Intra-expert Activation Sparsity for Efficient Mixture-of-Expert Model Execution](https://arxiv.org/abs/2605.08575), [MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs](https://arxiv.org/abs/2510.19366), [Mixture of Neuron Experts](https://arxiv.org/abs/2510.05781).

### F6_SPECULATIVE_EXPERT_UNIONS — EcoSpec, AcceptMoE, EdgeXpert, MoE-Spec and speculative expert prefetch

Label: `close/overlapping`.

The activated expert union is already an explicit optimization target across a draft tree or parallel block. EdgeXpert even uses depth-aware expert coalescing, while DES uses a sequence-level expert coreset. CRAFT's explicit KL-equivalence slates and exact ILP differ, but the broad system idea is occupied.

Primaire bronnen: [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2607.12696), [AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding](https://arxiv.org/abs/2608.02989), [EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding](https://arxiv.org/abs/2608.05303), [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052), [SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models](https://arxiv.org/abs/2607.24787), [Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](https://arxiv.org/abs/2602.00879).

### F7_DYNAMIC_EXPERT_SKIPPING — ZEDA, BEAM and post-trained dynamic expert skipping

Label: `close/overlapping`.

Post-trained token-dependent expert skipping and learned masks are occupied, including physical kernels. These methods require adaptation or learned gates and are not the same as CRAFT's teacher-oracle equivalent-route enumeration.

Primaire bronnen: [Post-Trained MoE Can Skip Half Experts via Self-Distillation](https://arxiv.org/abs/2605.18643), [BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE](https://arxiv.org/abs/2605.14438), [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](https://arxiv.org/abs/2602.07616).

### F8_RANDOM_SKETCHES_AND_BITPLANES — Random/JL sketches and adaptive bitplane acquisition

Label: `close/overlapping`.

Random norm-preserving sketches, residual-error sketches, recursive residual quantization, and adaptive residual bit slices all predate CRAFT. The targeted use of a tiny sketch of an MoE expert's Q3→Q4 output residual as a per-token precision decision was not located.

Primaire bronnen: [An Elementary Proof of a Theorem of Johnson and Lindenstrauss](https://doi.org/10.1002/rsa.10073), [Optimal Sketching for Residual Error Estimation for Matrix and Vector Norms](https://arxiv.org/abs/2408.08494), [MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM](https://arxiv.org/abs/2602.20191), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [US11586883B2 — Residual quantization for neural networks](https://patents.google.com/patent/US11586883B2/en), [US20250156706A1 — Pseudo random projection for machine learning compression](https://patents.google.com/patent/US20250156706A1/en).

## Claimmatrix

### CU1 — Joint alternative-route and quantization-bit selection

Label: `possibly novel intersection`. Technische status: `falsified_downstream`.

Counterfactual expert routes and token/expert-adaptive precision are both direct prior art. The targeted search did not locate the exact frozen-model Cartesian selection of an alternative equal-compute expert subset and a per-route Q3/Q4 mask against teacher full-vocabulary KL. This is negative search evidence only.

Eigen bewijs: H1's exhaustive layer-26 oracle required 9.831% validation and 12.240% test Q4 upgrades, but the preregistered layer-23 intervention required 75.521% and 70.508%; the <=15% downstream gate and >=25% hard-stop both failed.

Toelaatbare formulering: At most: a locally positive diagnostic intersection that failed the required earlier-layer downstream test. No practical method or Eureka claim.

Dichtstbijzijnde bronnen: [When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models](https://arxiv.org/abs/2605.07260), [D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving](https://doi.org/10.1145/3680207.3723493), [MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM](https://arxiv.org/abs/2602.20191), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models](https://arxiv.org/abs/2606.05688).

### CU2 — Blockwise route coalescing over explicit equivalence classes

Label: `close/overlapping`. Technische status: `hard_falsified`.

Speculative and parallel-decoding systems already optimize union size, expert reuse, shared verifier sets, depth-aware coalescing, or block coresets. Explicit KL-certified equivalence classes plus an exact set-union ILP are a narrower formulation, not a defensible broad novelty claim.

Eigen bewijs: H2's exact block-8 ILP reduced natural union by 19.65% validation and 20.24% test, below the >=40% gate and the 25% hard-falsification boundary; all 1,280 ILPs were optimal.

Toelaatbare formulering: An exact negative ceiling for the registered DeepSeek-V2-Lite slate, not a new serving system.

Dichtstbijzijnde bronnen: [When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models](https://arxiv.org/abs/2605.07260), [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2607.12696), [AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding](https://arxiv.org/abs/2608.02989), [EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding](https://arxiv.org/abs/2608.05303), [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052), [Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](https://arxiv.org/abs/2602.00879).

### CU3 — Randomized residual syndrome for precision acquisition

Label: `possibly novel intersection`. Technische status: `falsified`.

Linear/JL sketches, residual-error estimation, residual quantization and adaptive bit slices are established. The exact intersection—a small randomized checksum of the actual Q3→Q4 expert-output residual used to request the next bitplane—was not located in the bounded search.

Eigen bewijs: H4 met KL-recovery (84.35% validation, 82.26% test) but failed high-damage false negatives (22.73%/24.68% vs <=1%), down-only attribution (53.76%/65.89% vs >=70%), and the hardware model (24.30% vs <10%).

Toelaatbare formulering: A falsified, narrowly specified intersection; no precision controller or runtime claim.

Dichtstbijzijnde bronnen: [Optimal Sketching for Residual Error Estimation for Matrix and Vector Norms](https://arxiv.org/abs/2408.08494), [An Elementary Proof of a Theorem of Johnson and Lindenstrauss](https://doi.org/10.1002/rsa.10073), [MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM](https://arxiv.org/abs/2602.20191), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [US11586883B2 — Residual quantization for neural networks](https://patents.google.com/patent/US11586883B2/en), [US20250156706A1 — Pseudo random projection for machine learning compression](https://patents.google.com/patent/US20250156706A1/en).

### CU4 — Joint route–bit–atom–cache optimizer

Label: `not searched sufficiently`. Technische status: `not_implemented_dependencies_falsified`.

Every pair or major subset is densely occupied: route+bits, bits+cache, atoms+runtime, route+cache, and block+cache. No exact four-axis optimizer was located, but the project never implemented one, the required component gates failed, and the patent search was not exhaustive enough to label the full conjunction.

Eigen bewijs: H1, H2, H3 and H4 failed their required downstream/full-depth/system gates. H5, H9 and PACKED_RUNTIME remained dependency-blocked, so no joint candidate exists.

Toelaatbare formulering: A research question only. It cannot be described as a demonstrated method, novel contribution, runtime, or Eureka result.

Dichtstbijzijnde bronnen: [D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving](https://doi.org/10.1145/3680207.3723493), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs](https://arxiv.org/abs/2510.19366), [FloE: On-the-Fly MoE Inference on Memory-constrained GPU](https://arxiv.org/abs/2505.05950), [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2607.12696), [AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding](https://arxiv.org/abs/2608.02989), [US20250356164A1 — MoE inference with full and partial hot expert buffers](https://patents.google.com/patent/US20250356164A1/en).

### CU5 — Custom kernel or layout for CRAFT-MoE

Label: `clearly prior art`. Technische status: `not_implemented`.

Bit-packed MoE layouts, bespoke quantized decoders, sparse expert/neuron kernels, rerouting masks, and hardware co-design are established. A kernel's numerical semantics also require exact compatibility controls.

Eigen bewijs: CRAFT-MoE produced accounting and component microbenchmarks only. PACKED_RUNTIME was blocked and there is no custom packed kernel or end-to-end decode speedup.

Toelaatbare formulering: No CRAFT-specific kernel/layout claim exists.

Dichtstbijzijnde bronnen: [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795), [PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference](https://arxiv.org/abs/2511.04805), [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](https://arxiv.org/abs/2602.07616), [BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE](https://arxiv.org/abs/2605.14438), [FloE: On-the-Fly MoE Inference on Memory-constrained GPU](https://arxiv.org/abs/2505.05950), [SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference](https://arxiv.org/abs/2512.12990), [EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding](https://arxiv.org/abs/2608.05303), [From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference](https://arxiv.org/abs/2607.28097).

## Beperkte patentdatabasecontrole

Limited keyword search in Google Patents on 2026-08-10; not a legal search, claim chart, CPC-class search, family search, prosecution-history review, or FTO analysis.

The records reinforce that residual quantization, partial expert-weight caching, random-projection compression, and sparse MoE weights are not blank territory. They do not establish the absence or presence of any legally controlling claim.

Gevonden primaire records: [US11586883B2 — Residual quantization for neural networks](https://patents.google.com/patent/US11586883B2/en), [US20250356164A1 — MoE inference with full and partial hot expert buffers](https://patents.google.com/patent/US20250356164A1/en), [US20250156706A1 — Pseudo random projection for machine learning compression](https://patents.google.com/patent/US20250156706A1/en), [US20230316042A1 — Mixture of experts models with sparsified weights](https://patents.google.com/patent/US20230316042A1/en).

## Eindgrens

Do not claim a new compressor, optimizer, kernel, runtime speedup, or Eureka. Describe CRAFT-MoE as a preregistered negative-results and oracle-ceiling study.

Deze audit doet uitdrukkelijk geen uitspraak over patentability, freedom to operate of de volledige stand van de techniek.
