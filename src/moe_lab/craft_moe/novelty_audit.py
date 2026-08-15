from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
from urllib.parse import urlparse


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
AUDIT_COMPLETED_UTC = "2026-08-10T19:25:31.5227878Z"

ALLOWED_LABELS = {
    "clearly prior art",
    "close/overlapping",
    "possibly novel intersection",
    "not searched sufficiently",
}

MANDATORY_FAMILY_IDS = (
    "F1_CACHE_CONDITIONAL",
    "F2_COUNTERFACTUAL_ROUTING",
    "F3_REROUTING_AND_RESIDENCY",
    "F4_DYNAMIC_BITS_AND_ROUTING_PTQ",
    "F5_INTRA_EXPERT_SPARSITY",
    "F6_SPECULATIVE_EXPERT_UNIONS",
    "F7_DYNAMIC_EXPERT_SKIPPING",
    "F8_RANDOM_SKETCHES_AND_BITPLANES",
)

CLAIM_IDS = ("CU1", "CU2", "CU3", "CU4", "CU5")


SOURCES: dict[str, dict[str, Any]] = {
    "cache_conditional": {
        "title": "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference",
        "url": "https://arxiv.org/abs/2412.00099",
        "date": "2024-11-27",
        "source_type": "primary_paper",
        "evidence": (
            "Training-free cache-aware routing; Max Rank promotes resident experts within a "
            "bounded rank and Cache-Prior biases router ranking. Evaluated on mobile hardware."
        ),
    },
    "counterfactual_routing": {
        "title": "When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models",
        "url": "https://arxiv.org/abs/2605.07260",
        "date": "2026-05-08",
        "source_type": "primary_paper",
        "evidence": (
            "Scores sampled equal-compute alternative routes in frozen MoEs, including "
            "DeepSeek-V2-Lite, using downstream next-token utility."
        ),
    },
    "buddymoe": {
        "title": "BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference",
        "url": "https://arxiv.org/abs/2511.10054",
        "date": "2025-11-13",
        "source_type": "primary_paper",
        "evidence": (
            "Uses profiled expert redundancy and resident buddy substitution to handle "
            "prefetch misses without simply dropping the missing expert."
        ),
    },
    "sere": {
        "title": "SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models",
        "url": "https://arxiv.org/abs/2602.07616",
        "date": "2026-02-07",
        "source_type": "primary_paper",
        "evidence": (
            "Re-routes secondary experts to similar primary experts, preserves critical "
            "experts, and supplies a vLLM CUDA kernel."
        ),
    },
    "sere_code": {
        "title": "Official SERE implementation",
        "url": "https://github.com/JL-Cheng/SERE",
        "date": "2026-02-07",
        "source_type": "official_code",
        "evidence": "Official calibration and vLLM/CUDA implementation linked by the paper.",
    },
    "moe_eras": {
        "title": "MoE-ERAS: Expert Residency Aware Selection",
        "url": "https://openreview.net/forum?id=o43eHjPEMO",
        "date": "2024-05-30",
        "source_type": "primary_paper",
        "evidence": (
            "Selects experts using both model quality and current residency through "
            "thresholding or biasing, on top of caching and quantization."
        ),
    },
    "remoe": {
        "title": "ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing",
        "url": "https://arxiv.org/abs/2412.14711",
        "date": "2024-12-19",
        "source_type": "primary_paper",
        "evidence": (
            "Training-time ReLU routing enables dynamic allocation across tokens and layers; "
            "it is not a frozen-model post-training route optimizer."
        ),
    },
    "d2moe_dynamic": {
        "title": "D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving",
        "url": "https://doi.org/10.1145/3680207.3723493",
        "date": "2025-11-21",
        "source_type": "peer_reviewed_primary_paper",
        "evidence": (
            "Combines expert routing with dynamic expert bit-width allocation, nested "
            "Matryoshka weights, and bit-aware I/O/compute scheduling."
        ),
    },
    "d2moe_delta": {
        "title": "Delta Decompression for MoE-based LLMs Compression",
        "url": "https://arxiv.org/abs/2502.17298",
        "date": "2025-02-24",
        "source_type": "primary_paper",
        "evidence": (
            "A distinct D²-MoE work that decomposes experts into a Fisher-merged shared base "
            "plus compressed expert-specific deltas."
        ),
    },
    "d2moe_delta_code": {
        "title": "Official D²-MoE delta-decompression implementation",
        "url": "https://github.com/lliai/D2MoE",
        "date": "2025-02-24",
        "source_type": "official_code",
        "evidence": "Official implementation linked by the delta-decompression paper.",
    },
    "slicemoe": {
        "title": "SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference",
        "url": "https://arxiv.org/abs/2512.12990",
        "date": "2025-12-15",
        "source_type": "primary_paper",
        "evidence": (
            "Caches expert bit slices, assigns precision on demand, uses Matryoshka "
            "quantization and predictive cache warmup, including DeepSeek-V2-Lite results."
        ),
    },
    "mobiquant": {
        "title": "MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM",
        "url": "https://arxiv.org/abs/2602.20191",
        "date": "2026-02-21",
        "source_type": "primary_paper",
        "evidence": (
            "Uses recursive residual quantization and a token-aware router to select runtime "
            "precision from additive residual slices."
        ),
    },
    "vsraq": {
        "title": "Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models",
        "url": "https://arxiv.org/abs/2606.05688",
        "date": "2026-06-04",
        "source_type": "primary_paper",
        "evidence": (
            "MoE-specific PTQ objective preserves routing values, ordering, and top-k "
            "boundaries under quantization."
        ),
    },
    "floe": {
        "title": "FloE: On-the-Fly MoE Inference on Memory-constrained GPU",
        "url": "https://arxiv.org/abs/2505.05950",
        "date": "2025-05-09",
        "source_type": "peer_reviewed_primary_paper",
        "evidence": (
            "Compresses internal expert matrices, predicts inter- and intra-expert sparsity, "
            "and implements sparse execution kernels with measured wall-clock results."
        ),
    },
    "intra_expert_sparsity": {
        "title": "Uncovering Intra-expert Activation Sparsity for Efficient Mixture-of-Expert Model Execution",
        "url": "https://arxiv.org/abs/2605.08575",
        "date": "2026-05-09",
        "source_type": "primary_paper",
        "evidence": (
            "Finds substantial neuron-level sparsity in pretrained MoEs and extends vLLM to "
            "skip inactive-neuron computation."
        ),
    },
    "moe_prism": {
        "title": "MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs",
        "url": "https://arxiv.org/abs/2510.19366",
        "date": "2025-10-22",
        "source_type": "primary_paper",
        "evidence": (
            "Refactors experts into neuron-group sub-experts and schedules their online "
            "activation under service-level constraints."
        ),
    },
    "mone": {
        "title": "Mixture of Neuron Experts",
        "url": "https://arxiv.org/abs/2510.05781",
        "date": "2025-10-07",
        "source_type": "primary_paper",
        "evidence": (
            "Decomposes each expert into neuron-granular experts and performs per-expert "
            "top-k neuron selection."
        ),
    },
    "ecospec": {
        "title": "Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts",
        "url": "https://arxiv.org/abs/2607.12696",
        "date": "2026-07-14",
        "source_type": "primary_paper",
        "evidence": (
            "EcoSpec selects draft paths using marginal expert-union cost and reuse in a "
            "dynamic expert buffer while preserving target verification."
        ),
    },
    "acceptmoe": {
        "title": "AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding",
        "url": "https://arxiv.org/abs/2608.02989",
        "date": "2026-08-04",
        "source_type": "primary_paper",
        "evidence": (
            "Chooses a block-level verifier expert set from router scores, commitment "
            "probability, and—in offloading—cache residency."
        ),
    },
    "edgexpert": {
        "title": "EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding",
        "url": "https://arxiv.org/abs/2608.05303",
        "date": "2026-08-05",
        "source_type": "peer_reviewed_primary_paper",
        "evidence": (
            "Uses shared expert sets and depth-aware expert coalescing; loads salient channels "
            "rather than the complete union and supplies hardware co-design."
        ),
    },
    "moe_spec": {
        "title": "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding",
        "url": "https://arxiv.org/abs/2602.16052",
        "date": "2026-02-17",
        "source_type": "primary_paper",
        "evidence": (
            "Applies training-free layerwise expert capacity limits during parallel draft "
            "verification to bound the activated expert union."
        ),
    },
    "specprefetch": {
        "title": "SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models",
        "url": "https://arxiv.org/abs/2607.24787",
        "date": "2026-06-24",
        "source_type": "primary_paper",
        "evidence": (
            "Predicts next-layer expert candidates only for asynchronous transfer and uses a "
            "window-aware cache/bandwidth scheduler without changing native routing."
        ),
    },
    "dynamic_expert_sharing": {
        "title": "Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs",
        "url": "https://arxiv.org/abs/2602.00879",
        "date": "2026-01-31",
        "source_type": "primary_paper",
        "evidence": (
            "Selects a compact sequence-level expert coreset for an entire parallel-decoding "
            "block, directly optimizing expert reuse."
        ),
    },
    "zeda": {
        "title": "Post-Trained MoE Can Skip Half Experts via Self-Distillation",
        "url": "https://arxiv.org/abs/2605.18643",
        "date": "2026-05-18",
        "source_type": "primary_paper",
        "evidence": (
            "ZEDA converts a trained static MoE to dynamic execution using zero experts and "
            "two-stage self-distillation, eliminating over half of expert FLOPs."
        ),
    },
    "beam": {
        "title": "BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE",
        "url": "https://arxiv.org/abs/2605.14438",
        "date": "2026-05-14",
        "source_type": "primary_paper",
        "evidence": (
            "Learns token-adaptive binary masks over routed experts and implements a custom "
            "CUDA/vLLM path that skips masked computation."
        ),
    },
    "optimal_residual_sketch": {
        "title": "Optimal Sketching for Residual Error Estimation for Matrix and Vector Norms",
        "url": "https://arxiv.org/abs/2408.08494",
        "date": "2024-08-16",
        "source_type": "primary_paper",
        "evidence": (
            "Studies linear sketches specifically for estimating residual error and proves "
            "dimension bounds; not an MoE precision controller."
        ),
    },
    "jl_elementary": {
        "title": "An Elementary Proof of a Theorem of Johnson and Lindenstrauss",
        "url": "https://doi.org/10.1002/rsa.10073",
        "date": "2002-11-25",
        "source_type": "peer_reviewed_primary_paper",
        "evidence": (
            "Foundational norm/distance preservation by randomized low-dimensional "
            "projection; establishes that the sketch primitive itself is old."
        ),
    },
    "qmoe": {
        "title": "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models",
        "url": "https://arxiv.org/abs/2310.16795",
        "date": "2023-10-25",
        "source_type": "primary_paper",
        "evidence": (
            "Co-designs a custom sub-1-bit format and bespoke GPU decoding kernels for "
            "compressed MoE execution."
        ),
    },
    "puzzlemoe": {
        "title": "PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference",
        "url": "https://arxiv.org/abs/2511.04805",
        "date": "2025-11-06",
        "source_type": "primary_paper",
        "evidence": (
            "Introduces a bit-packed MoE encoding and measured GPU inference in addition to "
            "sparse expert merging."
        ),
    },
    "numerical_state": {
        "title": "From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference",
        "url": "https://arxiv.org/abs/2607.28097",
        "date": "2026-07-30",
        "source_type": "primary_paper",
        "evidence": (
            "Shows that operand conversion, accumulator precision, and reduction order can "
            "belong to the numerical compatibility contract of sparse-MoE runtimes."
        ),
    },
    "patent_residual_quant": {
        "title": "US11586883B2 — Residual quantization for neural networks",
        "url": "https://patents.google.com/patent/US11586883B2/en",
        "date": "2018-12-14",
        "source_type": "patent_publication",
        "evidence": (
            "Patent record with 2018 priority covering residual quantization mechanisms for "
            "neural networks; no conclusion is made about legal scope here."
        ),
    },
    "patent_partial_hot_expert": {
        "title": "US20250356164A1 — MoE inference with full and partial hot expert buffers",
        "url": "https://patents.google.com/patent/US20250356164A1/en",
        "date": "2025-08-01",
        "source_type": "patent_publication",
        "evidence": (
            "Patent publication describes full and partial expert-weight buffers, chunked "
            "compute, asynchronous prefetch, and cache updates from expert usage."
        ),
    },
    "patent_random_projection": {
        "title": "US20250156706A1 — Pseudo random projection for machine learning compression",
        "url": "https://patents.google.com/patent/US20250156706A1/en",
        "date": "2023-11-14",
        "source_type": "patent_publication",
        "evidence": (
            "Patent publication applies pseudo-random projection structures to machine-"
            "learning compression; it is not an MoE residual precision gate."
        ),
    },
    "patent_sparse_moe_weights": {
        "title": "US20230316042A1 — Mixture of experts models with sparsified weights",
        "url": "https://patents.google.com/patent/US20230316042A1/en",
        "date": "2022-03-31",
        "source_type": "patent_publication",
        "evidence": (
            "Patent publication covers MoE execution with sparsified expert weights at a "
            "broad level; no claim-scope comparison was performed."
        ),
    },
}


FAMILIES: list[dict[str, Any]] = [
    {
        "id": "F1_CACHE_CONDITIONAL",
        "family": "Cache-Conditional Experts / Max Rank / Cache-Prior",
        "mandatory_names": ["Cache-Conditional Experts", "Max Rank", "Cache-Prior"],
        "source_ids": ["cache_conditional", "moe_eras", "patent_partial_hot_expert"],
        "claim_units": ["CU2", "CU4"],
        "label": "clearly prior art",
        "assessment": (
            "Cache-aware promotion, residency-aware route selection, partial expert "
            "residency, and prefetch overlap are established mechanisms. CRAFT cannot claim "
            "cache-conditioned routing or cache-aware expert selection broadly."
        ),
    },
    {
        "id": "F2_COUNTERFACTUAL_ROUTING",
        "family": "Counterfactual Routing Analysis",
        "mandatory_names": ["Counterfactual Routing Analysis"],
        "source_ids": ["counterfactual_routing"],
        "claim_units": ["CU1", "CU2"],
        "label": "clearly prior art",
        "assessment": (
            "Evaluating frozen-model alternative equal-compute routes and showing that "
            "non-natural routes can improve downstream utility is direct prior art. CRAFT's "
            "exhaustive top-12-choose-6 KL protocol is a narrower diagnostic distinction."
        ),
    },
    {
        "id": "F3_REROUTING_AND_RESIDENCY",
        "family": "BuddyMoE, SERE, MoE-ERAS and ReMoE",
        "mandatory_names": ["BuddyMoE", "SERE", "MoE-ERAS", "ReMoE"],
        "source_ids": ["buddymoe", "sere", "sere_code", "moe_eras", "remoe"],
        "claim_units": ["CU1", "CU2", "CU4", "CU5"],
        "label": "close/overlapping",
        "assessment": (
            "Resident substitution, similarity rerouting, residency-aware routing, dynamic "
            "allocation, and custom rerouting kernels overlap strongly. None performs the "
            "exact frozen-model joint route-and-bit oracle tested in CRAFT."
        ),
    },
    {
        "id": "F4_DYNAMIC_BITS_AND_ROUTING_PTQ",
        "family": "D²MoE, SliceMoE, MoBiQuant and routing-consistent PTQ",
        "mandatory_names": ["D²MoE", "SliceMoE", "MoBiQuant", "routing-consistent PTQ"],
        "source_ids": [
            "d2moe_dynamic",
            "d2moe_delta",
            "d2moe_delta_code",
            "slicemoe",
            "mobiquant",
            "vsraq",
        ],
        "claim_units": ["CU1", "CU3", "CU4", "CU5"],
        "label": "close/overlapping",
        "assessment": (
            "Dual expert/precision routing, residual bit slices, on-demand precision, "
            "bit-sliced caching, and routing-aware PTQ are occupied. The exact selection of "
            "a different expert subset jointly with per-expert Q3/Q4 masks was not found."
        ),
    },
    {
        "id": "F5_INTRA_EXPERT_SPARSITY",
        "family": "FloE, intra-expert activation sparsity, MoE-Prism and Mixture of Neuron Experts",
        "mandatory_names": [
            "FloE",
            "intra-expert activation sparsity",
            "MoE-Prism",
            "Mixture of Neuron Experts",
        ],
        "source_ids": ["floe", "intra_expert_sparsity", "moe_prism", "mone"],
        "claim_units": ["CU4", "CU5"],
        "label": "clearly prior art",
        "assessment": (
            "Neuron-level expert decomposition, activation-based neuron skipping, sub-expert "
            "scheduling, sparse MVM execution, and associated kernels are all prior art."
        ),
    },
    {
        "id": "F6_SPECULATIVE_EXPERT_UNIONS",
        "family": "EcoSpec, AcceptMoE, EdgeXpert, MoE-Spec and speculative expert prefetch",
        "mandatory_names": [
            "EcoSpec",
            "AcceptMoE",
            "EdgeXpert",
            "MoE-Spec",
            "speculative expert prefetch",
        ],
        "source_ids": [
            "ecospec",
            "acceptmoe",
            "edgexpert",
            "moe_spec",
            "specprefetch",
            "dynamic_expert_sharing",
        ],
        "claim_units": ["CU2", "CU4", "CU5"],
        "label": "close/overlapping",
        "assessment": (
            "The activated expert union is already an explicit optimization target across a "
            "draft tree or parallel block. EdgeXpert even uses depth-aware expert coalescing, "
            "while DES uses a sequence-level expert coreset. CRAFT's explicit KL-equivalence "
            "slates and exact ILP differ, but the broad system idea is occupied."
        ),
    },
    {
        "id": "F7_DYNAMIC_EXPERT_SKIPPING",
        "family": "ZEDA, BEAM and post-trained dynamic expert skipping",
        "mandatory_names": ["ZEDA", "BEAM", "post-trained dynamic expert skipping"],
        "source_ids": ["zeda", "beam", "sere"],
        "claim_units": ["CU1", "CU2", "CU4", "CU5"],
        "label": "close/overlapping",
        "assessment": (
            "Post-trained token-dependent expert skipping and learned masks are occupied, "
            "including physical kernels. These methods require adaptation or learned gates "
            "and are not the same as CRAFT's teacher-oracle equivalent-route enumeration."
        ),
    },
    {
        "id": "F8_RANDOM_SKETCHES_AND_BITPLANES",
        "family": "Random/JL sketches and adaptive bitplane acquisition",
        "mandatory_names": ["random sketches", "JL sketches", "adaptive bitplane acquisition"],
        "source_ids": [
            "jl_elementary",
            "optimal_residual_sketch",
            "mobiquant",
            "slicemoe",
            "patent_residual_quant",
            "patent_random_projection",
        ],
        "claim_units": ["CU3", "CU4"],
        "label": "close/overlapping",
        "assessment": (
            "Random norm-preserving sketches, residual-error sketches, recursive residual "
            "quantization, and adaptive residual bit slices all predate CRAFT. The targeted "
            "use of a tiny sketch of an MoE expert's Q3→Q4 output residual as a per-token "
            "precision decision was not located."
        ),
    },
]


CLAIMS: list[dict[str, Any]] = [
    {
        "id": "CU1",
        "claim": "Joint alternative-route and quantization-bit selection",
        "label": "possibly novel intersection",
        "closest_source_ids": [
            "counterfactual_routing",
            "d2moe_dynamic",
            "mobiquant",
            "slicemoe",
            "vsraq",
        ],
        "reasoning": (
            "Counterfactual expert routes and token/expert-adaptive precision are both direct "
            "prior art. The targeted search did not locate the exact frozen-model Cartesian "
            "selection of an alternative equal-compute expert subset and a per-route Q3/Q4 "
            "mask against teacher full-vocabulary KL. This is negative search evidence only."
        ),
        "technical_status": "falsified_downstream",
        "technical_evidence": (
            "H1's exhaustive layer-26 oracle required 9.831% validation and 12.240% test Q4 "
            "upgrades, but the preregistered layer-23 intervention required 75.521% and "
            "70.508%; the <=15% downstream gate and >=25% hard-stop both failed."
        ),
        "admissible_claim": (
            "At most: a locally positive diagnostic intersection that failed the required "
            "earlier-layer downstream test. No practical method or Eureka claim."
        ),
    },
    {
        "id": "CU2",
        "claim": "Blockwise route coalescing over explicit equivalence classes",
        "label": "close/overlapping",
        "closest_source_ids": [
            "counterfactual_routing",
            "ecospec",
            "acceptmoe",
            "edgexpert",
            "moe_spec",
            "dynamic_expert_sharing",
        ],
        "reasoning": (
            "Speculative and parallel-decoding systems already optimize union size, expert "
            "reuse, shared verifier sets, depth-aware coalescing, or block coresets. Explicit "
            "KL-certified equivalence classes plus an exact set-union ILP are a narrower "
            "formulation, not a defensible broad novelty claim."
        ),
        "technical_status": "hard_falsified",
        "technical_evidence": (
            "H2's exact block-8 ILP reduced natural union by 19.65% validation and 20.24% "
            "test, below the >=40% gate and the 25% hard-falsification boundary; all 1,280 "
            "ILPs were optimal."
        ),
        "admissible_claim": (
            "An exact negative ceiling for the registered DeepSeek-V2-Lite slate, not a new "
            "serving system."
        ),
    },
    {
        "id": "CU3",
        "claim": "Randomized residual syndrome for precision acquisition",
        "label": "possibly novel intersection",
        "closest_source_ids": [
            "optimal_residual_sketch",
            "jl_elementary",
            "mobiquant",
            "slicemoe",
            "patent_residual_quant",
            "patent_random_projection",
        ],
        "reasoning": (
            "Linear/JL sketches, residual-error estimation, residual quantization and adaptive "
            "bit slices are established. The exact intersection—a small randomized checksum "
            "of the actual Q3→Q4 expert-output residual used to request the next bitplane—was "
            "not located in the bounded search."
        ),
        "technical_status": "falsified",
        "technical_evidence": (
            "H4 met KL-recovery (84.35% validation, 82.26% test) but failed high-damage false "
            "negatives (22.73%/24.68% vs <=1%), down-only attribution (53.76%/65.89% vs "
            ">=70%), and the hardware model (24.30% vs <10%)."
        ),
        "admissible_claim": (
            "A falsified, narrowly specified intersection; no precision controller or "
            "runtime claim."
        ),
    },
    {
        "id": "CU4",
        "claim": "Joint route–bit–atom–cache optimizer",
        "label": "not searched sufficiently",
        "closest_source_ids": [
            "d2moe_dynamic",
            "slicemoe",
            "moe_prism",
            "floe",
            "ecospec",
            "acceptmoe",
            "patent_partial_hot_expert",
        ],
        "reasoning": (
            "Every pair or major subset is densely occupied: route+bits, bits+cache, "
            "atoms+runtime, route+cache, and block+cache. No exact four-axis optimizer was "
            "located, but the project never implemented one, the required component gates "
            "failed, and the patent search was not exhaustive enough to label the full "
            "conjunction."
        ),
        "technical_status": "not_implemented_dependencies_falsified",
        "technical_evidence": (
            "H1, H2, H3 and H4 failed their required downstream/full-depth/system gates. H5, "
            "H9 and PACKED_RUNTIME remained dependency-blocked, so no joint candidate exists."
        ),
        "admissible_claim": (
            "A research question only. It cannot be described as a demonstrated method, "
            "novel contribution, runtime, or Eureka result."
        ),
    },
    {
        "id": "CU5",
        "claim": "Custom kernel or layout for CRAFT-MoE",
        "label": "clearly prior art",
        "closest_source_ids": [
            "qmoe",
            "puzzlemoe",
            "sere",
            "beam",
            "floe",
            "slicemoe",
            "edgexpert",
            "numerical_state",
        ],
        "reasoning": (
            "Bit-packed MoE layouts, bespoke quantized decoders, sparse expert/neuron kernels, "
            "rerouting masks, and hardware co-design are established. A kernel's numerical "
            "semantics also require exact compatibility controls."
        ),
        "technical_status": "not_implemented",
        "technical_evidence": (
            "CRAFT-MoE produced accounting and component microbenchmarks only. PACKED_RUNTIME "
            "was blocked and there is no custom packed kernel or end-to-end decode speedup."
        ),
        "admissible_claim": "No CRAFT-specific kernel/layout claim exists.",
    },
]


PATENT_AUDIT = {
    "scope": (
        "Limited keyword search in Google Patents on 2026-08-10; not a legal search, claim "
        "chart, CPC-class search, family search, prosecution-history review, or FTO analysis."
    ),
    "queries": [
        "mixture of experts cache routing expert weights",
        "mixture of experts dynamic precision quantization bit width",
        "mixture of experts neuron pruning inference",
        "random projection residual quantization neural network",
        "mixture-of-experts quantization caching",
        "expert weights partial cache neural network",
    ],
    "source_ids": [
        "patent_residual_quant",
        "patent_partial_hot_expert",
        "patent_random_projection",
        "patent_sparse_moe_weights",
    ],
    "conclusion": (
        "The records reinforce that residual quantization, partial expert-weight caching, "
        "random-projection compression, and sparse MoE weights are not blank territory. They "
        "do not establish the absence or presence of any legally controlling claim."
    ),
}


def _label_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {label: sum(row["label"] == label for row in rows) for label in sorted(ALLOWED_LABELS)}


def build_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "schema_version": 1,
        "kind": "craft_moe_independent_novelty_and_prior_art_audit",
        "created_at_utc": AUDIT_COMPLETED_UTC,
        "status": "complete",
        "verdict": "no_defensible_broad_novelty_claim_and_no_eureka",
        "search_cutoff": "2026-08-10",
        "methodology": {
            "source_policy": (
                "Primary papers, official proceedings/code, and patent publications for "
                "technical evidence; secondary pages only for discovery."
            ),
            "label_vocabulary": sorted(ALLOWED_LABELS),
            "negative_search_rule": (
                "Failure to locate an exact combination is not evidence that it is new."
            ),
            "patent_scope": PATENT_AUDIT["scope"],
            "no_patentability_opinion": True,
        },
        "baseline": {
            "model": "deepseek-ai/DeepSeek-V2-Lite",
            "model_revision": MODEL_REVISION,
            "dataset": "WikiText-2-raw-v1",
            "dataset_revision": DATASET_REVISION,
            "routing": "64 routed experts; top-6; selected weights not renormalized; 2 shared experts",
        },
        "mandatory_families": deepcopy(FAMILIES),
        "claim_units": deepcopy(CLAIMS),
        "patent_keyword_audit": deepcopy(PATENT_AUDIT),
        "source_catalog": deepcopy(SOURCES),
        "summary": {
            "mandatory_family_count": len(FAMILIES),
            "claim_unit_count": len(CLAIMS),
            "source_count": len(SOURCES),
            "patent_publication_count": sum(
                source["source_type"] == "patent_publication" for source in SOURCES.values()
            ),
            "claim_label_counts": _label_counts(CLAIMS),
            "family_label_counts": _label_counts(FAMILIES),
            "broad_novelty_claim_supported": False,
            "eureka_supported": False,
        },
        "overall": {
            "measured_fact": (
                "The only locally strong new intersections failed required downstream, "
                "full-depth, union-reduction, safety, or hardware gates."
            ),
            "prior_art_finding": (
                "Route alternatives, cache-conditioned selection, block-level expert-union "
                "optimization, token-adaptive residual precision, neuron sparsity, and custom "
                "execution paths are all represented in primary prior art."
            ),
            "inference": (
                "Two exact intersections were not located in this bounded search, but both "
                "were technically falsified in CRAFT and cannot support a contribution claim."
            ),
            "decision": (
                "Do not claim a new compressor, optimizer, kernel, runtime speedup, or Eureka. "
                "Describe CRAFT-MoE as a preregistered negative-results and oracle-ceiling study."
            ),
        },
        "limitations": [
            "The literature cutoff is 2026-08-10; future and unindexed work is absent.",
            "The search is targeted rather than a systematic-review database export.",
            "No citation graph, full-text similarity corpus, non-English database, or thesis repository was exhaustively searched.",
            "The patent pass is a limited keyword search and supports no legal conclusion.",
            "Several 2026 sources are preprints; their reported results were not independently reproduced here.",
            "A possibly novel intersection label is negative search evidence only and never a novelty determination.",
        ],
        "reproducibility": {
            "protocol": "reports/craft_moe/NOVELTY_AUDIT_PROTOCOL.md",
            "generator": "scripts/craft_moe/build_novelty_audit.py",
            "test": "tests/craft_moe/test_novelty_audit.py",
            "command": ".venv\\Scripts\\python.exe scripts\\craft_moe\\build_novelty_audit.py",
            "seed": None,
            "remote_content_frozen": False,
            "deterministic_local_render": True,
        },
    }
    validate_matrix(matrix)
    return matrix


def validate_matrix(matrix: Mapping[str, Any]) -> None:
    errors: list[str] = []
    families = matrix.get("mandatory_families", [])
    claims = matrix.get("claim_units", [])
    sources = matrix.get("source_catalog", {})

    family_ids = tuple(row.get("id") for row in families)
    claim_ids = tuple(row.get("id") for row in claims)
    if family_ids != MANDATORY_FAMILY_IDS:
        errors.append(f"mandatory family IDs/order differ: {family_ids!r}")
    if claim_ids != CLAIM_IDS:
        errors.append(f"claim IDs/order differ: {claim_ids!r}")

    for kind, rows in (("family", families), ("claim", claims)):
        for row in rows:
            if row.get("label") not in ALLOWED_LABELS:
                errors.append(f"{kind} {row.get('id')} has illegal label {row.get('label')!r}")
            referenced = row.get("source_ids", row.get("closest_source_ids", []))
            if not referenced:
                errors.append(f"{kind} {row.get('id')} has no sources")
            for source_id in referenced:
                if source_id not in sources:
                    errors.append(f"{kind} {row.get('id')} references missing source {source_id}")

    urls: list[str] = []
    for source_id, source in sources.items():
        url = source.get("url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"source {source_id} has invalid HTTPS URL {url!r}")
        else:
            urls.append(url)
        if source.get("source_type") not in {
            "primary_paper",
            "peer_reviewed_primary_paper",
            "official_code",
            "patent_publication",
        }:
            errors.append(f"source {source_id} has unsupported source_type")
        if not source.get("evidence"):
            errors.append(f"source {source_id} has no evidence summary")
    if len(urls) != len(set(urls)):
        errors.append("source URLs are not unique")

    patent_audit = matrix.get("patent_keyword_audit", {})
    for source_id in patent_audit.get("source_ids", []):
        if sources.get(source_id, {}).get("source_type") != "patent_publication":
            errors.append(f"patent audit source {source_id} is not a patent publication")

    summary = matrix.get("summary", {})
    if summary.get("mandatory_family_count") != len(MANDATORY_FAMILY_IDS):
        errors.append("summary family count mismatch")
    if summary.get("claim_unit_count") != len(CLAIM_IDS):
        errors.append("summary claim count mismatch")
    if summary.get("source_count") != len(sources):
        errors.append("summary source count mismatch")
    if summary.get("broad_novelty_claim_supported") is not False:
        errors.append("broad novelty claim must remain false")
    if summary.get("eureka_supported") is not False:
        errors.append("Eureka must remain false")

    if errors:
        raise ValueError("; ".join(errors))


def _source_link(source_id: str, sources: Mapping[str, Mapping[str, Any]]) -> str:
    source = sources[source_id]
    return f"[{source['title']}]({source['url']})"


def render_prior_art(matrix: Mapping[str, Any]) -> str:
    validate_matrix(matrix)
    sources = matrix["source_catalog"]
    lines = [
        "# CRAFT-MoE: onafhankelijke prior-artaudit",
        "",
        "Zoekcutoff: 2026-08-10. Dit document scheidt literatuurclaims van onze "
        "eigen metingen. Een niet-gevonden exacte combinatie is geen bewijs van "
        "nieuwheid. De patentpass was beperkt en levert geen juridische conclusie.",
        "",
        "## Samenvatting",
        "",
        "De brede bouwstenen van CRAFT-MoE zijn bezet: alternatieve routes, "
        "cache-aware routing, blockbrede expert-unies, dynamische bit-slices, "
        "neuronfijne sparsity en custom kernels. Twee smalle doorsneden werden in "
        "de gerichte search niet exact gevonden, maar beide faalden hun technische "
        "CRAFT-gates. Er is daarom geen verdedigbare brede nieuwheidsclaim.",
        "",
        "## Verplichte vergelijkingsfamilies",
        "",
    ]
    for family in matrix["mandatory_families"]:
        links = ", ".join(_source_link(source_id, sources) for source_id in family["source_ids"])
        lines.extend(
            [
                f"### {family['id']} — {family['family']}",
                "",
                f"Label: `{family['label']}`.",
                "",
                family["assessment"],
                "",
                f"Primaire bronnen: {links}.",
                "",
            ]
        )

    lines.extend(["## Claimmatrix", ""])
    for claim in matrix["claim_units"]:
        links = ", ".join(
            _source_link(source_id, sources) for source_id in claim["closest_source_ids"]
        )
        lines.extend(
            [
                f"### {claim['id']} — {claim['claim']}",
                "",
                f"Label: `{claim['label']}`. Technische status: "
                f"`{claim['technical_status']}`.",
                "",
                claim["reasoning"],
                "",
                f"Eigen bewijs: {claim['technical_evidence']}",
                "",
                f"Toelaatbare formulering: {claim['admissible_claim']}",
                "",
                f"Dichtstbijzijnde bronnen: {links}.",
                "",
            ]
        )

    patent_links = ", ".join(
        _source_link(source_id, sources)
        for source_id in matrix["patent_keyword_audit"]["source_ids"]
    )
    lines.extend(
        [
            "## Beperkte patentdatabasecontrole",
            "",
            matrix["patent_keyword_audit"]["scope"],
            "",
            matrix["patent_keyword_audit"]["conclusion"],
            "",
            f"Gevonden primaire records: {patent_links}.",
            "",
            "## Eindgrens",
            "",
            matrix["overall"]["decision"],
            "",
            "Deze audit doet uitdrukkelijk geen uitspraak over patentability, "
            "freedom to operate of de volledige stand van de techniek.",
            "",
        ]
    )
    return "\n".join(lines)


def render_verdict(matrix: Mapping[str, Any]) -> str:
    validate_matrix(matrix)
    claim_by_id = {row["id"]: row for row in matrix["claim_units"]}
    lines = [
        "# CRAFT-MoE noveltyverdict",
        "",
        "## Uitkomst",
        "",
        "**Geen verdedigbare brede nieuwheidsclaim en geen Eureka.** De audit vond "
        "veel directe en nabije prior art. De twee smalle, niet exact gevonden "
        "doorsneden zijn bovendien technisch gefalsificeerd; de volledige "
        "route–bit–atom–cache-stack is nooit dependency-vrij geïmplementeerd.",
        "",
        "| Claim | Label | Technische status |",
        "|---|---|---|",
    ]
    for claim_id in CLAIM_IDS:
        claim = claim_by_id[claim_id]
        lines.append(
            f"| {claim_id}: {claim['claim']} | `{claim['label']}` | "
            f"`{claim['technical_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Waarom dit sluitend genoeg is voor het projectbesluit",
            "",
            "- CU1 is alleen lokaal positief en faalt de verplichte eerdere-laagtest hard.",
            "- CU2 heeft met exacte optimale ILP's een te laag block-unionplafond.",
            "- CU3 haalt één recoverymetric maar faalt veiligheid, attributie en hardwaremodel.",
            "- CU4 heeft geen uitvoerbare kandidaat omdat de componentgates faalden.",
            "- CU5 bestaat niet in deze repository; packed runtime bleef geblokkeerd.",
            "",
            "De correcte bijdrageformulering is daarom: een streng gepinde, "
            "preregistered negatieve-resultatenstudie met exacte oracleplafonds en "
            "reproduceerbare falsificatie. Projected bytes of microbenchmarks zijn geen "
            "end-to-end speedup.",
            "",
            "## Beperkingen",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in matrix["limitations"])
    lines.extend(
        [
            "",
            "Stop/go: **stop de huidige CRAFT-hypothesefamilie en download V4 Flash "
            "niet**. Een vervolg vereist een nieuwe, onafhankelijk gemotiveerde hypothese "
            "met nieuwe preregistratie; post-hoc varianten van de gesloten mechanismen "
            "mogen dit verdict niet vervangen.",
            "",
        ]
    )
    return "\n".join(lines)
