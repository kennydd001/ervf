from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


MASTER_COMPLETED_UTC = "2026-08-10T19:27:13.5957082Z"

REVOLUTIONARY_GATE_IDS = (
    "G1_ACTIVE_EXPERT_BYTES",
    "G2_WALLCLOCK_DECODE",
    "G3_RELATIVE_CE",
    "G4_LONG_ROLLOUTS",
    "G5_P95_LATENCY",
    "G6_SECOND_MODEL",
)

TECHNICAL_IDS = (
    "H7_ROUTE_CORESET",
    "H1_CRCQ",
    "H3_ATOMIC_ORACLE",
    "H4_SKETCHGATE",
    "H2_BLOCK_COALESCING",
    "H5_ATOMIC_INDEX",
    "H6_QERC",
    "H8_CACHE_SPAN",
    "H9_BISPARSE",
    "H10_REDUCTION_ORDER",
    "PACKED_RUNTIME",
)


REVOLUTIONARY_GATES: list[dict[str, Any]] = [
    {
        "id": "G1_ACTIVE_EXPERT_BYTES",
        "requirement": ">=4,0× minder actieve expertbytes dan packed int4 + Mass-Budget delta=0,004",
        "satisfied": False,
        "status": "not_demonstrated",
        "evidence": (
            "Er bestaat geen full-depth, kwaliteitsgekwalificeerde CRAFT-kandidaat of "
            "packed layout. De 25%-BF16-atomic-oracle is ideaal 4 effectieve bits per "
            "volledig expert: slechts gelijk aan gewone int4 vóór metadata en vóór "
            "Mass-Budgets loadbesparing."
        ),
    },
    {
        "id": "G2_WALLCLOCK_DECODE",
        "requirement": ">=2,0× gemeten batch-1-decodespeedup versus packed int4 + Mass-Budget",
        "satisfied": False,
        "status": "not_evaluated_dependency_stop",
        "evidence": (
            "PACKED_RUNTIME was dependency-geblokkeerd. De H4/H8-CUDA-metingen zijn "
            "componentmicrobenchmarks zonder temperatuur-/kloktelemetrie en expliciet geen "
            "runtimeclaims."
        ),
    },
    {
        "id": "G3_RELATIVE_CE",
        "requirement": "<2% relatieve cross-entropyschade voor dezelfde kandidaat",
        "satisfied": False,
        "status": "not_satisfied_by_any_byte_qualified_candidate",
        "evidence": (
            "De sterkste relevante full-depth H3-25%-oracle bereikt +2,1129% WikiText-test-"
            "CE en faalt ook lokale-instructie-KL (0,03505 > 0,03). Mass-Budget slaagt voor "
            "zijn eigen CE-gate, maar benadert de vereiste bytereductie niet."
        ),
    },
    {
        "id": "G4_LONG_ROLLOUTS",
        "requirement": ">=512 gegenereerde tokens op >=20 prompts over meerdere taaktypes",
        "satisfied": False,
        "status": "not_evaluated_no_candidate",
        "evidence": (
            "Geen CRAFT-methode slaagde voor de vereiste oracle-, downstream- en full-depth-"
            "gates. Korte oudere Mass-Budget-smokerollouts voldoen niet aan deze gate."
        ),
    },
    {
        "id": "G5_P95_LATENCY",
        "requirement": "Geen problematische p95 batch-1-decodelatency",
        "satisfied": False,
        "status": "not_evaluated_no_packed_runtime",
        "evidence": "Er is geen packed end-to-end kandidaat-runtime of p95-verdeling.",
    },
    {
        "id": "G6_SECOND_MODEL",
        "requirement": "Alle relevante gates repliceren op een tweede MoE-familie",
        "satisfied": False,
        "status": "not_evaluated_v2_gate_failed",
        "evidence": (
            "De onveranderlijke stopregel verbiedt V4 Flash of escalatie naar een tweede "
            "familie voordat DeepSeek-V2-Lite de volledige revolutionaire gate haalt."
        ),
    },
]


TECHNICAL_PROGRAM: list[dict[str, str]] = [
    {
        "id": "H7_ROUTE_CORESET",
        "status": "inconclusive_negative",
        "decisive_result": "Validatie/test minimum-k-mediaan 4 en empirische p95 6; de positieve mediaan<=3- of p95<=4-gate faalde.",
    },
    {
        "id": "H1_CRCQ",
        "status": "falsified_downstream",
        "decisive_result": "De gezamenlijke laag-26-oracle vraagt 9,831%/12,240% upgrades, maar de laag-23 exact-tailinterventie 75,521%/70,508%.",
    },
    {
        "id": "H3_ATOMIC_ORACLE",
        "status": "falsified_full_depth",
        "decisive_result": "De lokale/spread-oracles bij 25% en 10% waren positief; gelijktijdig full-depth 25% faalt op +2,1129% WikiText-test-CE en instructie-KL 0,03505.",
    },
    {
        "id": "H4_SKETCHGATE",
        "status": "falsified",
        "decisive_result": "KL-recovery slaagt, maar high-damage false negatives, attributie en hardwaremodelgates falen.",
    },
    {
        "id": "H2_BLOCK_COALESCING",
        "status": "hard_falsified",
        "decisive_result": "Alle 1.280 exacte ILP's zijn optimaal; de block-8 natural-unionreductie is slechts 19,65%/20,24%, zelfs onder de 25%-hardstop.",
    },
    {
        "id": "H5_ATOMIC_INDEX",
        "status": "blocked_by_H3",
        "decisive_result": "De predictor was verboden nadat de gelijktijdige full-depth atomic-gate faalde.",
    },
    {
        "id": "H6_QERC",
        "status": "hard_falsified_phase_a",
        "decisive_result": "Natuurlijke Q3-cancellation is -1,129% validatie en -0,106% test; beide liggen in de vooraf vastgelegde near-zero-stopband.",
    },
    {
        "id": "H8_CACHE_SPAN",
        "status": "inconclusive_negative",
        "decisive_result": "Oracle-missreductie is 41,35%/48,54% en span-uplift boven zero-fill slechts +1,442/+0,225 procentpunt.",
    },
    {
        "id": "H9_BISPARSE",
        "status": "blocked_by_H3",
        "decisive_result": "Het bi-sparse-kernelpad was verboden nadat de gelijktijdige full-depth atomic-gate faalde.",
    },
    {
        "id": "H10_REDUCTION_ORDER",
        "status": "hard_falsified",
        "decisive_result": "Held-out Q3→Q4-KL-gapclosure is 1,487%/0,829% tegenover de >=20%-gate en <10%-hardstop.",
    },
    {
        "id": "PACKED_RUNTIME",
        "status": "blocked",
        "decisive_result": "Geen methode haalde de vereiste geprojecteerde winst en kwaliteits-/downstreamgates.",
    },
]


MEASURED_FACTS = [
    {
        "fact": "Mass-Budget is een bevestigde incrementele baseline, geen CRAFT-Eureka.",
        "evidence": (
            "Op het vooraf vastgelegde testvenster van 2.048 tokens reduceert delta=0,004 de "
            "expertloads met 14,017%, bij KL 0,003704 en relatieve CE -0,057% (95%-blokinterval "
            "-0,171% tot +0,060%); alle 16 blokken besparen loads."
        ),
    },
    {
        "fact": "Alle niet-geblokkeerde CRAFT-hypothesen bereikten een terminaal negatief besluit.",
        "evidence": (
            "H1, H2, H3, H4, H6 en H10 zijn gefalsificeerd; H7 en H8 zijn "
            "inconclusief-negatieve screens met gefaalde positieve gates; H5, H9 en packed "
            "runtime stopten door afhankelijkheden."
        ),
    },
    {
        "fact": "Er bestaat geen full-depth inzetbare CRAFT-kandidaat.",
        "evidence": (
            "Lokale H1/H3-oraclewinsten overleefden de vereiste eerdere-laag- of "
            "gelijktijdige-full-depthtest niet; daarna was geen predictor of kernel toegestaan."
        ),
    },
    {
        "fact": "Het technische verdict is onafhankelijk reproduceerbaar.",
        "evidence": (
            "De bevroren reproduceerbaarheidsaudit bevat 751 oorspronkelijke controles: "
            "748 geslaagd, nul verplichte fouten, drie claimbegrenzende waarschuwingen en een "
            "hashmanifest met 136 entries."
        ),
    },
    {
        "fact": "Brede nieuwheid wordt niet ondersteund.",
        "evidence": (
            "De novelty-audit omvat acht verplichte families, vijf claim-eenheden en 34 "
            "primaire/official/patentbronnen; geen brede claim blijft overeind."
        ),
    },
]


DERIVED_ACCOUNTING = [
    {
        "derivation": "Geprojecteerde strict-I/O-ratio van Mass-Budget",
        "calculation": "201,48 / 173,24 = 1,163011× (14,0163% geprojecteerde bytebesparing)",
        "boundary": "Alleen deterministische packed-int4-boekhouding; geen latency, throughput, energie of gemeten transfertraffic.",
    },
    {
        "derivation": "H2-test-block-unionfactor",
        "calculation": "1 - 0,2024 = 0,7976, gelijk aan slechts 1,2538× minder unieke-experteenheden in de geïdealiseerde uniontelling",
        "boundary": "Eén exacte late-laag-oracle; geen speculative runtime- of full-depthclaim.",
    },
    {
        "derivation": "H1 effectieve lokale precisie",
        "calculation": "Laag 26: 3 + 0,12240 = 3,12240 bits; laag-23-test: 3 + 0,70508 = 3,70508 bits",
        "boundary": "Oracle-geselecteerde masks vereisen teacherwerk en vormen geen causale runtimecontroller.",
    },
    {
        "derivation": "Ideale BF16-atomfractie versus int4",
        "calculation": "25% × 16 bit = 4,0 effectieve bits; 10% × 16 bit = 1,6 effectieve bits, of maximaal 2,5× versus int4 bij gelijke loads",
        "boundary": "Negeert indices, tiles, cachelines en kernels. Beide gelijktijdige full-depthpolicies faalden; 25% is op ruwe weightbits niet beter dan gewone int4.",
    },
    {
        "derivation": "Geen geldige multiplicatieve CRAFT-stack",
        "calculation": "Route-, bit-, atom- en cachefactoren mogen niet worden vermenigvuldigd, omdat ze op verschillende interventies zijn gemeten en meerdere vereiste gates faalden",
        "boundary": "Alle eerdere 7,36/4,94-GB-per-output-stackramingen blijven hypothetische scenario-aritmetiek, geen resultaat.",
    },
]


def build_master_verdict() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "craft_moe_master_verdict",
        "created_at_utc": MASTER_COMPLETED_UTC,
        "status": "complete",
        "project_status": "closed_no_eureka",
        "verdict": "hypothesis_pack_convincingly_closed_without_eureka",
        "baseline": {
            "model": "deepseek-ai/DeepSeek-V2-Lite",
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "dataset": "WikiText-2-raw-v1",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "routing": "64 routed experts; top-6; no selected-weight renormalization; 2 shared experts",
            "strong_system_baseline": "packed int4 + Mass-Budget delta=0.004",
        },
        "revolutionary_v2_gate": {
            "conditions": deepcopy(REVOLUTIONARY_GATES),
            "satisfied_count": sum(row["satisfied"] for row in REVOLUTIONARY_GATES),
            "condition_count": len(REVOLUTIONARY_GATES),
            "all_satisfied_by_one_candidate": False,
        },
        "technical_program": deepcopy(TECHNICAL_PROGRAM),
        "measured_facts": deepcopy(MEASURED_FACTS),
        "derived_accounting": deepcopy(DERIVED_ACCOUNTING),
        "subjective_inference": [
            (
                "Het dominante obstakel is cumulatief full-depthgedrag en uitvoerbare "
                "databeweging, niet een ontbrekende kleine predictortweak."
            ),
            (
                "Mass-Budget blijft een nuttige incrementele baseline, maar zijn 14% "
                "geprojecteerde loadbesparing verschilt kwalitatief van het >=4×-boven-int4-doel."
            ),
            (
                "Verdere post-hocvarianten binnen deze hypothesefamilie zouden het vooraf "
                "vastgelegde bewijs verzwakken, niet een geloofwaardige doorbraak creëren."
            ),
            (
                "De sterkste onderzoeksbijdrage is de negatieve kaart: exacte "
                "oracleplafonds, dependencystops en onafhankelijke gateverificatie."
            ),
        ],
        "novelty_status": {
            "verdict": "no_defensible_broad_novelty_claim",
            "possibly_novel_intersections": ["CU1", "CU3"],
            "qualification": (
                "Beide exacte doorsneden werden slechts niet gevonden in een begrensde search "
                "en beide zijn technisch gefalsificeerd; dit ondersteunt geen nieuwheids- of "
                "praktische claim."
            ),
            "joint_optimizer": "`not searched sufficiently` en niet geïmplementeerd",
            "custom_kernel_layout": "breed `clearly prior art`; geen CRAFT-implementatie",
            "source": "reports/craft_moe/novelty_matrix.json",
        },
        "audit_status": {
            "reproducibility": "passed_with_3_declared_warnings",
            "reproducibility_result": "reports/craft_moe/repro_audit.json",
            "novelty": "complete_no_broad_claim",
            "novelty_result": "reports/craft_moe/novelty_matrix.json",
        },
        "exact_next_action": {
            "action": "freeze_and_close_current_craft_hypothesis_pack",
            "do_not": [
                "DeepSeek V4 Flash downloaden of daarnaar escaleren",
                "een packed kernel bouwen voor de gefaalde CRAFT-kandidaten",
                "incompatibele oraclefactoren vermenigvuldigen tot een systemspeedupclaim",
                "een gefaalde gate herafstellen op held-out testdata",
            ],
            "if_research_restarts": (
                "Open een nieuw registry-item voor een mechanistisch onafhankelijke hypothese, "
                "registreer oracle-, full-depth-, packed-runtime- en tweede-modelgates vooraf "
                "en laat ieder gesloten CRAFT-resultaat ongewijzigd."
            ),
        },
        "closure_basis": (
            "Het gebruikersdoel staat sluiting toe zodra Eureka is bewezen of de onderzochte "
            "hypothesen overtuigend zijn gefalsificeerd. Aan de tweede voorwaarde is voldaan."
        ),
        "limitations": [
            "Dit sluit het geregistreerde CRAFT-MoE-hypothesepakket, niet iedere denkbare MoE-compressiemethode.",
            "H7 en H8 zijn inconclusief-negatief, geen universele onmogelijkheidsbewijzen.",
            "Er is geen fysieke packed-runtimebaseline gebouwd, omdat geen afhankelijke kandidaat de stopgates haalde.",
            "De lokale repository heeft geen Git-commit; integriteit rust op gepinde upstreamrevisions en artefacthashes.",
            "De novelty- en patentsearches zijn begrensd en ondersteunen geen juridische conclusie.",
        ],
        "reproducibility": {
            "protocol_sources": [
                "reports/craft_moe/EXPERIMENT_REGISTRY.yaml",
                "reports/craft_moe/REPRO_AUDIT_PROTOCOL.md",
                "reports/craft_moe/NOVELTY_AUDIT_PROTOCOL.md",
            ],
            "generator": "scripts/craft_moe/build_master_verdict.py",
            "test": "tests/craft_moe/test_master_verdict.py",
            "command": ".venv\\Scripts\\python.exe scripts\\craft_moe\\build_master_verdict.py",
            "deterministic_local_render": True,
        },
    }
    validate_master_verdict(payload)
    return payload


def validate_master_verdict(payload: Mapping[str, Any]) -> None:
    errors: list[str] = []
    gate = payload.get("revolutionary_v2_gate", {})
    conditions = gate.get("conditions", [])
    ids = tuple(row.get("id") for row in conditions)
    if ids != REVOLUTIONARY_GATE_IDS:
        errors.append(f"revolutionary gate IDs/order differ: {ids!r}")
    recalculated = sum(bool(row.get("satisfied")) for row in conditions)
    if gate.get("satisfied_count") != recalculated:
        errors.append("revolutionary gate satisfied_count mismatch")
    all_satisfied = bool(conditions) and all(bool(row.get("satisfied")) for row in conditions)
    if gate.get("all_satisfied_by_one_candidate") != all_satisfied:
        errors.append("revolutionary aggregate gate mismatch")
    if all_satisfied:
        errors.append("unexpected Eureka: all revolutionary gates cannot be true in this evidence pack")

    technical_ids = tuple(row.get("id") for row in payload.get("technical_program", []))
    if technical_ids != TECHNICAL_IDS:
        errors.append(f"technical program IDs/order differ: {technical_ids!r}")
    if payload.get("project_status") != "closed_no_eureka":
        errors.append("project status must be closed_no_eureka")
    if payload.get("verdict") != "hypothesis_pack_convincingly_closed_without_eureka":
        errors.append("master verdict was promoted or changed")

    novelty = payload.get("novelty_status", {})
    if novelty.get("verdict") != "no_defensible_broad_novelty_claim":
        errors.append("novelty verdict mismatch")
    next_action = payload.get("exact_next_action", {})
    if next_action.get("action") != "freeze_and_close_current_craft_hypothesis_pack":
        errors.append("next action mismatch")
    if not any("V4 Flash" in item for item in next_action.get("do_not", [])):
        errors.append("V4 Flash stop missing")

    accounting = " ".join(
        f"{row.get('derivation', '')} {row.get('calculation', '')} {row.get('boundary', '')}"
        for row in payload.get("derived_accounting", [])
    )
    if "mogen niet worden vermenigvuldigd" not in accounting:
        errors.append("invalid-factor multiplication boundary missing")
    if errors:
        raise ValueError("; ".join(errors))


def render_master_verdict(payload: Mapping[str, Any]) -> str:
    validate_master_verdict(payload)
    lines = [
        "# CRAFT-MoE masterverdict",
        "",
        "## Uitkomst",
        "",
        "**De geregistreerde CRAFT-MoE-hypothesefamilie is overtuigend gesloten "
        "zonder Eureka.** Geen enkele kandidaat voldoet aan de gezamenlijke V2-gate. "
        "Alle niet-geblokkeerde hypothesen hebben een terminal negatief besluit; de "
        "geblokkeerde engineeringstappen zijn volgens de vooraf vastgelegde stopregels "
        "terecht niet uitgevoerd.",
        "",
        "Dit is geen universeel onmogelijkheidsbewijs voor MoE-compressie. Het is wel "
        "voldoende bewijs om deze specifieke route–bit–atom–cache-stack niet verder "
        "post-hoc te variëren en niet naar V4 Flash te escaleren.",
        "",
        "## Gemeten feiten",
        "",
    ]
    for row in payload["measured_facts"]:
        lines.append(f"- **{row['fact']}** {row['evidence']}")

    lines.extend(
        [
            "",
            "## Technische hypothesen",
            "",
            "| ID | Terminale status | Doorslaggevend resultaat |",
            "|---|---|---|",
        ]
    )
    for row in payload["technical_program"]:
        lines.append(f"| {row['id']} | `{row['status']}` | {row['decisive_result']} |")

    lines.extend(
        [
            "",
            "## Revolutionaire V2-gate",
            "",
            "De gate geldt conjunctief: één en dezelfde kandidaat moet alle voorwaarden "
            "halen. De uitkomst is 0 van 6 bewezen voorwaarden.",
            "",
            "| Gate | Eis | Status | Bewijs |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["revolutionary_v2_gate"]["conditions"]:
        lines.append(
            f"| {row['id']} | {row['requirement']} | `{row['status']}` | {row['evidence']} |"
        )

    lines.extend(["", "## Afgeleide boekhouding", ""])
    for row in payload["derived_accounting"]:
        lines.append(
            f"- **{row['derivation']}:** {row['calculation'].rstrip('.')}. Grens: {row['boundary']}"
        )

    lines.extend(["", "## Subjectieve inferentie", ""])
    lines.extend(f"- {item}" for item in payload["subjective_inference"])

    novelty = payload["novelty_status"]
    lines.extend(
        [
            "",
            "## Novelty-status",
            "",
            "De onafhankelijke audit vindt **geen verdedigbare brede nieuwheidsclaim**. "
            f"CU1 en CU3 zijn hoogstens `possibly novel intersection`: {novelty['qualification']} "
            f"De gezamenlijke optimizer is {novelty['joint_optimizer']}; de kernel/layoutclaim "
            f"is {novelty['custom_kernel_layout']}.",
            "",
            "## Exacte volgende actie",
            "",
            "1. Bevries dit CRAFT-pakket en markeer de registry `closed_no_eureka`.",
            "2. Download of test DeepSeek V4 Flash niet vanuit deze onderzoekslijn.",
            "3. Bouw geen packed kernel voor de gefaalde kandidaten en claim geen snelheid "
            "uit projected bytes of componentmicrobenchmarks.",
            "4. Start alleen opnieuw via een nieuw registry-item met een mechanistisch "
            "onafhankelijke hypothese en vooraf vastgelegde oracle-, full-depth-, runtime- "
            "en tweede-modelgates.",
            "",
            "## Beperkingen",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(
        [
            "",
            "Eindstatus: `closed_no_eureka`. Sluitingsgrond: de onderzochte hypotheses "
            "zijn overtuigend gefalsificeerd of volgens preregistratie dependency-geblokkeerd.",
            "",
        ]
    )
    return "\n".join(lines)
