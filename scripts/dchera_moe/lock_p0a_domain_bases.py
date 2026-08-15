from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from moe_lab.reporting import ROOT


DOMAINS = ("general", "code", "math", "multilingual", "instruction")
CAPACITY = 4_280
PREREG = ROOT / "reports/dchera_moe/P0A_DOMAIN_CACHE_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/dchera_moe/p0a_domain_base_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not PREREG.is_file():
        raise FileNotFoundError(PREREG)
    bases = {}
    boundaries = {}
    for domain in DOMAINS:
        candidates = []
        for layer in range(48):
            path = ROOT / f"reports/hera_moe/p0_route_layers/layer_{layer:02d}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            for expert in range(128):
                candidates.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "router_weight_squared_sum": report["domains"][domain][
                            "router_weight_squared_sum"
                        ][expert],
                        "count": report["domains"][domain]["counts"][expert],
                    }
                )
        ordered = sorted(
            candidates,
            key=lambda row: (
                -row["router_weight_squared_sum"],
                -row["count"],
                row["layer"],
                row["expert"],
            ),
        )
        bases[domain] = ordered[:CAPACITY]
        boundaries[domain] = {
            "minimum_selected_router_weight_squared_sum": ordered[CAPACITY - 1][
                "router_weight_squared_sum"
            ],
            "maximum_rejected_router_weight_squared_sum": ordered[CAPACITY][
                "router_weight_squared_sum"
            ],
            "minimum_selected_count": min(row["count"] for row in ordered[:CAPACITY]),
        }
    payload = {
        "kind": "dchera_moe_p0a_domain_base_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_experts_each_domain": CAPACITY,
        "selection_rule": (
            "per-domain descending HERA-training router_weight_squared_sum, "
            "then count, layer, expert"
        ),
        "preregistration_sha256": sha256(PREREG),
        "opened_validation_routes_used_for_selection": False,
        "validation_cache_outcomes_computed_before_lock": False,
        "bases": bases,
        "boundaries": boundaries,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                domain: {
                    "experts": len(bases[domain]),
                    **boundaries[domain],
                }
                for domain in DOMAINS
            },
            indent=2,
        )
    )
