from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from moe_lab.reporting import ROOT


CAPACITY = 4280
OUTPUT = ROOT / "reports/dhera_moe/p0_base_lock.json"
HERA_RESULT = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    candidates = []
    for layer in range(48):
        report = json.loads((ROOT / f"reports/hera_moe/p0_route_layers/layer_{layer:02d}.json").read_text(encoding="utf-8"))
        for expert in range(128):
            squared_mass = sum(report["domains"][domain]["router_weight_squared_sum"][expert] for domain in report["domains"])
            count = sum(report["domains"][domain]["counts"][expert] for domain in report["domains"])
            candidates.append({"layer": layer, "expert": expert, "router_weight_squared_sum": squared_mass, "count": count})
    ordered = sorted(candidates, key=lambda row: (-row["router_weight_squared_sum"], -row["count"], row["layer"], row["expert"]))
    selected = ordered[:CAPACITY]
    payload = {
        "kind": "dhera_moe_p0_base_lock", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "source_hera_result_sha256": sha256(HERA_RESULT), "capacity_experts": CAPACITY,
        "selection_rule": "descending aggregate router_weight_squared_sum, then count, layer, expert",
        "selected": selected,
        "minimum_selected_count": min(row["count"] for row in selected),
        "maximum_rejected_router_weight_squared_sum": ordered[CAPACITY]["router_weight_squared_sum"],
        "minimum_selected_router_weight_squared_sum": selected[-1]["router_weight_squared_sum"],
        "validation_routes_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("capacity_experts", "minimum_selected_count", "minimum_selected_router_weight_squared_sum", "maximum_rejected_router_weight_squared_sum")}, indent=2))

