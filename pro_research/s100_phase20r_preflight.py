from __future__ import annotations

import json
import traceback

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase20a_identity import _schema_payload

OUT = REPO / "pro_research" / "results" / "s100_phase20r" / "S100_PHASE20R_PREFLIGHT.json"
HARD_SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"


def main():
    payload = {"kind": "s100_phase20r_preflight", "status": "started", "started_utc": utc_now()}
    try:
        model_dir = require_model_dir()
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if model_dir.name != HARD_SNAPSHOT:
            raise RuntimeError(f"wrong snapshot: {model_dir.name}")
        if cfg.get("architectures") != ["NemotronHForCausalLM"] or cfg.get("model_type") != "nemotron_h":
            raise RuntimeError(f"wrong architecture: {cfg.get('architectures')} / {cfg.get('model_type')}")
        if len(cfg.get("layers_block_type") or []) != 52:
            raise RuntimeError("expected exactly 52 target layers")
        _, audit = _schema_payload(model_dir)
        unknown = sorted(audit.get("UNKNOWN_UNUSED_WEIGHTS") or [])
        kv_unknown = sorted(x for x in unknown if x.endswith(".k_scale") or x.endswith(".v_scale"))
        other_unknown = sorted(set(unknown) - set(kv_unknown))
        layers = sorted(int(x.split(".")[2]) for x in kv_unknown)
        expected_layers = sorted(i for i, x in enumerate(cfg["layers_block_type"]) if x == "attention")
        green = bool(
            len(kv_unknown) == 12
            and not other_unknown
            and sorted(set(layers)) == expected_layers
            and len(expected_layers) == 6
            and not (audit.get("EXPECTED_BUT_MISSING_WEIGHTS") or [])
        )
        payload.update({
            "status": "measured",
            "model_dir": str(model_dir),
            "snapshot": model_dir.name,
            "attention_layers": expected_layers,
            "unknown_before_repair": unknown,
            "kv_scale_unknown": kv_unknown,
            "other_unknown": other_unknown,
            "expected_but_missing": audit.get("EXPECTED_BUT_MISSING_WEIGHTS") or [],
            "PREFLIGHT_GREEN": green,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "measured" and payload.get("PREFLIGHT_GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
