from __future__ import annotations

import json
import traceback

from common import REPO, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning15_common import (
    RESULTS, OLD_TRACE, OLD_TRACE_META,
    ensure_results, identity, model_signature,
)

OUT = RESULTS / "S100_LIGHTNING15_IDENTITY_AUDIT.json"

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning15_identity_audit",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        ident = identity()
        inherited = {
            "trace_exists": OLD_TRACE.exists(),
            "metadata_exists": OLD_TRACE_META.exists(),
            "metadata_has_model_signature": False,
            "baseline_control": None,
            "quarantined": True,
        }
        if OLD_TRACE.exists() and OLD_TRACE_META.exists():
            metadata = json.loads(
                OLD_TRACE_META.read_text(encoding="utf-8")
            )
            inherited["metadata_status"] = metadata.get("status")
            inherited["metadata_model_signature"] = metadata.get(
                "model_signature"
            )
            inherited["metadata_has_model_signature"] = (
                metadata.get("model_signature") is not None
            )
            inherited["signature_matches_lightning"] = (
                metadata.get("model_signature")
                == model_signature(ident)
            )
            # Run an explicit baseline control where possible. This result is
            # diagnostic only and can never rehabilitate unsigned metadata.
            try:
                from s100_phase5_quality import evaluate
                bundle = build()
                result = evaluate(bundle, "validation")
                inherited["baseline_control"] = {
                    "status": "measured",
                    "summary": result.get("summary"),
                    "strict_pass": result.get("strict_pass"),
                    "official_pass": result.get("official_pass"),
                }
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception as exc:
                inherited["baseline_control"] = {
                    "status": "technical_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        payload.update({
            "status": "measured",
            "identity": ident,
            "model_signature": model_signature(ident),
            "inherited_trace": inherited,
            "LIGHTNING_IDENTITY_GREEN": True,
            "INHERITED_TRACE_QUARANTINED": True,
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
