from __future__ import annotations
import json, re, traceback
from common import REPO, require_model_dir, utc_now, write_json_atomic

OUT=REPO/"pro_research"/"results"/"s100_phase20s"/"S100_PHASE20S_RECLASSIFY.json"

def main():
    payload={"kind":"s100_phase20s_reclassify","status":"started","started_utc":utc_now()}
    try:
        model=require_model_dir()
        cfg=json.loads((model/"config.json").read_text(encoding="utf-8"))
        p20=json.loads((REPO/"pro_research"/"results"/"s100_phase20a_identity.json")
                       .read_text(encoding="utf-8"))
        audit=p20["consumption_audit"]
        unknown=list(audit.get("UNKNOWN_UNUSED_WEIGHTS") or [])
        expected_layers=[i for i,t in enumerate(cfg["layers_block_type"]) if t=="attention"]
        expected=[]
        for i in expected_layers:
            expected += [
                f"backbone.layers.{i}.mixer.k_scale",
                f"backbone.layers.{i}.mixer.v_scale",
            ]

        # Require explicit KV quantization metadata in either config spelling.
        q=cfg.get("quantization_config") or {}
        kv_scheme=q.get("kv_cache_scheme")
        kv_algo=q.get("kv_cache_quant_algo")
        hfq=None
        if (model/"hf_quant_config.json").exists():
            hfq=json.loads((model/"hf_quant_config.json").read_text(encoding="utf-8"))
            q2=(hfq.get("quantization") or {})
            kv_algo=kv_algo or q2.get("kv_cache_quant_algo")

        kv_declared=bool(
            kv_scheme is not None
            or (isinstance(kv_algo,str) and "FP8" in kv_algo.upper())
        )
        exact12=sorted(unknown)==sorted(expected) and len(expected)==12
        no_missing=int(audit.get("expected_but_missing_count") or 0)==0
        gate=bool(exact12 and kv_declared and no_missing)

        payload.update({
            "status":"measured",
            "attention_layers":expected_layers,
            "unknown_original":unknown,
            "expected_kv_serving_metadata":expected,
            "kv_cache_declared":kv_declared,
            "kv_cache_scheme":kv_scheme,
            "kv_cache_quant_algo":kv_algo,
            "exact_12_only":exact12,
            "expected_but_missing_count":audit.get("expected_but_missing_count"),
            "intentional_serving_metadata":expected if gate else [],
            "TARGET_MATH_CONSUMPTION_GREEN":gate,
            "remaining_unknown_target_math":[] if gate else unknown,
            "reason":(
                "k_scale/v_scale classified as optional FP8 KV-cache serving "
                "metadata; Phase20B target math must use fp8_kv=False unless "
                "FP8-KV fidelity is separately green"
                if gate else
                "fail-closed: unknown set or quantization metadata does not match"
            ),
            "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
            "error":{"type":type(exc).__name__,"message":str(exc),
                     "traceback":traceback.format_exc()},
            "completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":
    raise SystemExit(main())
