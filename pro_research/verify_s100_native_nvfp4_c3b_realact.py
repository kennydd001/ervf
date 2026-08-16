"""Independent, GPU-free consistency verifier for C3B real-activation result."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_REAL_ACTIVATION.json"
MODEL_DIR = REPO / "models" / "nemotron_3_5_lightning_v35"
INDEX = MODEL_DIR / "model.safetensors.index.json"

TH = {"normalized_rmse_max": 0.080, "cosine_min": 0.9950,
      "normalized_max_abs_error_max": 0.200, "lm_top1_retention_min": 0.90,
      "lm_native_top1_in_ervf_top5_min": 0.97, "cold_working_set_over_l2_min": 4.0,
      "M8_over_M1_max": 1.20, "static_margin": 1.10}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def headers():
    idx = json.loads(INDEX.read_text(encoding="utf-8")); wm = idx["weight_map"]; hs = {}
    for shard in sorted(set(wm.values())):
        p = MODEL_DIR / shard
        with p.open("rb") as fh:
            n = int.from_bytes(fh.read(8), "little"); hs[shard] = json.loads(fh.read(n))
    return wm, hs


def raw_tensor(name, wm, hs):
    shard = wm[name]; rec = hs[shard][name]; a, b = map(int, rec["data_offsets"])
    p = MODEL_DIR / shard
    with p.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little"); fh.seek(8+n+a); raw = fh.read(b-a)
    return raw


def flat_metrics(actual, ref):
    aa = [float(x) for row in actual for x in row]; bb = [float(x) for row in ref for x in row]
    if len(aa) != len(bb) or not aa:
        return None
    mse = math.fsum((a-b)**2 for a,b in zip(aa,bb))/len(aa)
    rmse = math.sqrt(mse); rrms = math.sqrt(math.fsum(b*b for b in bb)/len(bb))
    dot = math.fsum(a*b for a,b in zip(aa,bb)); an = math.sqrt(math.fsum(a*a for a in aa)); bn = math.sqrt(math.fsum(b*b for b in bb))
    cos = dot/(an*bn) if an and bn else (1.0 if an == bn else 0.0)
    ma = max(abs(a-b) for a,b in zip(aa,bb)); rmax = max(abs(b) for b in bb)
    return {"normalized_rmse": rmse/max(rrms,1e-12), "cosine": cos,
            "normalized_max_abs_error": ma/max(rmax,1e-12)}


def main() -> int:
    if not RESULT.exists():
        print(f"FAIL: missing {RESULT}"); return 2
    d = json.loads(RESULT.read_text(encoding="utf-8")); failures = []
    for k,v in TH.items():
        got = (d.get("thresholds") or {}).get(k)
        if got is None or not math.isclose(float(got), float(v), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"threshold drift {k}: {got} != {v}")
    cap = d.get("capture_manifest") or {}
    if cap.get("status") != "captured": failures.append("capture manifest not captured")
    for name, rec in (cap.get("arrays") or {}).items():
        p = REPO / rec["path"]
        if not p.exists() or sha_file(p) != rec.get("sha256"):
            failures.append(f"capture hash mismatch {name}")
    wm, hs = headers()
    for f in d.get("families") or []:
        c = f.get("checkpoint") or {}; s = f.get("selected") or {}; label=f.get("label")
        for key, field in (("weight","weight_sha256"),("scale","scale_sha256"),("global","global_sha256")):
            try: fresh = sha(raw_tensor(s[key], wm, hs))
            except Exception as exc:
                failures.append(f"{label} {key} reread failed: {exc}"); continue
            if fresh != c.get(field): failures.append(f"{label} {key} checkpoint SHA mismatch")
        for arm in ("dynamic", "static_1p10"):
            smp = (f.get("quality_samples") or {}).get(arm) or {}
            fresh = flat_metrics(smp.get("actual") or [], smp.get("reference") or [])
            recm = (((f.get("quality") or {}).get(arm) or {}).get("metrics") or {})
            if fresh is None:
                failures.append(f"{label}/{arm}: missing quality samples"); continue
            for key in ("normalized_rmse","cosine","normalized_max_abs_error"):
                if not math.isclose(float(fresh[key]), float(recm.get(key, math.nan)), rel_tol=2e-5, abs_tol=2e-6):
                    failures.append(f"{label}/{arm}: metric self-consistency failure {key}")
    lm = d.get("lm_head_quality") or {}
    for arm in ("dynamic", "static_1p10"):
        r = lm.get(arm) or {}; pred = r.get("native_top1_ids") or []; exact = r.get("exact_top1_ids") or []; ex5 = r.get("exact_top5_ids") or []
        if not pred or len(pred) != len(exact) or len(pred) != len(ex5):
            failures.append(f"lm/{arm}: malformed token vectors"); continue
        top1 = sum(int(a)==int(b) for a,b in zip(pred,exact))/len(pred)
        in5 = sum(int(pred[i]) in set(int(z) for z in ex5[i]) for i in range(len(pred)))/len(pred)
        if not math.isclose(top1, float(r.get("top1_retention", -1)), abs_tol=1e-7): failures.append(f"lm/{arm}: top1 retention mismatch")
        if not math.isclose(in5, float(r.get("native_top1_in_ervf_top5", -1)), abs_tol=1e-7): failures.append(f"lm/{arm}: top5 containment mismatch")
    gates = d.get("gates") or {}
    required_base = ("C3B_G1_C3A_v2_parent_green","C3B_G2_capture_hashes","C3B_G3_quant_layout_native_preflight_exact","C3B_G4_all_native_outputs_finite","C3B_P1_cold_rotation_ge_4x_L2")
    for k in required_base:
        if gates.get(k) is not True: failures.append(f"required base gate not green: {k}={gates.get(k)}")
    selected = ((d.get("summary") or {}).get("selected_candidate_arm"))
    if d.get("status") == "real_activation_native_candidate":
        if selected not in {"dynamic","static_1p10"}: failures.append("candidate status without selected arm")
        if selected == "dynamic" and not all(gates.get(k) is True for k in ("C3B_G5_DYNAMIC_LOCAL","C3B_G7_DYNAMIC_LM","C3B_P2_DYNAMIC_M8_geometry")):
            failures.append("dynamic selected but its gates are not all green")
        if selected == "static_1p10" and not all(gates.get(k) is True for k in ("C3B_G6_STATIC_LOCAL","C3B_G8_STATIC_LM","C3B_P3_STATIC_M8_geometry")):
            failures.append("static selected but its gates are not all green")
    print(json.dumps({"status":"PASS" if not failures else "FAIL", "result_status":d.get("status"),
                      "selected_candidate_arm":selected, "failures":failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
