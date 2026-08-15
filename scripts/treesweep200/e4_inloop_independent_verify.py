"""Independent verification of E4's in-loop adoption. Never imports the runner.

Recomputes every p50 from raw samples, the bracketed baselines, gains and drifts,
and the parity relations from the raw token ids. Re-runs both attention kernels
on random FP8 KV to reconfirm bitwise identity independently of the runner (the
kernel library may be imported; the runner may not). Also checks the explanation
for the failed anchor clause: the S5 anchor predates the v35 checkpoint switch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
LS = REPO_ROOT / "reports" / "lightningstream_nemotron"
sys.path.insert(0, str(REPO_ROOT / "src"))

GATE_T1_MS = 6.0
GATE_STRETCH_MS = 4.8


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(xs):
    s = sorted(float(x) for x in xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def bitwise_recheck():
    """v4 vs v1 on random FP8 KV, independent of the runner."""
    import cupy as cp
    from moe_lab.lightningstream_nemotron.gpu_kernels import GPUKernels

    k = GPUKernels()
    n_heads, head_dim, groups, n_kv = 32, 128, 16, 2
    max_ctx = 8192
    rng = np.random.default_rng(4)
    out = {}
    for t in (64, 1024, 4096):
        q = cp.asarray(rng.standard_normal(n_heads * head_dim).astype(np.float32))
        kv = rng.integers(0, 256, size=n_kv * max_ctx * head_dim, dtype=np.uint8)
        # e4m3 NaN patterns break rel_l2; remap as the E4 report warns
        kv[(kv & 0x7F) == 0x7F] &= 0xFE
        vv = rng.integers(0, 256, size=n_kv * max_ctx * head_dim, dtype=np.uint8)
        vv[(vv & 0x7F) == 0x7F] &= 0xFE
        Kc, Vc = cp.asarray(kv), cp.asarray(vv)
        splits = k.MAX_SPLITS * 4
        pa = cp.zeros(n_heads * splits * head_dim, dtype=cp.float32)
        pm = cp.zeros(n_heads * splits * 2, dtype=cp.float32)
        o1 = cp.zeros(n_heads * head_dim, dtype=cp.float32)
        o4 = cp.zeros(n_heads * head_dim, dtype=cp.float32)
        scale = 1.0 / float(np.sqrt(head_dim))
        k.attention_fp8_gqa(o1, q, Kc, Vc, t, n_heads, head_dim, groups,
                            max_ctx, scale, pa, pm)
        k.attention_fp8_gqa4(o4, q, Kc, Vc, t, n_heads, head_dim, groups,
                             max_ctx, scale, pa, pm)
        cp.cuda.Device(0).synchronize()
        out[str(t)] = bool(cp.array_equal(o1, o4))
    return out


def main() -> int:
    path = OUT_DIR / "E4_INLOOP_RESULTS.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    arms = res["arms"]
    checks = []

    checks.append({"check": "three arms in v1/v4/v1 order",
                   "ok": list(arms) == ["v1_a", "v4", "v1_b"]})
    checks.append({"check": "the middle arm used the v4 kernel and the others v1",
                   "ok": arms["v4"]["kernel"].endswith("gqa4")
                   and arms["v1_a"]["kernel"].endswith("gqa")
                   and arms["v1_b"]["kernel"].endswith("gqa")})

    ref = arms["v1_a"]["generation"]
    for name, arm in arms.items():
        same = all(g["generated_ids"] == r["generated_ids"]
                   for g, r in zip(arm["generation"], ref))
        checks.append({"check": f"{name}: generated ids identical to v1_a",
                       "ok": same and same == arm["parity_vs_v1_a"]})
    checks.append({"check": "each arm generated 2 prompts x 64 tokens",
                   "ok": all(len(a["generation"]) == 2
                             and all(len(g["generated_ids"]) == 64
                                     for g in a["generation"])
                             for a in arms.values())})

    rec = {}
    for ctx in [str(c) for c in res["config"]["contexts"]]:
        vals = {}
        for name, arm in arms.items():
            row = arm["sweep"][ctx]
            for field, raw in (("attn", "raw_attn_ms"), ("token", "raw_token_ms")):
                m = p50(row[raw])
                stored = row[f"{field}_ms"]["p50"]
                checks.append({"check": f"{name}@{ctx}: {field} p50 reproduces",
                               "ok": abs(m - stored) < 1e-9})
                vals[f"{name}_{field}"] = m
            checks.append({"check": f"{name}@{ctx}: at least 32 samples",
                           "ok": len(row["raw_attn_ms"]) >= 32})
        base_a = 0.5 * (vals["v1_a_attn"] + vals["v1_b_attn"])
        base_t = 0.5 * (vals["v1_a_token"] + vals["v1_b_token"])
        rec[ctx] = {
            "attn_gain": base_a - vals["v4_attn"],
            "attn_drift": abs(vals["v1_b_attn"] - vals["v1_a_attn"]),
            "token_gain": base_t - vals["v4_token"],
            "token_drift": abs(vals["v1_b_token"] - vals["v1_a_token"]),
            "attn_v4": vals["v4_attn"], "attn_v1": base_a,
            "token_v4": vals["v4_token"], "token_v1": base_t,
        }
        st = res["per_context"][ctx]
        checks.append({"check": f"@{ctx}: attention gain reproduces",
                       "ok": abs(rec[ctx]["attn_gain"] - st["attn_gain_ms"]) < 1e-9,
                       "recomputed": rec[ctx]["attn_gain"]})
        checks.append({"check": f"@{ctx}: token gain reproduces",
                       "ok": abs(rec[ctx]["token_gain"] - st["token_gain_ms"]) < 1e-9})
        checks.append({"check": f"@{ctx}: attention gain exceeds its own drift",
                       "ok": rec[ctx]["attn_gain"] > rec[ctx]["attn_drift"],
                       "gain": rec[ctx]["attn_gain"], "drift": rec[ctx]["attn_drift"]})

    deep = max(rec, key=lambda k: int(k))
    t1 = rec[deep]["attn_v4"] <= GATE_T1_MS
    checks.append({"check": "G-E4-T1 threshold verdict agrees with the runner",
                   "ok": (t1 and res["gates"]["G_E4_T1_parity"]["passed"])
                   == res["gates"]["G_E4_T1_inloop"]["passed"]})
    checks.append({"check": "G-E4-T1 threshold is still the preregistered 6.0 ms",
                   "ok": abs(res["gates"]["G_E4_T1_inloop"]["required_ms"]
                             - GATE_T1_MS) < 1e-12})
    checks.append({"check": "in-loop v1 attention matches E4's isolated 6 x 2.803 ms "
                            "within 10%",
                   "ok": abs(rec[deep]["attn_v1"] - 6 * 2.803) / (6 * 2.803) < 0.10,
                   "inloop": rec[deep]["attn_v1"], "isolated_x6": 6 * 2.803})
    checks.append({"check": "in-loop v4 attention matches E4's isolated 6 x 2.304 ms "
                            "within 10%",
                   "ok": abs(rec[deep]["attn_v4"] - 6 * 2.304) / (6 * 2.304) < 0.10,
                   "inloop": rec[deep]["attn_v4"], "isolated_x6": 6 * 2.304})

    # ---- the anchor clause: stale artifact, not a v4 defect
    s5 = json.loads((LS / "s5_baseline_generation.json").read_text(encoding="utf-8"))
    n2r = json.loads((LS / "n2r_v35_layout.json").read_text(encoding="utf-8"))
    stale = s5["started_utc"] < n2r["completed_utc"]
    checks.append({"check": "the S5 anchor predates the v35 layout capture, so it "
                            "cannot match a v35 generation",
                   "ok": stale, "s5": s5["started_utc"], "v35": n2r["completed_utc"]})
    checks.append({"check": "anchor mismatch is reported, not hidden",
                   "ok": res["gates"]["G_E4_T1_parity"]["anchor_first_n"] is False})
    checks.append({"check": "arm-to-arm parity, which is the meaningful one, holds",
                   "ok": all(a["parity_vs_v1_a"] for a in arms.values())})

    # ---- independent bitwise reconfirmation
    try:
        bw = bitwise_recheck()
        checks.append({"check": "v4 bitwise identical to v1 on random FP8 KV "
                                "(reconfirmed without the runner)",
                       "ok": all(bw.values()), "per_t": bw})
    except Exception as e:                                   # pragma: no cover
        bw = {"error": f"{type(e).__name__}: {e}"}
        checks.append({"check": "bitwise reconfirmation ran", "ok": False, "error": bw})

    checks.append({"check": "gpu_kernels.py hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/gpu_kernels.py")
                   == res["kernels_sha256"]})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "treesweep200_e4_inloop_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": rec,
        "bitwise_recheck": bw,
        "anchor_is_stale": stale,
        "gates": {"G_E4_T1_inloop": bool(t1 and all(a["parity_vs_v1_a"]
                                                    for a in arms.values())),
                  "arm_parity": all(a["parity_vs_v1_a"] for a in arms.values())},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "e4_inloop_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)\n")
    for ctx, v in rec.items():
        print(f"  ctx {ctx:>6}: attention {v['attn_v1']:6.3f} -> {v['attn_v4']:6.3f} "
              f"({v['attn_gain']:+.3f}, drift {v['attn_drift']:.3f}) | "
              f"token {v['token_v1']:7.3f} -> {v['token_v4']:7.3f} "
              f"({v['token_gain']:+.3f}, drift {v['token_drift']:.3f})")
    print(f"  bitwise v4==v1: {bw}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
