"""Independent verification of S11. Imports nothing from the runner or runtime.

Recomputes the slot sizes straight from the checkpoint's safetensors headers
(not from constants copied out of runtime.py), re-derives every p50 and tok/s
from the raw millisecond samples with its own percentile, redoes the identity
comparison of the generated token ids, and re-evaluates all three gates.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
ADOPT_MARGIN = 0.03
EXPECTED_SAMPLES = 16
MOE_LAYERS_EXPECTED = 23


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(xs):
    """Median with the same convention numpy uses for even-length input."""
    s = sorted(float(x) for x in xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def tensor_bytes(model_dir: Path, names: list[str]) -> dict[str, int]:
    """Byte length of each named tensor, read from the shard headers."""
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    wanted = {n: weight_map[n] for n in names}
    out: dict[str, int] = {}
    for shard in sorted(set(wanted.values())):
        with (model_dir / shard).open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen).decode("utf-8"))
        for name, sh in wanted.items():
            if sh == shard:
                a, b = header[name]["data_offsets"]
                out[name] = b - a
    return out


def main() -> int:
    path = OUT_DIR / "s11_cache_mode_ab.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    checks: list[dict] = []
    arms = res["arms"]

    # ------------------------------------------------ slot size from the model
    model_dir = REPO_ROOT / "models" / res["model_dir"]
    pre = "backbone.layers.1.mixer.experts.0"
    sizes = tensor_bytes(model_dir, [f"{pre}.up_proj.weight", f"{pre}.up_proj.weight_scale",
                                     f"{pre}.down_proj.weight", f"{pre}.down_proj.weight_scale"])
    up_half = sizes[f"{pre}.up_proj.weight"] + sizes[f"{pre}.up_proj.weight_scale"]
    down_half = sizes[f"{pre}.down_proj.weight"] + sizes[f"{pre}.down_proj.weight_scale"]
    checks.append({"check": "down half is exactly the same size as the up half",
                   "ok": up_half == down_half,
                   "up_half": up_half, "down_half": down_half})

    for name, arm in arms.items():
        slot = up_half if arm["mode"] == "up_only" else up_half + down_half
        expect = slot * arm["capacity"] * MOE_LAYERS_EXPECTED
        checks.append({"check": f"{name}: cache bytes = slot x capacity x 23 MoE layers",
                       "ok": expect == arm["cache_bytes"],
                       "recomputed": expect, "stored": arm["cache_bytes"]})

    byte_sets = {arm["cache_bytes"] for arm in arms.values()}
    checks.append({"check": "all arms hold exactly the same number of cache bytes",
                   "ok": len(byte_sets) == 1, "bytes": sorted(byte_sets)})

    # --------------------------------------------------------------- identity
    ref = arms["A1"]["generation_token_ids"]
    checks.append({"check": "reference generation is 2 prompts x 64 tokens",
                   "ok": len(ref) == 2 and all(len(g) == 64 for g in ref)})
    for name, arm in arms.items():
        same = arm["generation_token_ids"] == ref
        checks.append({"check": f"{name}: generated token ids identical to A1",
                       "ok": same == arm["identical_to_A1"] and (name == "A1" or same),
                       "recomputed_identical": same,
                       "stored_identical": arm["identical_to_A1"]})

    # ------------------------------------------------------- timing recompute
    contexts = [str(c) for c in res["config"]["contexts"]]
    tok_s = {}
    for name, arm in arms.items():
        tok_s[name] = {}
        for ctx in contexts:
            row = arm["context_sweep"][ctx]
            raw = row["raw_ms"]
            checks.append({"check": f"{name}@{ctx}: {EXPECTED_SAMPLES} timing samples",
                           "ok": len(raw) == EXPECTED_SAMPLES, "n": len(raw)})
            r50 = p50(raw)
            checks.append({"check": f"{name}@{ctx}: p50 reproduces from raw samples",
                           "ok": abs(r50 - row["ms"]["p50"]) < 1e-9,
                           "recomputed": r50, "stored": row["ms"]["p50"]})
            t = 1000.0 / r50
            checks.append({"check": f"{name}@{ctx}: tok/s reproduces from p50",
                           "ok": abs(t - row["tok_s_p50"]) < 1e-6,
                           "recomputed": t, "stored": row["tok_s_p50"]})
            tok_s[name][ctx] = t

    # ------------------------------------------------------------------ gates
    deep, shallow = contexts[-1], contexts[0]
    a1, b, a2 = tok_s["A1"][deep], tok_s["B"][deep], tok_s["A2"][deep]
    gain = (b - a1) / a1
    drift = abs(a2 - a1)
    effect = abs(b - a1)
    ctx0_ok = tok_s["B"][shallow] >= tok_s["A1"][shallow]

    identity_pass = arms["B"]["generation_token_ids"] == ref
    adopt_pass = gain >= ADOPT_MARGIN and ctx0_ok
    conclusive = drift < effect

    g = res["gates"]
    checks.append({"check": "G-S11-C1 verdict agrees with the runner",
                   "ok": identity_pass == g["G_S11_C1_bit_identical"]["passed"]})
    checks.append({"check": "G-S11-P1 verdict agrees with the runner",
                   "ok": adopt_pass == g["G_S11_P1_adopt"]["passed"],
                   "recomputed_gain": gain})
    checks.append({"check": "G-S11-D1 conclusiveness agrees with the runner",
                   "ok": conclusive == g["G_S11_D1_drift"]["conclusive"],
                   "drift": drift, "effect": effect})
    checks.append({"check": "adoption margin is still the preregistered 3%",
                   "ok": abs(g["G_S11_P1_adopt"]["required_relative_gain_at_deep"]
                             - ADOPT_MARGIN) < 1e-12})
    checks.append({"check": "runtime.py on disk still hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                   == res["runtime_sha256"]})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s11_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {
            "up_half_bytes": up_half, "down_half_bytes": down_half,
            "tok_s": tok_s, "relative_gain_at_deep": gain,
            "drift_tok_s": drift, "effect_tok_s": b - a1,
            "hit_rate": {n: a["sweep_cache"]["hit_rate"] for n, a in arms.items()},
        },
        "gates": {
            "G_S11_C1_bit_identical": identity_pass,
            "G_S11_P1_adopt": adopt_pass,
            "G_S11_D1_conclusive": conclusive,
        },
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s11_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"\ndeep ctx {deep}: A1 {a1:.3f}  B {b:.3f}  A2 {a2:.3f} tok/s "
          f"(gain {gain * 100:+.2f}%, drift {drift:.3f})")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
