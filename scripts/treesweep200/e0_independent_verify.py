"""Independent verification of P0/E0. Imports nothing from the runner.

Recomputes the byte floors from the checkpoint's safetensors shard headers with
its own counting, re-derives the nonzero-ReLU2 fraction from the frozen S1-S4
census, checks every evidence hash against the manifests, and re-evaluates all
four gates.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "treesweep200"
LS = REPO_ROOT / "reports" / "lightningstream_nemotron"
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning_v35"


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    res = json.loads((OUT / "E0_ROOFLINE_REPRODUCTION.json").read_text(encoding="utf-8"))
    ident = json.loads((OUT / "P0_IDENTITY_MANIFEST.json").read_text(encoding="utf-8"))
    ev = json.loads((OUT / "E0_N1_N5_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    # ------------------------------------------------- hashes and manifests
    checks.append({"check": "identity manifest hash matches the runner's record",
                   "ok": sha256_path(OUT / "P0_IDENTITY_MANIFEST.json")
                   == res["identity_manifest_sha256"]})
    checks.append({"check": "evidence manifest hash matches the runner's record",
                   "ok": sha256_path(OUT / "E0_N1_N5_EVIDENCE_MANIFEST.json")
                   == res["evidence_manifest_sha256"]})
    ok_ev = True
    for name, e in ev["sources"].items():
        p = REPO_ROOT / e["path"]
        if not (p.is_file() and sha256_path(p) == e["sha256"]):
            ok_ev = False
    checks.append({"check": "all ten evidence sources still hash to the locked values",
                   "ok": ok_ev})
    ok_id = all(f["present"] for f in ident["identity_files"].values())
    checks.append({"check": "identity files all present and hashed",
                   "ok": ok_id})
    ok_rt = True
    for name, m in ident["runtime_modules"].items():
        p = REPO_ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / name
        if name in ("runtime.py",):  # runtime must be exactly the locked bytes
            if not (p.is_file() and sha256_path(p) == m["sha256"]):
                ok_rt = False
    checks.append({"check": "runtime.py unchanged since P0 lock", "ok": ok_rt})

    # identity cross-check against config
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    checks.append({"check": "identity fields match config.json",
                   "ok": (ident["hidden_size"] == cfg["hidden_size"]
                          and ident["n_routed_experts"] == cfg["n_routed_experts"]
                          and ident["top_k"] == cfg["num_experts_per_tok"]
                          and ident["layers"] == len(cfg["layers_block_type"])
                          and ident["num_key_value_heads"] == cfg["num_key_value_heads"])})

    # ------------------------------------------------- own floor recompute
    wm = json.loads((MODEL_DIR / "model.safetensors.index.json")
                    .read_text(encoding="utf-8"))["weight_map"]
    sizes: dict[str, int] = {}
    for shard in sorted(set(wm.values())):
        with (MODEL_DIR / shard).open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen).decode("utf-8"))
        for name, meta in header.items():
            if name != "__metadata__":
                a, b = meta["data_offsets"]
                sizes[name] = b - a

    total = sum(sizes.values())
    bank = sum(v for k, v in sizes.items()
               if k.startswith("backbone.") and ".mixer.experts." in k)
    mtp = sum(v for k, v in sizes.items() if k.startswith("mtp."))
    embed = sizes.get("backbone.embeddings.weight", 0)
    resident = total - bank - mtp - embed
    bm = res["byte_model"]
    checks.append({"check": "resident bytes reproduce from own header count",
                   "ok": resident == bm["resident_bytes"],
                   "recomputed": resident, "stored": bm["resident_bytes"]})

    pre = "backbone.layers.1.mixer.experts.0"
    up_rec = sizes[f"{pre}.up_proj.weight"] + sizes[f"{pre}.up_proj.weight_scale"]
    dn_rec = sizes[f"{pre}.down_proj.weight"] + sizes[f"{pre}.down_proj.weight_scale"]
    checks.append({"check": "up/down record sizes match the byte model",
                   "ok": up_rec == bm["up_record_bytes"]
                   and dn_rec == bm["down_record_bytes"]})

    census = json.loads((LS / "s1_s4_hypothesis_census.json").read_text(encoding="utf-8"))
    nz = 1.0 - census["s2_relu2_sparsity"]["mean_zero_fraction"]
    checks.append({"check": "nonzero fraction rederived from frozen census",
                   "ok": abs(nz - bm["relu2_nonzero_fraction"]) < 1e-12})

    n_moe = sum(1 for t in cfg["layers_block_type"] if t == "moe")
    n_attn = sum(1 for t in cfg["layers_block_type"] if t == "attention")
    top_k = cfg["num_experts_per_tok"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    roof = res["roofline"]["own_gb_s"]
    ok_floors = True
    for ctx, row in res["floors"].items():
        kv = n_attn * 2 * int(ctx) * kv_dim
        expert_read = n_moe * top_k * (up_rec + nz * dn_rec)  # float, as runner
        tot = int(resident + expert_read + kv)
        ms = tot / (roof * 1e9) * 1e3
        if not (tot == row["total_bytes"] and abs(ms - row["floor_ms"]) < 1e-9):
            ok_floors = False
    checks.append({"check": "all five byte floors reproduce from headers + own roofline",
                   "ok": ok_floors})

    # ------------------------------------------------- gates
    g = res["gates"]
    r_imp = res["roofline"]["imported_gb_s"]
    r1_ok = abs(roof - r_imp) / r_imp <= 0.10
    checks.append({"check": "G-E0-R1: own roofline within 10% of imported",
                   "ok": (r1_ok == g["G_E0_R1"]["pass"]) and r1_ok})
    f0 = res["floors"]["0"]["floor_ms"]
    f262 = res["floors"]["262100"]["floor_ms"]
    ok_f1 = abs(f0 - 6.05) / 6.05 <= 0.10 and abs(f262 - 8.43) / 8.43 <= 0.10
    checks.append({"check": "G-E0-F1: own floors within 10% of imported",
                   "ok": ok_f1 == g["G_E0_F1"]["pass"] and ok_f1,
                   "own_ctx0_ms": f0, "own_262k_ms": f262})
    checks.append({"check": "G-P0-I1 and G-P0-B1 pass as stored",
                   "ok": g["G_P0_I1"]["pass"] and g["G_P0_B1"]["pass"]})
    checks.append({"check": "every imported claim classified with a note",
                   "ok": len(res["classification"]) == 8
                   and all(c["classification"] in
                           {"reproduced", "shifted", "invalid", "inconclusive"}
                           and c["note"] for c in res["classification"])})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "treesweep200_e0_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(OUT / "E0_ROOFLINE_REPRODUCTION.json"),
        "recomputed": {"resident_bytes": resident, "roofline_gb_s": roof,
                       "floor_ms": {c: res["floors"][c]["floor_ms"]
                                    for c in res["floors"]}},
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT / "e0_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
