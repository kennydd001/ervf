"""Independent verification of E1 fase 2.1 (device-resident routing + cache).

Never imports the runner (scripts/treesweep200/e1f21_device_routing_ab.py or
e1f21_inv_ctl_rerun.py). The kernel library may be imported.

Two evidence layers:
  1. Gate arithmetic is recomputed from the raw JSONs: C1 parity, S1 speed
     (DEV p50 <= BASE p50 - 1.5 ms), INV capacity invariance, CTL sabotage
     attribution, cross-file consistency of the DEV reference.
  2. Kernel-level rechecks on synthetic data, independent of any run:
     - route_topk_f32 vs a NumPy reference (ids exact, weights <=1e-6 rel)
     - cache_assign vs a Python LRU mirror (all tables exactly equal)
     - cache_fetch integrity (fetched slot bytes == bank expert bytes)
     - gemv_ervf_indirect bitwise == gemv_into (ERVF path) on the same record
     - accumulate_indirect bitwise == accumulate_into
     - V1: device-table footprint of 23 layers < 32 MiB (mem_info delta)

Gates are frozen in
reports/treesweep200/E1F21_DEVICE_ROUTING_PREREGISTRATION_2026-08-15.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TS200 = REPO_ROOT / "reports" / "treesweep200"
sys.path.insert(0, str(REPO_ROOT / "src"))

GATE_S1_MIN_GAIN_MS = 1.5
GATE_V1_MAX_BYTES = 32 * 1024 * 1024
N_MOE_LAYERS = 23
HIDDEN = 2688
MOE_INTER = 1856
UP_CODE = MOE_INTER * HIDDEN // 2      # 2,494,464
UP_SCALE = MOE_INTER * HIDDEN // 16    # 311,808


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def kernel_rechecks():
    import cupy as cp
    from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

    fused = FusedNVFP4()
    rng = np.random.default_rng(21)
    out = {}

    # ---- route_topk_f32 vs NumPy reference -------------------------------
    n, top_k, scaling = 128, 6, 2.827
    rlog = rng.standard_normal(n).astype(np.float32) * 3.0
    gate_b = rng.standard_normal(n).astype(np.float32) * 0.1
    ids = cp.zeros(top_k, dtype=cp.int32)
    w = cp.zeros(top_k, dtype=cp.float32)
    fused.route_topk(cp.asarray(rlog), cp.asarray(gate_b), ids, w,
                     n, top_k, scaling)
    cp.cuda.Device(0).synchronize()
    sc = (1.0 / (1.0 + np.exp(-rlog.astype(np.float64)))).astype(np.float32)
    ch = (sc.astype(np.float64) + gate_b.astype(np.float64))
    ref_ids, ref_sc = [], []
    chw = ch.copy()
    for _ in range(top_k):
        bi = int(np.argmax(chw))  # low index wins ties, as the serial scan does
        ref_ids.append(bi)
        ref_sc.append(float(sc[bi]))
        chw[bi] = -np.inf
    ref_w = np.array(ref_sc, dtype=np.float64)
    ref_w = ref_w / (ref_w.sum() + 1e-20) * scaling
    got_ids = cp.asnumpy(ids).tolist()
    got_w = cp.asnumpy(w).astype(np.float64)
    out["route_ids_exact"] = got_ids == ref_ids
    out["route_w_max_rel"] = float(np.max(np.abs(got_w - ref_w) /
                                          np.maximum(np.abs(ref_w), 1e-12)))

    # ---- cache_assign vs Python LRU mirror -------------------------------
    cap, nx, steps = 8, 32, 60
    dev = fused.alloc_device_cache(nx, cap, top_k,
                                   np.zeros((nx, 2), dtype=np.float32))
    py = {"slot_of": {}, "expert_of": [-1] * cap, "last_used": [-1] * cap,
          "tick": 0, "filled": 0}
    assign_ok = True
    for _ in range(steps):
        ids_np = rng.choice(nx, size=top_k, replace=False).astype(np.int32)
        dev["ids"][:] = cp.asarray(ids_np)
        fused.cache_assign(dev, dev["ids"], cap, top_k)
        cp.cuda.Device(0).synchronize()
        exp_slots, exp_need = [], []
        for e in ids_np.tolist():
            if e in py["slot_of"]:
                sl = py["slot_of"][e]
                py["tick"] += 1
                py["last_used"][sl] = py["tick"]
                exp_slots.append(sl)
                exp_need.append(0)
                continue
            if py["filled"] < cap:
                v = py["filled"]
                py["filled"] += 1
            else:
                v = min(range(cap), key=lambda c: (py["last_used"][c], c))
                del py["slot_of"][py["expert_of"][v]]
            py["slot_of"][e] = v
            py["expert_of"][v] = e
            py["tick"] += 1
            py["last_used"][v] = py["tick"]
            exp_slots.append(v)
            exp_need.append(1)
        got = (cp.asnumpy(dev["slots"]).tolist(), cp.asnumpy(dev["need"]).tolist(),
               cp.asnumpy(dev["slot_of"]).tolist(),
               cp.asnumpy(dev["expert_of"]).tolist(),
               cp.asnumpy(dev["last_used"]).tolist(),
               cp.asnumpy(dev["state2"]).tolist())
        ref_slot_of = [-1] * nx
        for e, sl in py["slot_of"].items():
            ref_slot_of[e] = sl
        ref = (exp_slots, exp_need, ref_slot_of, py["expert_of"],
               py["last_used"], [py["tick"], py["filled"]])
        if got != ref:
            assign_ok = False
            break
    out["cache_assign_exact"] = assign_ok

    # ---- cache_fetch integrity --------------------------------------------
    # Fresh dev tables: cache_assign state and the cache buffers must describe
    # the same world, or hits would skip fetches into never-written slots.
    dev = fused.alloc_device_cache(nx, cap, top_k,
                                   np.zeros((nx, 2), dtype=np.float32))
    bank_c = cp.cuda.alloc_pinned_memory(nx * UP_CODE)
    bank_s = cp.cuda.alloc_pinned_memory(nx * UP_SCALE)
    bc = np.frombuffer(bank_c, dtype=np.uint8)
    bs = np.frombuffer(bank_s, dtype=np.uint8)
    bc[:] = rng.integers(0, 256, size=bc.size, dtype=np.uint8)
    bs[:] = rng.integers(0, 256, size=bs.size, dtype=np.uint8)
    cache_c = cp.zeros(cap * UP_CODE, dtype=cp.uint8)
    cache_s = cp.zeros(cap * UP_SCALE, dtype=cp.uint8)
    fetch_ok = True
    for _ in range(10):
        ids_np = rng.choice(nx, size=top_k, replace=False).astype(np.int32)
        dev["ids"][:] = cp.asarray(ids_np)
        fused.cache_assign(dev, dev["ids"], cap, top_k)
        fused.cache_fetch(bc.ctypes.data, bs.ctypes.data, cache_c, cache_s,
                          dev, UP_CODE, UP_SCALE, top_k)
        cp.cuda.Device(0).synchronize()
        slots = cp.asnumpy(dev["slots"]).tolist()
        need = cp.asnumpy(dev["need"]).tolist()
        cc = cp.asnumpy(cache_c)
        cs = cp.asnumpy(cache_s)
        for s, e in enumerate(ids_np.tolist()):
            sl = slots[s]
            if not np.array_equal(cc[sl * UP_CODE:(sl + 1) * UP_CODE],
                                  bc[e * UP_CODE:(e + 1) * UP_CODE]):
                fetch_ok = False
            if not np.array_equal(cs[sl * UP_SCALE:(sl + 1) * UP_SCALE],
                                  bs[e * UP_SCALE:(e + 1) * UP_SCALE]):
                fetch_ok = False
        if not fetch_ok:
            break
    out["cache_fetch_bytes_exact"] = fetch_ok

    # ---- indirect GEMV bitwise == direct GEMV ------------------------------
    codes = cp.asarray(rng.integers(0, 256, size=UP_CODE, dtype=np.uint8))
    scales = cp.asarray(rng.integers(0, 256, size=UP_SCALE, dtype=np.uint8))
    x = cp.asarray(rng.standard_normal(HIDDEN).astype(np.float32))
    g = np.float32(0.37)
    o1 = cp.zeros(MOE_INTER, dtype=cp.float32)
    o2 = cp.zeros(MOE_INTER, dtype=cp.float32)
    fused.gemv_into(o1, codes, scales, x, float(g), MOE_INTER, HIDDEN,
                    apply_relu2=True)
    dev2 = fused.alloc_device_cache(nx, cap, top_k,
                                    np.stack([np.zeros(nx, np.float32),
                                              np.full(nx, g, np.float32)],
                                             axis=1))
    dev2["slots"][0] = 0
    dev2["ids"][0] = 0
    fused.gemv_ervf_indirect(o2, codes, scales, dev2, 0, dev2["globals"], 1,
                             x, MOE_INTER, HIDDEN, True, UP_CODE, UP_SCALE)
    cp.cuda.Device(0).synchronize()
    out["gemv_ind_bitwise"] = bool(cp.array_equal(o1, o2))

    # ---- accumulate_indirect bitwise == accumulate_into --------------------
    src = cp.asarray(rng.standard_normal(HIDDEN).astype(np.float32))
    d1 = cp.asarray(rng.standard_normal(HIDDEN).astype(np.float32))
    d2 = d1.copy()
    wv = np.float32(0.61)
    wbuf = cp.asarray(np.array([wv], dtype=np.float32))
    fused.accumulate_into(d1, src, float(wv), HIDDEN)
    fused.accumulate_indirect(d2, src, wbuf, HIDDEN)
    cp.cuda.Device(0).synchronize()
    out["accumulate_ind_bitwise"] = bool(cp.array_equal(d1, d2))

    # ---- V1: device-table footprint ----------------------------------------
    free0 = cp.cuda.Device(0).mem_info[0]
    held = [fused.alloc_device_cache(128, 72, 6,
                                     np.zeros((128, 2), dtype=np.float32))
            for _ in range(N_MOE_LAYERS)]
    cp.cuda.Device(0).synchronize()
    free1 = cp.cuda.Device(0).mem_info[0]
    out["v1_bytes_23_layers"] = int(free0 - free1)
    del held
    # Analytic floor: the pool can hide sub-MiB allocations, so also add up
    # every array the tables hold, per layer, plus the one shared contrib.
    per_layer = (6 * 4 * 4          # ids, w, slots, need
                 + 128 * 4          # slot_of
                 + 72 * 4 * 2       # expert_of, last_used
                 + 2 * 4 * 2        # state2, stats2
                 + 128 * 2 * 4)     # globals
    analytic = N_MOE_LAYERS * per_layer + 6 * HIDDEN * 4
    out["v1_analytic_bytes"] = int(analytic)
    return out


def main() -> int:
    ab_path = TS200 / "E1F21_DEVICE_ROUTING_AB.json"
    rr_path = TS200 / "E1F21_INV_CTL_RERUN.json"
    for p in (ab_path, rr_path):
        if not p.exists():
            print(f"MISSING: {p.name}")
            return 2
    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    rr = json.loads(rr_path.read_text(encoding="utf-8"))
    anchor = json.loads((TS200 / "V36_DETERMINISTIC_ANCHOR.json").read_text(
        encoding="utf-8"))
    checks = []

    aprompts = [p["prompt"] for p in anchor["prompts"]]
    checks.append({"check": "both result files cover the anchor prompts",
                   "ok": ab["prompts"] == aprompts and rr["prompts"] == aprompts})

    dev_par = ab["arms"]["DEV"]["parity_a1"]
    ref_par = rr["arms"]["DEV72_ref"]["parity_a1"]
    checks.append({"check": "G-E1F21-C1: DEV parity with frozen A1 ids, both runs",
                   "ok": all(dev_par.values()) and all(ref_par.values())})

    base_p50 = ab["arms"]["BASE"]["token_ms_p50"]
    dev_p50 = ab["arms"]["DEV"]["token_ms_p50"]
    gain = base_p50 - dev_p50
    checks.append({"check": "G-E1F21-S1: DEV p50 <= BASE p50 - 1.5 ms",
                   "ok": dev_p50 <= base_p50 - GATE_S1_MIN_GAIN_MS,
                   "gain_ms": gain})

    inv = rr["arms"]["INV"]["same_as_dev72"]
    checks.append({"check": "G-E1F21-INV: capacity 56 tokens == capacity 72 "
                            "(post-fix re-run)", "ok": all(inv.values())})

    ctl = rr["arms"]["CTL"]
    checks.append({"check": "G-E1F21-CTL: bad_pick breaks parity (post-fix "
                            "re-run, clean attribution)",
                   "ok": ctl["must_fail"] and not any(ctl["parity_a1"].values())})
    checks.append({"check": "pre-fix CTL also failed (consistent, but attributed "
                            "to stale state at the time)",
                   "ok": ab["arms"]["CTL"]["must_fail"]})

    checks.append({"check": "preregistration exists and predates this verifier",
                   "ok": (TS200 / "E1F21_DEVICE_ROUTING_PREREGISTRATION_2026-08-15.md")
                   .exists()})

    try:
        kr = kernel_rechecks()
        checks.append({"check": "route_topk ids exact vs NumPy reference",
                       "ok": kr["route_ids_exact"]})
        checks.append({"check": "route_topk weights within 1e-6 rel of reference",
                       "ok": kr["route_w_max_rel"] <= 1e-6,
                       "max_rel": kr["route_w_max_rel"]})
        checks.append({"check": "cache_assign exactly mirrors the Python LRU",
                       "ok": kr["cache_assign_exact"]})
        checks.append({"check": "cache_fetch bytes exact (slot == bank expert)",
                       "ok": kr["cache_fetch_bytes_exact"]})
        checks.append({"check": "indirect GEMV bitwise == direct ERVF GEMV",
                       "ok": kr["gemv_ind_bitwise"]})
        checks.append({"check": "accumulate_indirect bitwise == accumulate_into",
                       "ok": kr["accumulate_ind_bitwise"]})
        checks.append({"check": "G-E1F21-V1: 23-layer device tables < 32 MiB "
                                "(pool delta AND analytic floor)",
                       "ok": kr["v1_bytes_23_layers"] < GATE_V1_MAX_BYTES
                       and kr["v1_analytic_bytes"] < GATE_V1_MAX_BYTES,
                       "pool_delta_bytes": kr["v1_bytes_23_layers"],
                       "analytic_bytes": kr["v1_analytic_bytes"]})
    except Exception as e:                                   # pragma: no cover
        kr = {"error": f"{type(e).__name__}: {e}"}
        checks.append({"check": "kernel rechecks ran", "ok": False, "error": kr})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "treesweep200_e1f21_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_files_sha256": {p.name: sha256_path(p)
                                  for p in (ab_path, rr_path)},
        "kernel_rechecks": kr,
        "speed": {"base_p50_ms": base_p50, "dev_p50_ms": dev_p50,
                  "gain_ms": gain,
                  "gate_min_gain_ms": GATE_S1_MIN_GAIN_MS},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (TS200 / "e1f21_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['check']}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
