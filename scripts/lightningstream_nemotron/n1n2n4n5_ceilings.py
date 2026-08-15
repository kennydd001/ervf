"""N1/N2/N4/N5: four ceilings, measured.

Preregistered in N1_N5_OWN_HYPOTHESES_PREREGISTRATION_2026-08-15.md.

N1  Capture one token's kernel sequence as a CUDA graph (routes frozen, because
    capture forbids synchronisation) and replay it. Same kernels, same arguments,
    same bytes; the only difference is who issues them. The delta bounds every
    design that moves work off the host: megakernel, device routing, persistent
    kernels.
N2  Split down_masked_into into panel_scan / gather / masked GEMV by in-loop
    replication, to see whether fusing the three stages is worth a build.
N4  Attention time against KV bytes, by varying context, to see what a cheaper
    KV cache could buy at 262K.
N5  The roofline of one token: bytes any correct forward must read, over the
    MEASURED streaming read bandwidth of this device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import (  # noqa: E402
    LightningRuntime, UP_CODE, UP_SCALE, DOWN_PANEL_BYTES)

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

GATE_N2_SHARE = 0.30
GATE_N4_R2 = 0.98
S14_DOWN_MS = 8.393

_STREAM_SRC = r"""
extern "C" __global__ void stream_read(const float4* __restrict__ src,
                                       float* __restrict__ sink, const long n4)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n4; i += stride) {
        const float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}
"""


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(v):
    return float(np.percentile(np.asarray(v, dtype=np.float64), 50))


class ProbedRuntime(LightningRuntime):
    probe = None
    frozen_routes = None      # layer -> (idx, w); set for capture, no readback
    emb_row = None            # pre-staged device embedding row

    def freeze_routes(self, token_id):
        """Record one token's official routes so the sequence needs no readback."""
        cp = self.cp
        self.frozen_routes = None
        cap = {}
        self._capture_rw = cap
        self.step(token_id)                 # warms the cache and records routes
        self.step(token_id)                 # second pass: every expert now resident
        self._capture_rw = None
        self.frozen_routes = dict(cap)
        row = self.embed_host[token_id * self.hidden:(token_id + 1) * self.hidden]
        self.emb_row = cp.asarray(row)
        cp.cuda.Device(0).synchronize()

    def _moe_frozen(self, i, out):
        """The routed loop with the route already known: no device->host sync."""
        cp, d = self.cp, self.layer[i]
        self._route_device(i)                      # device work stays
        out.fill(0)
        f = self.fused
        act_sh = self.act[:self.shared_inter]
        act_moe = self.act[:self.moe_inter]
        f.gemv_into(act_sh, d["sh_up_c"], d["sh_up_s"], self.normed,
                    d["sh_up_g"], self.shared_inter, self.hidden, apply_relu2=True)
        f.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"], act_sh, d["sh_dn_g"],
                    self.hidden, self.shared_inter)
        idx, w = self.frozen_routes[i]
        bank, c = self.bank[i], self.cache[i]
        for s in range(len(idx)):
            e = int(idx[s])
            slot = c["map"].get(e, -1)
            if slot < 0:
                raise RuntimeError(f"layer {i} expert {e} not resident; "
                                   "prewarm before capture")
            f.gemv_into(act_moe, c["slot_codes"][slot], c["slot_scales"][slot],
                        self.normed, bank["g_up"][e], self.moe_inter, self.hidden,
                        apply_relu2=True)
            f.down_masked_into(self.tmp, bank["down_ptr"][e], act_moe, self.mstate,
                               bank["g_dn"][e], self.hidden, self.moe_inter)
            f.accumulate_into(out, self.tmp, float(w[s]), self.hidden)
        return idx, w

    def step_graph_body(self, token_id):
        """One token, capture-safe: frozen routes, pre-staged embedding, no argmax."""
        cp, k = self.cp, self.k
        self.h[:] = (self.emb_row.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
        for i, ch in enumerate(self.pattern):
            d = self.layer[i]
            k.norm(self.normed, self.h, d["norm"], self.hidden, self.eps)
            if ch == "M":
                self._mamba(i, self.acc)
            elif ch == "*":
                self._attention(i, self.acc)
            else:
                self._moe_frozen(i, self.acc)
            k.add_(self.h, self.acc, self.hidden)
        k.norm(self.normed, self.h, self.norm_f, self.hidden, self.eps)
        self.fused.gemv_into(self.logits, self.lm_head_codes, self.lm_head_scales,
                             self.normed, self.lm_head_g, self.vocab, self.hidden)
        self.pos += 1

    def alloc_probe(self):
        cp = self.cp
        self.p_out = cp.zeros(self.hidden, dtype=cp.float32)
        self.p_state = self.fused.alloc_masked_state(self.hidden, self.moe_inter)

    def _moe_cached(self, i, out):
        idx, w = super()._moe_cached(i, out)
        sink = getattr(self, "_capture_rw", None)
        if sink is not None:
            sink[i] = (np.asarray(idx, dtype=np.int64).copy(),
                       np.asarray(w, dtype=np.float64).copy())
        p = self.probe
        if p is None:
            return idx, w
        bank, st = self.bank[i], self.p_state
        f, cp = self.fused, self.cp
        act = self._act_moe if hasattr(self, "_act_moe") else self.act[:self.moe_inter]
        for e in idx:
            e = int(e)
            if p == "scan":
                f.panel_scan_k((1,), (256,),
                               (act, np.int32(self.moe_inter), st["masks"],
                                st["plist"], st["pcount"], st["nz"], st["nzc"]))
            elif p == "gather":
                npanel = self.moe_inter // 16
                blocks = ((self.moe_inter + npanel) * 32 + 255) // 256
                f.gather_k((blocks,), (256,),
                           (np.uint64(bank["down_base_ptr"] + e * DOWN_PANEL_BYTES),
                            st["mirror"], st["plist"], st["pcount"],
                            st["nz"], st["nzc"], np.int32(self.hidden)))
            elif p == "gemv":
                f.down_masked_into(self.p_out,
                                   bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                                   act, st, float(bank["globals"][e, 0]),
                                   self.hidden, self.moe_inter,
                                   gather_from_host=False)
            elif p == "down_all":
                f.down_masked_into(self.p_out,
                                   bank["down_base_ptr"] + e * DOWN_PANEL_BYTES,
                                   act, st, float(bank["globals"][e, 0]),
                                   self.hidden, self.moe_inter)
        return idx, w


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=262144)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n4-contexts", type=int, nargs="*",
                    default=[32768, 65536, 131072, 196608, 262100])
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = ProbedRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                       embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.alloc_probe()
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=8192)]

    # -------------------------------------------------------------- N5 first
    print("\nN5: measured streaming read bandwidth", flush=True)
    mod = cp.RawModule(code=_STREAM_SRC, options=("-std=c++14",))
    kern = mod.get_function("stream_read")
    buf = cp.zeros(256 * 1024 * 1024 // 4, dtype=cp.float32)   # 256 MiB
    sink = cp.zeros(4, dtype=cp.float32)
    n4 = buf.size // 4
    for _ in range(3):
        kern((1024,), (256,), (buf, sink, np.int64(n4)))
    cp.cuda.Device(0).synchronize()
    ts = []
    for _ in range(15):
        t0 = time.perf_counter_ns()
        kern((1024,), (256,), (buf, sink, np.int64(n4)))
        cp.cuda.Device(0).synchronize()
        ts.append((time.perf_counter_ns() - t0) / 1e9)
    bw = buf.nbytes / p50(ts) / 1e9
    print(f"  streaming read {bw:.1f} GB/s over {buf.nbytes / 2**20:.0f} MiB", flush=True)

    # -------------------------------------------------------------------- N1
    print("\nN1: CUDA-graph replay against eager", flush=True)
    rt.reset()
    for j in range(64):
        rt.step(varied[j % len(varied)])
    cp.cuda.Device(0).synchronize()

    eager = []
    for _ in range(args.reps * 4):
        t0 = time.perf_counter_ns()
        rt.step(varied[0])
        cp.cuda.Device(0).synchronize()
        eager.append((time.perf_counter_ns() - t0) / 1e6)
    t_eager = p50(eager)

    # Freeze the routes so the sequence has no device->host sync, then capture.
    graph, capture_ok, capture_err = None, False, None
    try:
        rt.freeze_routes(varied[1])
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            rt.step_graph_body(varied[1])
            graph = stream.end_capture()
        capture_ok = True
    except Exception as e:                                   # pragma: no cover
        capture_err = f"{type(e).__name__}: {e}"
        print(f"  capture failed: {capture_err}", flush=True)

    t_graph = None
    if capture_ok:
        for _ in range(3):
            graph.launch()
        cp.cuda.Device(0).synchronize()
        gs = []
        for _ in range(args.reps * 4):
            t0 = time.perf_counter_ns()
            graph.launch()
            cp.cuda.Device(0).synchronize()
            gs.append((time.perf_counter_ns() - t0) / 1e6)
        t_graph = p50(gs)
        print(f"  eager {t_eager:.3f} ms | graph {t_graph:.3f} ms | "
              f"removable {(1 - t_graph / t_eager) * 100:+.1f}%", flush=True)

    # -------------------------------------------------------------------- N2
    print("\nN2: down-path stages by in-loop replication", flush=True)
    n2 = {}
    schedule = [("base0", None), ("scan", "scan"), ("base1", None),
                ("gather", "gather"), ("base2", None), ("gemv", "gemv"),
                ("base3", None), ("down_all", "down_all"), ("base4", None)]
    for name, probe in schedule:
        rt.probe = probe
        rt.reset()
        for j in range(48):
            rt.step(varied[j % len(varied)])
        cp.cuda.Device(0).synchronize()
        s = []
        for j in range(12):
            t0 = time.perf_counter_ns()
            rt.step(varied[(j + 64) % len(varied)])
            cp.cuda.Device(0).synchronize()
            s.append((time.perf_counter_ns() - t0) / 1e6)
        n2[name] = {"p50": p50(s), "raw": s}
        print(f"    {name:<9} {n2[name]['p50']:7.3f} ms", flush=True)
    rt.probe = None
    marg = {}
    for k, (name, probe) in enumerate([p for p in schedule if p[1]]):
        b0 = n2[f"base{k}"]["p50"]
        b1 = n2[f"base{k + 1}"]["p50"]
        marg[name] = {"marginal_ms": n2[name]["p50"] - 0.5 * (b0 + b1),
                      "local_drift_ms": abs(b1 - b0)}
    scan_gather = marg["scan"]["marginal_ms"] + marg["gather"]["marginal_ms"]
    share = scan_gather / max(1e-9, marg["down_all"]["marginal_ms"])
    print(f"  scan+gather {scan_gather:.3f} ms of down_all "
          f"{marg['down_all']['marginal_ms']:.3f} ms = {share * 100:.1f}%", flush=True)

    # -------------------------------------------------------------------- N4
    print("\nN4: attention time against KV bytes", flush=True)
    kv_dim = rt.kv_dim
    n4rows = {}
    for ctx in args.n4_contexts:
        if ctx >= args.max_ctx - 8:
            continue
        rt.reset()
        for j in range(min(ctx, 64)):
            rt.step(varied[j % len(varied)])
        rt.pos = ctx
        for j in range(16):
            rt.step(varied[(j + 64) % len(varied)])
        cp.cuda.Device(0).synchronize()
        s = []
        for j in range(20):
            t0 = time.perf_counter_ns()
            for _ in range(len(rt.attn_layers)):
                rt.k.attention_fp8_gqa(rt.ctx, rt.qv, rt.kc[rt.attn_layers[0]],
                                       rt.vc[rt.attn_layers[0]], ctx + 1,
                                       rt.n_heads, rt.head_dim, rt.groups,
                                       rt.max_ctx, 1.0 / float(np.sqrt(rt.head_dim)),
                                       rt.part_acc, rt.part_ml)
            cp.cuda.Device(0).synchronize()
            s.append((time.perf_counter_ns() - t0) / 1e6)
        kv_bytes = len(rt.attn_layers) * 2 * ctx * kv_dim
        n4rows[str(ctx)] = {"context": ctx, "kv_bytes": int(kv_bytes),
                            "ms_p50": p50(s), "raw": s,
                            "gb_s": kv_bytes / (p50(s) * 1e-3) / 1e9}
        print(f"    ctx {ctx:>6}: {p50(s):7.3f} ms  {kv_bytes / 2**20:7.1f} MiB  "
              f"{n4rows[str(ctx)]['gb_s']:6.1f} GB/s", flush=True)

    xs = np.array([r["kv_bytes"] for r in n4rows.values()], dtype=np.float64)
    ys = np.array([r["ms_p50"] for r in n4rows.values()], dtype=np.float64)
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    r2 = 1 - float(((ys - pred) ** 2).sum()) / float(((ys - ys.mean()) ** 2).sum())
    deep = str(max(int(k) for k in n4rows))
    t_deep = n4rows[deep]["ms_p50"]
    t_half = a * (n4rows[deep]["kv_bytes"] / 2) + b
    print(f"  fit {a * 1e9:.4f} ms per GB + {b:.3f} ms fixed, R2={r2:.4f}", flush=True)
    print(f"  halving KV bytes at ctx {deep}: {t_deep:.3f} -> {t_half:.3f} ms "
          f"({(1 - t_half / t_deep) * 100:.1f}%)", flush=True)

    # ---------------------------------------------------------------- N5 sum
    shell_touched = 0
    for i, ch in enumerate(rt.pattern):
        d = rt.layer[i]
        for key in ("in_w8", "out_w8", "in_codes", "in_scales", "out_codes",
                    "out_scales", "q_proj", "k_proj", "v_proj", "o_proj",
                    "sh_up_c", "sh_up_s", "sh_dn_c", "sh_dn_s", "norm",
                    "conv_w", "conv_b", "D", "dt_bias", "m_norm"):
            v = d.get(key)
            if v is not None and hasattr(v, "nbytes"):
                shell_touched += int(v.nbytes)
        if ch == "E":
            shell_touched += int(d["gate_w"].nbytes)
    lm_head_bytes = int(rt.lm_head_codes.nbytes + rt.lm_head_scales.nbytes)
    expert_bytes = len(rt.moe_layers) * rt.top_k * (UP_CODE + UP_SCALE)
    floors = {}
    for label, ctx in (("ctx0", 0), ("ctx262100", 262100)):
        kv = len(rt.attn_layers) * 2 * ctx * kv_dim
        total = shell_touched + lm_head_bytes + expert_bytes + kv
        ms = total / (bw * 1e9) * 1e3
        floors[label] = {"context": ctx, "shell_bytes": shell_touched,
                         "lm_head_bytes": lm_head_bytes,
                         "expert_bytes": int(expert_bytes), "kv_bytes": int(kv),
                         "total_bytes": int(total), "floor_ms": ms,
                         "ceiling_tok_s": 1000.0 / ms}
        print(f"  {label}: {total / 2**20:8.1f} MiB -> floor {ms:6.2f} ms "
              f"-> ceiling {1000.0 / ms:7.1f} tok/s", flush=True)

    gates = {
        "G_N1_1_removable": {"eager_ms": t_eager, "graph_ms": t_graph,
                             "removable_fraction": (1 - t_graph / t_eager)
                             if t_graph else None,
                             "capture_ok": capture_ok, "capture_error": capture_err},
        "G_N2_1_scan_gather_share": {"required": GATE_N2_SHARE, "measured": share,
                                     "passed": bool(share >= GATE_N2_SHARE),
                                     "marginals": marg},
        "G_N4_1_kv_slope": {"ms_per_gb": float(a * 1e9), "fixed_ms": float(b),
                            "r2": r2, "halving_saves_at_deep":
                            float(1 - t_half / t_deep)},
        "G_N4_2_fit_quality": {"required_r2": GATE_N4_R2, "r2": r2,
                               "passed": bool(r2 >= GATE_N4_R2)},
        "G_N5_1_floor": floors,
        "G_N5_2_bandwidth_measured": {"gb_s": bw, "buffer_bytes": int(buf.nbytes)},
    }

    payload = {
        "kind": "lightningstream_nemotron_n1n2n4n5_ceilings",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N1_N2_N4_N5_CEILINGS",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "reps": args.reps, "n4_contexts": args.n4_contexts,
                   "moe_layers": len(rt.moe_layers), "attn_layers": len(rt.attn_layers),
                   "top_k": rt.top_k},
        "n5_bandwidth_gb_s": bw,
        "n1": {"eager_ms": t_eager, "graph_ms": t_graph, "capture_ok": capture_ok,
               "capture_error": capture_err, "eager_raw": eager},
        "n2": {"arms": n2, "marginals": marg, "scan_gather_ms": scan_gather,
               "share_of_down": share, "s14_down_ms": S14_DOWN_MS},
        "n4": {"rows": n4rows, "fit_ms_per_byte": float(a), "fit_fixed_ms": float(b),
               "r2": r2, "deep_context": int(deep), "deep_ms": t_deep,
               "half_kv_ms": float(t_half)},
        "n5": floors,
        "gates": gates,
        "claim_boundary": (
            "N1 replays a captured kernel sequence with FROZEN routes; after the "
            "first token that is semantically wrong and it is used only as a "
            "timing oracle for how much of a token is issue overhead rather than "
            "arithmetic. N2 marginals are in-loop replication lower bounds and "
            "are not shares of the token. N4 times the attention kernel alone at "
            "several context depths on the same KV allocation, so it measures the "
            "kernel's bytes-to-time slope, not the token. N5's floor counts the "
            "bytes a correct forward must read and divides by the MEASURED "
            "streaming read bandwidth of this device; it is a HARD UPPER BOUND on "
            "tokens per second for any implementation preserving semantics, not "
            "an achievable figure and not a measurement of this runtime."),
    }
    (OUT_DIR / "n1n2n4n5_ceilings.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nwritten n1n2n4n5_ceilings.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
