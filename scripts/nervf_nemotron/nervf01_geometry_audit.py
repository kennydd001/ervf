"""NERVF-0/1/2: baseline lock, reduction-geometry audit, and the ERVF microkernel.

Preregistered in reports/nervf_nemotron/NERVF_0_1_PREREGISTRATION_2026-08-15.md.

The production gemv_nvfp4_rows gives every output row a full 256-thread block and
reduces through shared memory with a __syncthreads -- structurally the Qwen shape
from before ERVF. This measures where 338 GB/s of raw scan collapses to 81 GB/s
of GEMV, and then builds the ERVF form: w-lane subwarps, 256/w rows per block,
each lane holding 256/w separate virtual-thread accumulators, and the reference
reduction tree reconstructed exactly.
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

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron import nvfp4  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "nervf_nemotron"
WIDTHS = [4, 8, 16, 32]
GATE_1A = 0.40
GATE_1B_SHARE = 0.25
GATE_SPEEDUP = 1.35
CALLS = 200
ROUNDS = 9

# ---------------------------------------------------------------- audit arms
_AUDIT = r"""
extern "C" __global__ void raw_scan(const uchar4* __restrict__ codes,
                                    const uchar4* __restrict__ scales,
                                    float* __restrict__ sink,
                                    const long nc4, const long ns4)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (long j = i; j < nc4; j += stride) { uchar4 v = codes[j]; acc += v.x + v.y + v.z + v.w; }
    for (long j = i; j < ns4; j += stride) { uchar4 v = scales[j]; acc += v.x + v.y + v.z + v.w; }
    if (acc == 0xFFFFFFFFu) sink[0] = 1.0f;
}

// The real per-output-row access pattern, no MAC and no reduction.
extern "C" __global__ void row_pattern_scan(const unsigned char* __restrict__ codes,
                                            const unsigned char* __restrict__ scales,
                                            float* __restrict__ sink,
                                            const int rows, const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    // codes/scales are already offset to this call's replica by the host
    const int n_bytes = cols >> 1, n_scales = cols >> 4;
    const uchar4* crow = reinterpret_cast<const uchar4*>(codes + (size_t)row * n_bytes);
    const unsigned char* srow = scales + (size_t)row * n_scales;
    unsigned int acc = 0u;
    for (int v = threadIdx.x; v < (n_bytes >> 2); v += blockDim.x) {
        uchar4 q = crow[v];
        acc += q.x + q.y + q.z + q.w + srow[(v << 2) >> 3];
    }
    if (acc == 0xFFFFFFFFu) sink[row] = 1.0f;
}

// Decode + scales, no full dot reduction (one accumulator, no tree, no shared x).
extern "C" __global__ void decode_scale(const unsigned char* __restrict__ codes,
                                        const unsigned char* __restrict__ scales,
                                        const float* __restrict__ e2m1_lut,
                                        const float* __restrict__ e4m3_lut,
                                        float* __restrict__ sink,
                                        const float gs, const int rows, const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();
    const int n_bytes = cols >> 1;
    const unsigned char* crow = codes + (size_t)row * n_bytes;
    const unsigned char* srow = scales + (size_t)row * (cols >> 4);
    const uchar4* crow4 = reinterpret_cast<const uchar4*>(crow);
    float acc = 0.0f;
    for (int v = threadIdx.x; v < (n_bytes >> 2); v += blockDim.x) {
        const uchar4 q = crow4[v];
        const int b = v << 2;
        const float s = e4m3_lut[srow[b >> 3]] * gs;
        acc += s_e2m1[q.x & 15] * s + s_e2m1[q.x >> 4] * s;
        acc += s_e2m1[q.y & 15] * s + s_e2m1[q.y >> 4] * s;
        acc += s_e2m1[q.z & 15] * s + s_e2m1[q.z >> 4] * s;
        acc += s_e2m1[q.w & 15] * s + s_e2m1[q.w >> 4] * s;
    }
    if (acc == 1e30f) sink[row] = acc;
}
"""

# ------------------------------------------------------------ ERVF microkernel
_ERVF_TEMPLATE = r"""
#define WIDTH __W__
#define VIRTUAL (256 / WIDTH)
#define ROWS_PER_BLOCK (256 / WIDTH)

// ERVF form of gemv_nvfp4_rows.
//
// Reference: 256 threads per row; thread tid walks v = tid, tid+256, ...; then a
// 32-wide butterfly per warp, warp sums through shared memory, then a second
// butterfly over the 8 warp sums.
//
// Here: WIDTH physical lanes per row, ROWS_PER_BLOCK rows per 256-thread block.
// Lane L holds VIRTUAL accumulators for the virtual threads tid = L + WIDTH*vi,
// so no MAC moves and no accumulator merges early. The reference tree is then
// rebuilt exactly:
//   * its first step (offset 16 inside a 32-warp) pairs tid and tid+16, which in
//     this mapping are two virtual accumulators OF THE SAME PHYSICAL LANE ->
//     a lane-local add, no shuffle;
//   * offsets 8/4/2/1 stay shuffles, now within a WIDTH-wide subwarp;
//   * the 8 warp sums combine in registers in exactly the order the reference's
//     second butterfly imposes: ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)).
extern "C" __global__ void gemv_nvfp4_ervf(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const float*         __restrict__ e2m1_lut,
    const float*         __restrict__ e4m3_lut,
    const float*         __restrict__ x,
    float*               __restrict__ out,
    const float global_scale,
    const int rows, const int cols,
    const int apply_relu2, const float out_scale)
{
    extern __shared__ float sx[];                 // [cols], shared by all rows
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
    __shared__ float s_e2m1[16];
    if (threadIdx.x < 16) s_e2m1[threadIdx.x] = e2m1_lut[threadIdx.x];
    __syncthreads();

    const int lane = (int)threadIdx.x & (WIDTH - 1);
    const int sub  = (int)threadIdx.x / WIDTH;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + sub;
    if (row >= rows) return;

    const int n_bytes  = cols >> 1;
    const int n_vec    = n_bytes >> 2;
    const unsigned char* __restrict__ crow = codes  + (size_t)row * n_bytes;
    const unsigned char* __restrict__ srow = scales + (size_t)row * (cols >> 4);
    const uchar4* __restrict__ crow4 = reinterpret_cast<const uchar4*>(crow);

    float part[VIRTUAL];
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) part[vi] = 0.0f;

    // Each virtual thread walks exactly the stride the reference gave it.
    #pragma unroll
    for (int vi = 0; vi < VIRTUAL; ++vi) {
        const int tid = lane + WIDTH * vi;
        float acc = 0.0f;
        for (int v = tid; v < n_vec; v += 256) {
            const uchar4 q = crow4[v];
            const int b = v << 2;
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[q.x & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[q.x >> 4]   * s, sx[k + 1], acc);
            acc = fmaf(s_e2m1[q.y & 0x0F] * s, sx[k + 2], acc);
            acc = fmaf(s_e2m1[q.y >> 4]   * s, sx[k + 3], acc);
            acc = fmaf(s_e2m1[q.z & 0x0F] * s, sx[k + 4], acc);
            acc = fmaf(s_e2m1[q.z >> 4]   * s, sx[k + 5], acc);
            acc = fmaf(s_e2m1[q.w & 0x0F] * s, sx[k + 6], acc);
            acc = fmaf(s_e2m1[q.w >> 4]   * s, sx[k + 7], acc);
        }
        for (int b = (n_vec << 2) + tid; b < n_bytes; b += 256) {
            const unsigned char byte = crow[b];
            const float s = e4m3_lut[srow[b >> 3]] * global_scale;
            const int k = b << 1;
            acc = fmaf(s_e2m1[byte & 0x0F] * s, sx[k],     acc);
            acc = fmaf(s_e2m1[byte >> 4]   * s, sx[k + 1], acc);
        }
        part[vi] = acc;
    }

    // ---- rebuild the reference reduction tree, exactly.
    // Virtual tid t sits in reference warp t/32 at intra-warp lane t%32.
    // With tid = lane + WIDTH*vi and lane < WIDTH <= 32, the reference's
    // offset-16 step pairs virtual accumulators of THIS lane whenever
    // WIDTH <= 16; for WIDTH == 32 lane and lane+16 are different lanes and the
    // step stays a shuffle. Both cases are handled below.
    float s8[8];
#if WIDTH <= 16
    // Reference offsets >= WIDTH act on the virtual index, not on lanes, so they
    // must be folded in BUTTERFLY order (16, 8, 4, ... scaled by WIDTH), not by
    // summing the virtual accumulators sequentially. Sequential folding is what
    // made w=4 and w=8 non-exact in the first pass while w=16 -- where the
    // butterfly happens to be a single step -- came out identical.
    const int per_warp = 32 / WIDTH;              // virtual indices per ref warp
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float loc[per_warp];
        #pragma unroll
        for (int u = 0; u < per_warp; ++u) loc[u] = part[w * per_warp + u];
        #pragma unroll
        for (int stride = per_warp >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int u = 0; u < per_warp; ++u)
                if (u < stride) loc[u] += loc[u + stride];
        }
        float v = loc[0];
        for (int off = WIDTH >> 1; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, WIDTH);
        s8[w] = v;
    }
#else
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float v = part[w];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off, 32);
        s8[w] = v;
    }
#endif
    if (lane == 0) {
        const float t0 = s8[0] + s8[4];
        const float t1 = s8[1] + s8[5];
        const float t2 = s8[2] + s8[6];
        const float t3 = s8[3] + s8[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        const float v  = u0 + u1;
        if (apply_relu2) { const float r = fmaxf(v, 0.0f); out[row] = r * r; }
        else             { out[row] = v * out_scale; }
    }
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


def timeit(fn, cp, calls=CALLS, rounds=ROUNDS):
    for _ in range(10):
        fn()
    cp.cuda.Device(0).synchronize()
    per = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        cp.cuda.Device(0).synchronize()
        per.append((time.perf_counter_ns() - t0) / 1e3 / calls)
    return p50(per), per


def main() -> int:
    import cupy as cp

    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=3)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = ShardIndex(MODEL_DIR)
    cfg = idx.config
    hidden, inter = cfg["hidden_size"], cfg["moe_intermediate_size"]
    moe_layers = [i for i, t in enumerate(cfg["layers_block_type"]) if t == "moe"]

    # ---------------------------------------------------------- NERVF-0 lock
    src_files = {
        "runtime.py": REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py",
        "fused_nvfp4.py": REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py",
        "gpu_kernels.py": REPO_ROOT / "src/moe_lab/lightningstream_nemotron/gpu_kernels.py",
        "runner": Path(__file__),
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    lock = {
        "kind": "nervf_nemotron_baseline_lock", "completed_utc": started,
        "model_dir": MODEL_DIR.name,
        "sha256": {k: sha256_path(v) for k, v in src_files.items()},
        "gpu": gpu, "cupy": cp.__version__,
        "config": {"hidden": hidden, "moe_inter": inter,
                   "moe_layers": len(moe_layers)},
        "frozen_throughput_baseline": {
            "source": "reports/lightningstream_nemotron/n7b_cached_decode.json",
            "tok_s": {"ctx0": 27.574, "ctx32768": 25.523,
                      "ctx131072": 21.794, "ctx262100": 18.358}},
        "frozen_generation_anchor": "reports/treesweep200/V35_GENERATION_ANCHOR.json",
        "prior_ervf": {
            "line": "streamq5_moe (read-only)",
            "report": "reports/streamq5_moe/P7_ERVF_FINAL_REPORT_2026-08-12.md",
            "qwen_q8_speedup": 1.725, "qwen_q5_speedup": 2.386,
            "qwen_width": 16},
    }
    (OUT_DIR / "nervf0_baseline_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"NERVF-0 locked: {gpu}", flush=True)

    # real weights
    pre = f"backbone.layers.{moe_layers[0]}.mixer.experts.0.up_proj"
    codes_h = idx.read_raw(f"{pre}.weight")
    scales_h = idx.read_raw(f"{pre}.weight_scale")
    gs = idx.get_scalar(f"{pre}.weight_scale_2")
    codes = cp.asarray(codes_h)
    scales = cp.asarray(scales_h)
    rng = np.random.default_rng(7)
    x = cp.asarray(rng.standard_normal(hidden).astype(np.float32))
    out_ref = cp.zeros(inter, dtype=cp.float32)
    out_new = cp.zeros(inter, dtype=cp.float32)
    fused = FusedNVFP4()
    nbytes = int(codes.nbytes + scales.nbytes)

    # ------------------------------------------------------------- NERVF-1
    print("\nNERVF-1 geometry audit", flush=True)
    amod = cp.RawModule(code=_AUDIT, options=("-std=c++14",))
    raw_k = amod.get_function("raw_scan")
    row_k = amod.get_function("row_pattern_scan")
    dec_k = amod.get_function("decode_scale")
    sink = cp.zeros(inter, dtype=cp.float32)
    nc4, ns4 = codes.size // 4, scales.size // 4

    # --- L2 defeat: every arm streams through a pool of replicated records so
    # that no arm is measuring L2 residency. The first pass of this audit had a
    # single 2.81 MiB record, which fits L2 and made RAW_SCAN swing 5-10x
    # between runs while the real arms moved 4-12%. N5 avoided this with a
    # 256 MiB buffer; the same discipline is applied here to ALL arms equally.
    l2_bytes = int(cp.cuda.Device(0).attributes.get("L2CacheSize", 32 << 20))
    pool_target = max(256 << 20, 8 * l2_bytes)
    n_rep = max(2, pool_target // nbytes)
    pool_codes = cp.tile(codes, n_rep)
    pool_scales = cp.tile(scales, n_rep)
    pool_bytes = int(pool_codes.nbytes + pool_scales.nbytes)
    cstride, sstride = int(codes.size), int(scales.size)
    print(f"  L2 {l2_bytes / 2**20:.0f} MiB -> pool {n_rep} replicas, "
          f"{pool_bytes / 2**20:.0f} MiB", flush=True)

    counter = {"i": 0}

    def rep():
        r = counter["i"] % n_rep
        counter["i"] += 1
        return (pool_codes[r * cstride:(r + 1) * cstride],
                pool_scales[r * sstride:(r + 1) * sstride])

    def a_raw():
        c_, s_ = rep()
        raw_k((512,), (256,), (c_, s_, sink, np.int64(nc4), np.int64(ns4)))

    def a_row():
        c_, s_ = rep()
        row_k((inter,), (256,), (c_, s_, sink, np.int32(inter), np.int32(hidden)))

    def a_dec():
        c_, s_ = rep()
        dec_k((inter,), (256,), (c_, s_, fused.e2m1, fused.e4m3, sink,
                                 np.float32(gs), np.int32(inter), np.int32(hidden)))

    def a_full():
        c_, s_ = rep()
        fused.gemv_into(out_ref, c_, s_, x, gs, inter, hidden)

    arms = {}
    for nm, fn in (("RAW_SCAN", a_raw), ("ROW_PATTERN_SCAN", a_row),
                   ("DECODE_SCALE", a_dec), ("FULL_GEMV", a_full)):
        counter["i"] = 0
        arms[nm] = timeit(fn, cp)
    audit = {}
    for name, (us, raw) in arms.items():
        audit[name] = {"us_p50": us, "gb_s": nbytes / (us * 1e-6) / 1e9, "raw": raw}
        print(f"  {name:<17} {us:8.2f} us  {audit[name]['gb_s']:7.1f} GB/s", flush=True)

    # The gate is a BANDWIDTH-EFFICIENCY ratio: what fraction of raw-scan
    # bandwidth the full GEMV achieves. Coding it as a time ratio makes it
    # degenerate -- a full GEMV is always slower than a raw scan, so no kernel
    # could ever pass. Same class of error as the C1 sign bug: the tell was that
    # the criterion was unsatisfiable by construction.
    ratio = audit["FULL_GEMV"]["gb_s"] / audit["RAW_SCAN"]["gb_s"]
    red_share = ((audit["FULL_GEMV"]["us_p50"] - audit["DECODE_SCALE"]["us_p50"])
                 / audit["FULL_GEMV"]["us_p50"])
    g1a = ratio <= GATE_1A
    g1b = red_share >= GATE_1B_SHARE
    print(f"  FULL/RAW bandwidth efficiency = {ratio:.4f} (gate <= {GATE_1A}) -> {g1a}", flush=True)
    print(f"  reduction+sync share = {red_share * 100:.1f}% (gate >= 25%) -> {g1b}",
          flush=True)
    x_traffic = inter * hidden * 4
    print(f"  x re-staging traffic {x_traffic / 2**20:.1f} MiB against "
          f"{nbytes / 2**20:.1f} MiB of weights ({x_traffic / nbytes:.1f}x)", flush=True)

    geometry_ok = g1a and g1b
    result = {
        "kind": "nervf_nemotron_geometry_audit", "completed_utc":
            datetime.now(timezone.utc).isoformat(),
        "tensor": pre, "bytes": nbytes, "rows": inter, "cols": hidden,
        "l2_defeat": {"l2_bytes": l2_bytes, "replicas": int(n_rep),
                      "pool_bytes": pool_bytes,
                      "note": "all arms cycle replicas so none is L2-resident"},
        "arms": audit,
        "full_over_raw_bandwidth_efficiency": ratio, "reduction_sync_share": red_share,
        "x_restaging_bytes": int(x_traffic),
        "x_over_weight_ratio": x_traffic / nbytes,
        "gates": {"G_NERVF_1A": {"metric": "full_gemv_gb_s / raw_scan_gb_s",
                                 "required_max": GATE_1A, "measured": ratio,
                                 "passed": bool(g1a)},
                  "G_NERVF_1B": {"required_min_share": GATE_1B_SHARE,
                                 "measured": red_share, "passed": bool(g1b)}},
        "geometry_gate_open": bool(geometry_ok),
        "claim_boundary": (
            "Single-kernel microbenchmarks on one real NVFP4 up_proj record, all "
            "arms on the same bytes, 200 calls per sync. Effective GB/s counts "
            "the weight record only. Not a token time and not a throughput "
            "result."),
    }
    (OUT_DIR / "nervf1_geometry_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not geometry_ok:
        print("\nGEOMETRY GATE CLOSED - ERVF not opened, per preregistration.")
        return 0

    # ------------------------------------------------------------- NERVF-2
    print("\nNERVF-2 ERVF microkernel", flush=True)
    exact_rng = np.random.default_rng(99)
    cases = []
    for li in (moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]):
        for e in range(args.experts):
            p = f"backbone.layers.{li}.mixer.experts.{e}.up_proj"
            cases.append((li, e, cp.asarray(idx.read_raw(f"{p}.weight")),
                          cp.asarray(idx.read_raw(f"{p}.weight_scale")),
                          idx.get_scalar(f"{p}.weight_scale_2")))
    acts = {
        "random": cp.asarray(exact_rng.standard_normal(hidden).astype(np.float32)),
        "adversarial": cp.asarray((exact_rng.standard_normal(hidden)
                                   * 10.0 ** exact_rng.integers(-8, 8, hidden)
                                   ).astype(np.float32)),
        "zero_heavy": cp.asarray(np.where(exact_rng.random(hidden) < 0.9, 0.0,
                                          exact_rng.standard_normal(hidden)
                                          ).astype(np.float32)),
        "dense": cp.asarray((exact_rng.random(hidden) + 0.5).astype(np.float32)),
    }

    ervf = {}
    for w in WIDTHS:
        mod = cp.RawModule(code=_ERVF_TEMPLATE.replace("__W__", str(w)),
                           options=("-std=c++14",))
        k = mod.get_function("gemv_nvfp4_ervf")
        rows_per_block = 256 // w
        grid = (inter + rows_per_block - 1) // rows_per_block
        shared = hidden * 4

        def run(kk=k, gg=grid, ss=shared):
            kk((gg,), (256,), (codes, scales, fused.e2m1, fused.e4m3, x, out_new,
                               np.float32(gs), np.int32(inter), np.int32(hidden),
                               np.int32(0), np.float32(1.0)), shared_mem=ss)

        # exactness over every case x activation x relu2 setting
        mism, total = 0, 0
        for li, e, cc, sc, g in cases:
            for aname, av in acts.items():
                for relu2 in (0, 1):
                    fused.gemv_into(out_ref, cc, sc, av, g, inter, hidden,
                                    apply_relu2=bool(relu2))
                    k((grid,), (256,), (cc, sc, fused.e2m1, fused.e4m3, av, out_new,
                                        np.float32(g), np.int32(inter), np.int32(hidden),
                                        np.int32(relu2), np.float32(1.0)),
                      shared_mem=shared)
                    cp.cuda.Device(0).synchronize()
                    total += 1
                    if not bool(cp.array_equal(out_ref, out_new)):
                        mism += 1
        us, raw = timeit(run, cp)
        ervf[str(w)] = {
            "width": w, "rows_per_block": rows_per_block, "grid": grid,
            "us_p50": us, "gb_s": nbytes / (us * 1e-6) / 1e9,
            "speedup": audit["FULL_GEMV"]["us_p50"] / us,
            "bitwise_cases": total, "bitwise_mismatches": mism,
            "bitwise_identical": mism == 0,
            "regs": int(k.num_regs), "static_shared": int(k.shared_size_bytes),
            "raw": raw,
        }
        r = ervf[str(w)]
        print(f"  w={w:<3} rows/block {rows_per_block:>3} | {us:7.2f} us | "
              f"{r['gb_s']:6.1f} GB/s | speedup {r['speedup']:.3f}x | "
              f"bitwise {mism}/{total} mismatches | regs {r['regs']}", flush=True)

    exact = {k: v for k, v in ervf.items() if v["bitwise_identical"]}
    best = max(exact, key=lambda k: exact[k]["speedup"]) if exact else None
    payload = {
        "kind": "nervf_nemotron_ervf_microkernel",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_us": audit["FULL_GEMV"]["us_p50"],
        "baseline_gb_s": audit["FULL_GEMV"]["gb_s"],
        "widths": ervf, "exact_widths": sorted(exact),
        "best_exact_width": int(best) if best else None,
        "best_speedup": exact[best]["speedup"] if best else None,
        "cases_per_width": {"experts": len(cases), "activations": list(acts),
                            "relu2_variants": 2},
        "gates": {
            "G_NERVF_2C_exact": {"all_widths_bitwise":
                                 {k: v["bitwise_identical"] for k, v in ervf.items()},
                                 "passed": bool(exact)},
            "G_NERVF_2S_primary": {"required": GATE_SPEEDUP,
                                   "measured": exact[best]["speedup"] if best else None,
                                   "passed": bool(best and exact[best]["speedup"]
                                                  >= GATE_SPEEDUP)},
        },
        "claim_boundary": (
            "Isolated projection-plane microbenchmark on one real NVFP4 up_proj "
            "record; bitwise exactness checked against the production kernel over "
            "3 layers x N experts x 4 activation regimes x 2 relu2 settings. This "
            "is NOT a token time, NOT a throughput result, and NOT integrated: "
            "runtime integration is NERVF-3. Speedups are projection-plane only "
            "and may not be added to other component gains."),
    }
    (OUT_DIR / "nervf2_ervf_microkernel.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  best exact width: {best} at {payload['best_speedup']}"
          if best else "\n  no width was bitwise exact")
    print("\nwritten nervf0/1/2 artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
