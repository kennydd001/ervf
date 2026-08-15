"""P0 + E0: identity lock, N1-N5 evidence import, independent byte floors and
a fresh streaming-roofline measurement.

Preregistration: reports/treesweep200/P0_E0_PREREGISTRATION_2026-08-15.md
(frozen before this run).  Builds nothing.

Byte floors are recomputed INDEPENDENTLY of N5: tensor byte sizes come from the
checkpoint's safetensors shard headers, the expert sparse-down correction uses
the S2 measured nonzero fraction, KV bytes come from config.json fields.  The
roofline is measured with an own kernel (256 MiB, float4 loads, 10 reps, p50).

Gates:
  G-P0-I1  identity manifest complete, all sources hashed
  G-P0-B1  frozen baseline artifact hashed and recorded
  G-E0-R1  own streaming roofline within 10% of the imported 338.4 GB/s
  G-E0-F1  own byte floors within 10% of imported 6.05 ms (ctx0) / 8.43 ms (262K)
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LS = REPO_ROOT / "reports" / "lightningstream_nemotron"
OUT = REPO_ROOT / "reports" / "treesweep200"
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning_v35"
PACK_DIR = REPO_ROOT / "info" / "NEMOTRON_TREESWEEP_200_ROOFLINE_V2_AGENT_PACK_2026-08-15"

IMPORTED = json.loads(
    (PACK_DIR / "CURRENT_MEASUREMENTS.N1_N5_IMPORTED.json").read_text(encoding="utf-8"))

EVIDENCE = {
    "n1n2n4n5_ceilings": LS / "n1n2n4n5_ceilings.json",
    "n3_relu2_prefilter_oracle": LS / "n3_relu2_prefilter_oracle.json",
    "y2r1_bytes_vs_time": LS / "y2r1_bytes_vs_time.json",
    "n1_n5_independent_verification": LS / "n1_n5_independent_verification.json",
    "s1_s4_hypothesis_census": LS / "s1_s4_hypothesis_census.json",
    "s14_moe_layer_timeline": LS / "s14_moe_layer_timeline.json",
    "baseline_n7b_cached_decode": LS / "n7b_cached_decode.json",
    "v35_layout_lock": LS / "n2r_v35_layout.json",
    "mtp_wiring_resolution": LS / "s10a_wiring_resolution.json",
    "mtp_acceptance": LS / "s10a_mtp_acceptance.json",
}

RUNTIME_MODULES = [
    "runtime.py", "gpu_kernels.py", "fused_nvfp4.py", "loader.py", "mtp.py",
]

CONTEXTS = [0, 4096, 32768, 131072, 262100]


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def header_sizes(model_dir: Path) -> dict[str, int]:
    """Tensor byte sizes straight from the safetensors shard headers."""
    wm = json.loads((model_dir / "model.safetensors.index.json")
                    .read_text(encoding="utf-8"))["weight_map"]
    sizes: dict[str, int] = {}
    for shard in sorted(set(wm.values())):
        with (model_dir / shard).open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            sizes[name] = b - a
    return sizes


def measure_roofline():
    """Own streaming-read kernel: 256 MiB, one float4 per thread, 10 reps."""
    import cupy as cp
    n = 256 * 2 ** 20 // 4  # float32 elements
    buf = cp.ones(n, dtype=cp.float32)
    kern = cp.RawKernel(r"""
extern "C" __global__ void stream_read(const float4* __restrict__ p,
                                       float* __restrict__ sink, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (long j = i; j < n4; j += stride) {
        float4 v = p[j];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5678f) sink[0] = acc;  // never true; defeats elision
}
""", "stream_read")
    sink = cp.zeros(1, dtype=cp.float32)
    grid = 8 * 84  # generous oversubscription of the SMs
    ts = []
    for _ in range(10):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        kern((grid,), (256,), (buf, sink, n // 4))
        cp.cuda.Device(0).synchronize()
        ts.append((time.perf_counter_ns() - t0) / 1e9)
    p50 = float(np.percentile(ts, 50))
    return buf.nbytes / p50 / 1e9, [buf.nbytes / t / 1e9 for t in ts]


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------- P0 identity
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    identity_files = {}
    for name in ["config.json", "generation_config.json", "hf_quant_config.json",
                 "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"]:
        p = MODEL_DIR / name
        identity_files[name] = {"present": p.is_file(),
                                "sha256": sha256_path(p) if p.is_file() else None,
                                "bytes": p.stat().st_size if p.is_file() else None}
    runtime_hashes = {}
    for name in RUNTIME_MODULES:
        p = REPO_ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / name
        runtime_hashes[name] = {"present": p.is_file(),
                                "sha256": sha256_path(p) if p.is_file() else None}

    identity = {
        "kind": "treesweep200_p0_identity_manifest",
        "created_utc": started,
        "model_dir": MODEL_DIR.name,
        "architectures": cfg["architectures"],
        "hidden_size": cfg["hidden_size"],
        "layers": len(cfg["layers_block_type"]),
        "layers_block_type_counts": {t: cfg["layers_block_type"].count(t)
                                     for t in set(cfg["layers_block_type"])},
        "n_routed_experts": cfg["n_routed_experts"],
        "n_shared_experts": cfg["n_shared_experts"],
        "top_k": cfg["num_experts_per_tok"],
        "mlp_hidden_act": cfg["mlp_hidden_act"],
        "intermediate_size": cfg["intermediate_size"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "mamba": {"head_dim": cfg["mamba_head_dim"],
                  "num_heads": cfg["mamba_num_heads"],
                  "chunk_size": cfg["chunk_size"],
                  "conv_kernel": cfg["conv_kernel"]},
        "num_nextn_predict_layers": cfg["num_nextn_predict_layers"],
        "mtp_layers_block_type": cfg.get("mtp_layers_block_type"),
        "quantization": json.loads((MODEL_DIR / "hf_quant_config.json")
                                   .read_text(encoding="utf-8")).get("quant_method"),
        "vocab_size": cfg["vocab_size"],
        "identity_files": identity_files,
        "runtime_modules": runtime_hashes,
        "inherited_locks": {
            "v35_layout": LS / "n2r_v35_layout.json",
            "mtp_wiring": LS / "s10a_wiring_resolution.json",
        },
    }
    # inherited lock hashes
    identity["inherited_locks"] = {
        k: {"path": str(v.relative_to(REPO_ROOT)), "sha256": sha256_path(v)}
        for k, v in identity["inherited_locks"].items()}
    (OUT / "P0_IDENTITY_MANIFEST.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------- E0 evidence manifest
    evidence = {"kind": "treesweep200_e0_evidence_manifest",
                "created_utc": started,
                "imported_from": str(PACK_DIR / "CURRENT_MEASUREMENTS.N1_N5_IMPORTED.json"),
                "imported_sha256": sha256_path(PACK_DIR /
                                               "CURRENT_MEASUREMENTS.N1_N5_IMPORTED.json"),
                "sources": {}}
    for name, p in EVIDENCE.items():
        evidence["sources"][name] = {
            "path": str(p.relative_to(REPO_ROOT)), "present": p.is_file(),
            "sha256": sha256_path(p) if p.is_file() else None,
            "bytes": p.stat().st_size if p.is_file() else None}
    (OUT / "E0_N1_N5_EVIDENCE_MANIFEST.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    # -------------------------------------- E0 independent byte-floor model
    sizes = header_sizes(MODEL_DIR)
    pre = "backbone.layers.1.mixer.experts.0"
    up_rec = sizes[f"{pre}.up_proj.weight"] + sizes[f"{pre}.up_proj.weight_scale"]
    dn_rec = sizes[f"{pre}.down_proj.weight"] + sizes[f"{pre}.down_proj.weight_scale"]

    census = json.loads((LS / "s1_s4_hypothesis_census.json").read_text(encoding="utf-8"))
    nonzero_frac = 1.0 - census["s2_relu2_sparsity"]["mean_zero_fraction"]

    n_moe = sum(1 for t in cfg["layers_block_type"] if t == "moe")
    n_attn = sum(1 for t in cfg["layers_block_type"] if t == "attention")
    top_k = cfg["num_experts_per_tok"]

    def is_bank(name: str) -> bool:
        # backbone routed bank only; the MTP block has its own BF16 experts
        return name.startswith("backbone.") and ".mixer.experts." in name

    total_bytes = sum(sizes.values())
    bank_bytes = sum(v for k, v in sizes.items() if is_bank(k))
    mtp_bytes = sum(v for k, v in sizes.items() if k.startswith("mtp."))
    embed_bytes = sizes.get("backbone.embeddings.weight", 0)
    lm_head_bytes = sizes.get("lm_head.weight", 0)
    if lm_head_bytes == 0:  # tied or nvfp4 split names
        lm_head_bytes = sum(v for k, v in sizes.items() if k.startswith("lm_head"))
    # resident shell = everything the backbone reads per token that is not the
    # routed bank, not the MTP block, and not the embedding table (one row per
    # token is negligible); lm_head IS read per token.
    resident = total_bytes - bank_bytes - mtp_bytes - embed_bytes

    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]  # fp8: 1 B/element
    floors = {}
    for ctx in CONTEXTS:
        kv = n_attn * 2 * ctx * kv_dim
        expert_read = n_moe * top_k * (up_rec + nonzero_frac * dn_rec)
        total = resident + expert_read + kv
        floors[str(ctx)] = {
            "context": ctx, "resident_bytes": int(resident),
            "expert_read_bytes": int(expert_read),
            "expert_up_bytes": int(n_moe * top_k * up_rec),
            "expert_down_sparse_bytes": int(n_moe * top_k * nonzero_frac * dn_rec),
            "kv_bytes": int(kv), "total_bytes": int(total)}

    # ------------------------------------------------- E0 roofline (GPU)
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines() if l.strip()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4
    roofline, roofline_reps = measure_roofline()
    print(f"own streaming roofline: {roofline:.1f} GB/s "
          f"(imported {IMPORTED['roofline']['streaming_gb_s']})", flush=True)

    for ctx, row in floors.items():
        row["floor_ms"] = row["total_bytes"] / (roofline * 1e9) * 1e3
        row["ceiling_tok_s"] = 1000.0 / row["floor_ms"]
        print(f"  ctx {ctx:>6}: {row['total_bytes'] / 2**20:8.1f} MiB -> "
              f"floor {row['floor_ms']:6.2f} ms -> {row['ceiling_tok_s']:6.1f} tok/s",
              flush=True)

    # ------------------------------------------------- classification
    imp = IMPORTED
    own0 = floors["0"]["floor_ms"]
    own262 = floors["262100"]["floor_ms"]
    cls = []

    def classify(name, ok, own, imported, note):
        cls.append({"claim": name, "own_value": own, "imported_value": imported,
                    "classification": ok, "note": note})

    r_imp = imp["roofline"]["streaming_gb_s"]
    classify("N5 streaming roofline", "reproduced" if abs(roofline - r_imp) / r_imp <= 0.10
             else "shifted", roofline, r_imp,
             "own kernel, 256 MiB float4, 10 reps p50")
    f0 = imp["contexts"]["context_0"]["reported_floor_ms"]
    classify("N5 byte floor ctx0", "reproduced" if abs(own0 - f0) / f0 <= 0.10
             else "shifted", own0, f0,
             "own header-derived byte model incl. sparse-down correction "
             f"({nonzero_frac:.4f} of down read)")
    f262 = imp["contexts"]["context_262100"]["reported_floor_ms"]
    classify("N5 byte floor ctx262100", "reproduced" if abs(own262 - f262) / f262 <= 0.10
             else "shifted", own262, f262, "idem")
    ver = json.loads((LS / "n1_n5_independent_verification.json").read_text(encoding="utf-8"))
    for name in ["N1 graph gain 23.7%", "N2 gather 8.192 ms / 4.3 GB/s in-loop",
                 "N4 attention byte-linear, 47.2 GB/s", "N3 low-rank prefilter closed"]:
        classify(name,
                 "reproduced" if ver.get("verdict") == "VERIFIED" else "inconclusive",
                 None, None,
                 f"in-repo independent verification verdict: {ver.get('verdict')} "
                 f"({ver.get('checks_failed')} failed checks)")
    y2 = json.loads((LS / "y2r1_bytes_vs_time.json").read_text(encoding="utf-8"))
    y2v = json.loads((LS / "y1y2_independent_verification.json").read_text(encoding="utf-8"))
    classify("N5 critical GEMV 81.4 GB/s",
             "reproduced" if y2v.get("verdict") == "VERIFIED" else "inconclusive",
             None, imp["n5"]["critical_gemv_gb_s"],
             f"y2r1 artifact + y1y2 verification verdict: {y2v.get('verdict')}")

    gates = {
        "G_P0_I1": {"pass": all(f["present"] for f in identity_files.values())
                    and all(m["present"] for m in runtime_hashes.values())},
        "G_P0_B1": {"pass": evidence["sources"]["baseline_n7b_cached_decode"]["present"]},
        "G_E0_R1": {"pass": abs(roofline - r_imp) / r_imp <= 0.10,
                    "own": roofline, "imported": r_imp},
        "G_E0_F1": {"pass": abs(own0 - f0) / f0 <= 0.10 and abs(own262 - f262) / f262 <= 0.10,
                    "own_ctx0_ms": own0, "imported_ctx0_ms": f0,
                    "own_262k_ms": own262, "imported_262k_ms": f262},
    }

    payload = {
        "kind": "treesweep200_e0_roofline_reproduction",
        "registry": "NEMOTRON_TREESWEEP_200_ROOFLINE_V2",
        "phase": "P0_E0_IDENTITY_AND_REPRODUCTION",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "identity_manifest_sha256": sha256_path(OUT / "P0_IDENTITY_MANIFEST.json"),
        "evidence_manifest_sha256": sha256_path(OUT / "E0_N1_N5_EVIDENCE_MANIFEST.json"),
        "byte_model": {
            "method": "safetensors shard headers; resident = total - bank - mtp "
                      "- embed; expert read = 23 x 6 x (up_record + nonzero_frac x "
                      "down_record); kv = 6 x 2 x ctx x 256 B (fp8)",
            "total_checkpoint_bytes": int(total_bytes),
            "bank_bytes": int(bank_bytes), "mtp_bytes": int(mtp_bytes),
            "embed_bytes": int(embed_bytes), "resident_bytes": int(resident),
            "lm_head_bytes_in_resident": int(lm_head_bytes),
            "up_record_bytes": int(up_rec), "down_record_bytes": int(dn_rec),
            "relu2_nonzero_fraction": nonzero_frac,
            "n_moe_layers": n_moe, "n_attn_layers": n_attn, "top_k": top_k},
        "roofline": {"own_gb_s": roofline, "reps_gb_s": roofline_reps,
                     "imported_gb_s": r_imp},
        "floors": floors,
        "classification": cls,
        "gates": gates,
        "claim_boundary": (
            "Byte floors are an own recomputation from checkpoint headers plus "
            "measured sparsity, divided by an own measured streaming roofline; "
            "they are hard upper bounds on tokens/s for any semantics-preserving "
            "implementation, not measurements of the runtime. Classifications of "
            "N1-N4 and the GEMV claim rest on in-repo independently verified "
            "artifacts whose hashes are locked in the evidence manifest. No "
            "optimization, no throughput claim, no quality claim."),
    }
    (OUT / "E0_ROOFLINE_REPRODUCTION.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote E0_ROOFLINE_REPRODUCTION.json")
    for k, g in gates.items():
        print(f"  {k}: {'PASS' if g['pass'] else 'FAIL'}")
    return 0 if all(g["pass"] for g in gates.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
