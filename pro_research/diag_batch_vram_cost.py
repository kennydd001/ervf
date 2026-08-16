"""Closes risk #4 in agents/BATCH_ARCHITECTURE_DESIGN.md ("VRAM... batch>1
kost VRAM die er niet is... een nieuwe afweging, niet gemeten"): computes the
EXACT per-additional-sequence VRAM cost of a batch>1 integration (N-fold
KV-cache + Mamba ssm/conv state -- the two buffer classes the design doc
identifies as needing a batch dimension with no sharing possible, since they
are not expert-routed) directly from the real config and runtime.py's own
_alloc_state formulas, then checks it against actually-measured free VRAM at
two real operating points: eager + device cache (no graph) and, from the
existing V4/V6 preregistration record, full graph capture.

CPU/host-only arithmetic (no GPU allocation needed for the byte-cost
formulas -- they come straight from config), plus one real nvidia-smi read
at the eager+cache operating point for a concrete headroom number.

Not a gated PRO experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, nvidia_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

CONTEXTS_MAX = 4096   # same value every script this session uses
CACHE_CAP = 72        # shipped default (V6 / A1 adoption)


def main() -> int:
    from moe_lab.lightningstream_nemotron.runtime import ShardIndex

    model_dir = require_model_dir()
    idx = ShardIndex(model_dir)
    c = idx.config
    pattern = idx.pattern_string()

    attn_layers = [i for i, ch in enumerate(pattern) if ch == "*"]
    mamba_layers = [i for i, ch in enumerate(pattern) if ch == "M"]
    moe_layers = [i for i, ch in enumerate(pattern) if ch == "E"]

    n_kv = c["num_key_value_heads"]
    head_dim = c["head_dim"]
    kv_dim = n_kv * head_dim
    m_heads = c["mamba_num_heads"]
    m_hdim = c["mamba_head_dim"]
    n_state = c["ssm_state_size"]
    n_groups = c["n_groups"]
    conv_k = c["conv_kernel"]
    d_inner = m_heads * m_hdim
    conv_dim = d_inner + 2 * n_groups * n_state

    # -- per-sequence buffer cost, exactly mirroring LightningRuntime._alloc_state.
    # KV cache: FP8 E4M3 (1 byte/element), K and V, per attention layer.
    kv_bytes_per_layer = 2 * CONTEXTS_MAX * kv_dim * 1  # uint8
    kv_bytes_total = kv_bytes_per_layer * len(attn_layers)

    # Mamba ssm state + conv state: float32 (4 bytes/element), per mamba layer.
    ssm_bytes_per_layer = m_heads * m_hdim * n_state * 4
    conv_bytes_per_layer = conv_dim * conv_k * 4
    mamba_bytes_total = (ssm_bytes_per_layer + conv_bytes_per_layer) * len(mamba_layers)

    per_sequence_bytes = kv_bytes_total + mamba_bytes_total
    per_sequence_mib = per_sequence_bytes / (1024 ** 2)

    # -- real headroom at the eager + device-cache operating point (no graph
    # capture): build exactly that stack and read nvidia-smi at the end, same
    # config every diag_*/proto_batch_*.py script this session used.
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(model_dir, contexts_max=CONTEXTS_MAX, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(CACHE_CAP)
    rt.load_routed_bank()
    cp.cuda.Device(0).synchronize()
    smi_after_load = nvidia_snapshot()

    def parse_mib(field: str) -> float:
        return float(field.strip().split()[0])

    parts = [p.strip() for p in smi_after_load.split(",")]
    total_mib = parse_mib(parts[2])
    used_mib = parse_mib(parts[3])
    free_mib_eager = total_mib - used_mib

    max_additional_sequences_eager = int(free_mib_eager // per_sequence_mib)

    # Known from V4/V6 preregistration records (agents/TODO.md, 2026-08-16):
    # full graph capture (setup_graph/step_graph) leaves ~0 MiB free at this
    # same contexts_max/cache_cap -- recorded here as a fact from that prior,
    # separately-verified measurement, not re-derived by this script.
    known_full_graph_free_mib = 0

    payload = {
        "kind": "diag_batch_vram_cost",
        "created_utc": utc_now(),
        "note": "host-side arithmetic (KV-cache + Mamba-state byte cost from config, mirrors runtime.py's own _alloc_state formulas) plus one real nvidia-smi read at the eager+device-cache operating point -- closes design-doc risk #4 with an actual number",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "contexts_max": CONTEXTS_MAX,
        "cache_capacity": CACHE_CAP,
        "n_attn_layers": len(attn_layers),
        "n_mamba_layers": len(mamba_layers),
        "n_moe_layers": len(moe_layers),
        "kv_dim": kv_dim,
        "kv_bytes_per_attn_layer": kv_bytes_per_layer,
        "kv_bytes_total_one_sequence": kv_bytes_total,
        "ssm_bytes_per_mamba_layer": ssm_bytes_per_layer,
        "conv_bytes_per_mamba_layer": conv_bytes_per_layer,
        "mamba_bytes_total_one_sequence": mamba_bytes_total,
        "per_additional_sequence_bytes": per_sequence_bytes,
        "per_additional_sequence_mib": per_sequence_mib,
        "nvidia_smi_after_eager_cache_load": smi_after_load,
        "total_vram_mib": total_mib,
        "used_vram_mib_eager_cache": used_mib,
        "free_vram_mib_eager_cache": free_mib_eager,
        "max_additional_sequences_fit_eager_no_graph": max_additional_sequences_eager,
        "max_n_eager_no_graph": max_additional_sequences_eager + 1,
        "known_full_graph_free_mib": known_full_graph_free_mib,
        "max_additional_sequences_fit_full_graph": max(0, known_full_graph_free_mib) // per_sequence_mib if per_sequence_mib else None,
    }
    out = REPO / "pro_research" / "diag_batch_vram_cost.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
