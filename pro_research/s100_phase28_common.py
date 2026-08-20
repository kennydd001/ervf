from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path

import numpy as np

from common import REPO
from moe_dev_batched import UP_CODE, UP_SCALE, DOWN_PANEL_BYTES
from scale_resident_kernels import PLANE_BYTES
from s100_phase21_common import load_trace, release
from s100_phase24_common import (
    selected_config,
    make_synth,
    timed_synth_blocks,
)
from s100_phase27_common import (
    PipelinedMoEH4 as Phase27MoE,
    Variant as Phase27Variant,
)
from s100_phase28_kernels import (
    Phase28MirrorlessKernels,
    ROUTES,
    GROUPS,
    NCHUNKS,
)

RESULTS = REPO / "pro_research" / "results" / "s100_phase28"
SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"

ARM_NAMES = (
    "p27_control",
    "direct_route",
    "group_chunk_v16",
    "group_allchunks_v4",
    "group_allchunks_v16",
    "group_allchunks_v16_overlap",
)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def phase28_gate():
    p24 = load_json(
        REPO
        / "pro_research"
        / "results"
        / "s100_phase24"
        / "S100_PHASE24_SUMMARY.json"
    )
    t24 = load_json(
        REPO
        / "pro_research"
        / "results"
        / "s100_phase24"
        / "S100_PHASE24_THERMAL_ADJUDICATION.json"
    )
    s24 = load_json(
        REPO
        / "pro_research"
        / "results"
        / "s100_phase24"
        / "S100_PHASE24_STATE_CHECK.json"
    )
    p27 = load_json(
        REPO
        / "pro_research"
        / "results"
        / "s100_phase27"
        / "S100_PHASE27_SUMMARY.json"
    )
    p27r = load_json(
        REPO
        / "pro_research"
        / "results"
        / "s100_phase27r"
        / "S100_PHASE27R_SUMMARY.json"
    )

    if not p24.get("instrumentation_complete"):
        raise RuntimeError("Phase24 active parent is incomplete")
    if not t24.get("BEST_OF_ALL_ADOPTED"):
        raise RuntimeError("Phase24 active parent is not adopted")
    if not s24.get("BEST_OF_ALL_STATE_GREEN"):
        raise RuntimeError("Phase24 active parent state gate is not green")

    if not p27.get("instrumentation_complete"):
        raise RuntimeError("Phase27 is incomplete")
    if not p27.get("SELECTED_STATE_GREEN"):
        raise RuntimeError("Phase27 selected state gate is not green")
    if p27.get("PHASE27_ACTIVE_PARENT_ADOPTED"):
        raise RuntimeError("Phase27 unexpectedly replaced Phase24")

    if not p27r.get("instrumentation_complete"):
        raise RuntimeError("Phase27R is incomplete")
    if p27r.get("PHASE27_PIPELINE_ADOPTED"):
        raise RuntimeError("Phase27R unexpectedly adopted the pipeline")
    if p27r.get("NEXT_ROUTE") != (
        "FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC"
    ):
        raise RuntimeError(
            f"unexpected Phase27R route: {p27r.get('NEXT_ROUTE')}"
        )

    adjudication = p27r.get("thermal_adjudication") or {}
    gain = adjudication.get("median_round_gain_fraction")
    if gain is None or not (0.0 < float(gain) < 0.05):
        raise RuntimeError(
            f"Phase27R positive-sub5 premise invalid: gain={gain}"
        )

    cfg = selected_config()
    if cfg is None:
        raise RuntimeError("Phase24 selected config missing")
    if cfg.attention_m4 or cfg.router_m4 or cfg.shared_m4:
        raise RuntimeError(
            "closed Phase24 dense M4 components must remain disabled"
        )
    if len(cfg.sres_layers) != 23:
        raise RuntimeError(
            f"expected 23 resident H-SCALE layers, got "
            f"{len(cfg.sres_layers)}"
        )

    return cfg, p24, p27, p27r


@dataclass(frozen=True)
class Arm:
    name: str

    def __post_init__(self):
        if self.name not in ARM_NAMES:
            raise ValueError(self.name)

    @property
    def mirrorless(self) -> bool:
        return self.name != "p27_control"

    @property
    def mode(self) -> str:
        if self.name == "direct_route":
            return "direct_route"
        if self.name == "group_chunk_v16":
            return "group_chunk"
        if self.name.startswith("group_allchunks"):
            return "group_allchunks"
        return "phase27_control"

    @property
    def vector_bytes(self):
        if self.name == "group_allchunks_v4":
            return 4
        if self.name in (
            "group_allchunks_v16",
            "group_allchunks_v16_overlap",
            "group_chunk_v16",
        ):
            return 16
        return None

    @property
    def shared_overlap(self) -> bool:
        return self.name in (
            "p27_control",
            "group_allchunks_v16_overlap",
        )

    @property
    def eligible_for_phase28_selection(self) -> bool:
        return self.mirrorless

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mirrorless": self.mirrorless,
            "mode": self.mode,
            "vector_bytes": self.vector_bytes,
            "shared_overlap": self.shared_overlap,
            "eligible_for_phase28_selection": (
                self.eligible_for_phase28_selection
            ),
        }


class MirrorlessMoEH4:
    H = 4
    TOPK = 6

    def __init__(self, base, arm: Arm):
        import cupy as cp

        if not arm.mirrorless:
            raise ValueError(arm)

        self.cp = cp
        self.base = base
        self.rt = base.rt
        self.arm = arm
        self.mk = Phase28MirrorlessKernels()
        self.panel_chunk = cp.empty(
            (ROUTES, base.npanel),
            dtype=cp.int8,
        )

        self.alignment = self.alignment_audit()
        if arm.vector_bytes == 16 and not self.alignment[
            "all_naturally_aligned_16"
        ]:
            raise RuntimeError(
                f"v16 arm requires natural alignment: {self.alignment}"
            )

        # The candidate never reads or writes the sparse device mirror.
        self.freed_mirror_bytes = 0
        old_mirror = getattr(base, "mirrors", None)
        if old_mirror is not None:
            self.freed_mirror_bytes = int(old_mirror.nbytes)
            base.mirrors = None
            del old_mirror
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        self.shared_stream = None
        self.shared_out = None
        self.shared_fork = {}
        self.shared_done = {}

        if arm.shared_overlap:
            self.shared_stream = cp.cuda.Stream(non_blocking=True)
            self.shared_out = cp.empty(
                (self.H, base.hidden),
                cp.float32,
            )
            self.shared_fork = {
                int(layer): cp.cuda.Event()
                for layer in self.rt.moe_layers
            }
            self.shared_done = {
                int(layer): cp.cuda.Event()
                for layer in self.rt.moe_layers
            }

    def __getattr__(self, name):
        return getattr(self.base, name)

    def alignment_audit(self) -> dict:
        rows = int(self.base.hidden)
        rowhalf = rows // 2
        panel_stride = rows + 16 * rowhalf
        layers = []

        for layer in self.rt.moe_layers:
            index = int(layer)
            pointer = int(self.rt.bank[index]["down_base_ptr"])
            layers.append(
                {
                    "layer": index,
                    "down_base_mod16": pointer % 16,
                    "panel_stride_mod16": panel_stride % 16,
                    "rowhalf_mod16": rowhalf % 16,
                    "row_tile_pair_offset_mod16": (128 // 2) % 16,
                }
            )

        return {
            "all_naturally_aligned_16": all(
                row["down_base_mod16"] == 0
                and row["panel_stride_mod16"] == 0
                and row["rowhalf_mod16"] == 0
                and row["row_tile_pair_offset_mod16"] == 0
                for row in layers
            ),
            "rows": rows,
            "rowhalf": rowhalf,
            "panel_stride": panel_stride,
            "layers": layers,
        }

    def _shared_parent(self, layer_data, normed, out):
        base = self.base
        runtime = self.rt

        out.fill(0)
        for token in range(self.H):
            runtime.fused.gemv_into(
                base.shared_act[token],
                layer_data["sh_up_c"],
                layer_data["sh_up_s"],
                normed[token],
                layer_data["sh_up_g"],
                base.shared,
                base.hidden,
                apply_relu2=True,
            )
            runtime.fused.gemv_into(
                out[token],
                layer_data["sh_dn_c"],
                layer_data["sh_dn_s"],
                base.shared_act[token],
                layer_data["sh_dn_g"],
                base.hidden,
                base.shared,
            )

    def _fork_shared(self, layer, layer_data, normed, main_stream):
        base = self.base
        runtime = self.rt

        self.shared_fork[layer].record(main_stream)
        with self.shared_stream:
            self.shared_stream.wait_event(self.shared_fork[layer])
            for token in range(self.H):
                runtime.fused.gemv_into(
                    base.shared_act[token],
                    layer_data["sh_up_c"],
                    layer_data["sh_up_s"],
                    normed[token],
                    layer_data["sh_up_g"],
                    base.shared,
                    base.hidden,
                    apply_relu2=True,
                )
                runtime.fused.gemv_into(
                    self.shared_out[token],
                    layer_data["sh_dn_c"],
                    layer_data["sh_dn_s"],
                    base.shared_act[token],
                    layer_data["sh_dn_g"],
                    base.hidden,
                    base.shared,
                )
            self.shared_done[layer].record(self.shared_stream)

    def __call__(self, layer, normed, out, collect_stats=False):
        cp = self.cp
        base = self.base
        runtime = self.rt
        index = int(layer)
        layer_data = runtime.layer[index]
        bank = runtime.bank[index]
        cache = runtime.cache[index]
        device_cache = base._dev(index)
        main_stream = cp.cuda.get_current_stream()

        if self.arm.shared_overlap:
            self._fork_shared(
                index,
                layer_data,
                normed,
                main_stream,
            )

        # Frozen Phase24 route/cache/group.
        for token in range(self.H):
            begin = token * self.TOPK
            end = begin + self.TOPK
            runtime.k.mv_f32(
                base.rlog[token],
                layer_data["gate_w"],
                normed[token],
                base.nexp,
                base.hidden,
            )
            runtime.fused.route_topk(
                base.rlog[token],
                layer_data["gate_b"],
                base.ids[begin:end],
                base.w[begin:end],
                base.nexp,
                self.TOPK,
                runtime.scaling,
                bad_pick=runtime._bad_pick,
            )

        base.k.cache_assign(
            device_cache,
            base.ids,
            base.slots,
            base.need,
            int(cache["cap"]),
        )
        base.k.group(
            base.ids,
            base.route_group,
            base.group_ids,
            base.group_count,
            base.group_refs,
            base.ngroups,
        )

        runtime.evt[0].record()
        with runtime.copy_stream:
            runtime.copy_stream.wait_event(runtime.evt[0])
            runtime.fused.cache_fetch_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["up_codes"].ctypes.data),
                    np.uint64(bank["up_scales"].ctypes.data),
                    cache["codes"],
                    cache["scales"],
                    base.ids,
                    base.slots,
                    base.need,
                    np.uint64(UP_CODE),
                    np.uint64(UP_SCALE),
                ),
            )
            if index not in base.sres.planes:
                raise RuntimeError(
                    f"missing resident scale plane on layer {index}"
                )
            base.sres.fetch_plane_k(
                (ROUTES, 64),
                (256,),
                (
                    np.uint64(bank["down_base_ptr"]),
                    base.sres.planes[index],
                    base.ids,
                    base.slots,
                    base.need,
                    np.uint64(DOWN_PANEL_BYTES),
                    np.uint64(PLANE_BYTES),
                    np.int32(base.hidden),
                    np.int32(base.npanel),
                ),
            )
            runtime.evt[1].record(runtime.copy_stream)

        if not self.arm.shared_overlap:
            self._shared_parent(layer_data, normed, out)

        main_stream.wait_event(runtime.evt[1])

        # Frozen Phase24 grouped routed-up and sparsity scan.
        for multiplicity in (1, 2, 3, 4):
            base.k.up(
                multiplicity,
                cache["codes"],
                cache["scales"],
                base.slots,
                base.ids,
                device_cache["globals"],
                runtime.fused.e2m1,
                runtime.fused.e4m3,
                normed,
                base.route_act,
                base.inter,
                base.hidden,
                UP_CODE,
                UP_SCALE,
            )

        base.k.scan(
            base.route_act,
            base.group_count,
            base.group_refs,
            base.route_masks,
            base.route_plist,
            base.route_pcount,
            base.union_masks,
            base.union_plist,
            base.union_pcount,
            base.union_nz,
            base.union_nzc,
            base.inter,
        )

        # Mirrorless exact sparse-down.
        if self.arm.mode == "direct_route":
            self.mk.direct_route(
                int(bank["down_base_ptr"]),
                base.sres.planes[index],
                base.slots,
                base.ids,
                device_cache["globals"],
                base.route_act,
                base.route_plist,
                base.route_masks,
                base.route_pcount,
                runtime.fused.e2m1,
                runtime.fused.e4m3,
                base.partials,
                base.hidden,
                base.inter,
                base.nc,
            )
        elif self.arm.mode == "group_chunk":
            self.mk.group_chunk_v16(
                int(bank["down_base_ptr"]),
                base.sres.planes[index],
                base.slots,
                base.group_ids,
                base.group_count,
                base.group_refs,
                device_cache["globals"],
                base.route_act,
                base.route_plist,
                base.route_masks,
                base.route_pcount,
                runtime.fused.e2m1,
                runtime.fused.e4m3,
                base.partials,
                base.hidden,
                base.inter,
                base.nc,
            )
        else:
            self.mk.build_panel_chunk(
                base.route_plist,
                base.route_pcount,
                self.panel_chunk,
                base.inter,
                base.nc,
            )
            self.mk.allchunks(
                self.arm.vector_bytes,
                int(bank["down_base_ptr"]),
                base.sres.planes[index],
                base.slots,
                base.group_ids,
                base.group_count,
                base.group_refs,
                device_cache["globals"],
                base.route_act,
                base.route_masks,
                base.union_plist,
                base.union_pcount,
                self.panel_chunk,
                runtime.fused.e2m1,
                runtime.fused.e4m3,
                base.partials,
                base.hidden,
                base.inter,
                base.nc,
            )

        # Frozen Phase23 reduction and exact route-slot FMA accumulation.
        base.k.reduce(
            base.partials,
            base.route_down,
            base.hidden,
            base.nc,
        )

        if self.arm.shared_overlap:
            main_stream.wait_event(self.shared_done[index])
            cp.copyto(out, self.shared_out)

        base.k.accumulate(
            out,
            base.route_down,
            base.w,
            base.hidden,
        )

        stats = None
        if collect_stats:
            stats = {
                "layer": index,
                "arm": self.arm.as_dict(),
                "mirror_bytes_removed": self.freed_mirror_bytes,
                "alignment": self.alignment,
                "arithmetic_order": (
                    "same route/chunk panel+column FMAs, "
                    "parent reduce, parent slot0..slot5"
                ),
            }
        return None, None, stats


def make_arm(context: int, arm: Arm):
    cfg, _, _, _ = phase28_gate()
    runtime, graph, keep = make_synth(int(context), cfg)

    if arm.name == "p27_control":
        wrapped = Phase27MoE(
            graph.gmoe,
            Phase27Variant(
                gather_y=4,
                batches=3,
                shared_overlap=True,
            ),
        )
    else:
        wrapped = MirrorlessMoEH4(graph.gmoe, arm)

    graph.gmoe = wrapped
    graph.v.moeb = wrapped
    keep = list(keep) + [wrapped]
    return runtime, graph, keep


def timed_arm(runtime, graph, tokens, context, blocks, warmup):
    return timed_synth_blocks(
        runtime,
        graph,
        tokens,
        context,
        blocks,
        warmup,
    )


def capture_arrays(runtime, logits, position, ids, ids_repeat=None):
    import cupy as cp

    arrays = {
        "ids": np.asarray(ids, np.int32),
        "ids_repeat": np.asarray(
            ids if ids_repeat is None else ids_repeat,
            np.int32,
        ),
        "logits": cp.asnumpy(logits).astype(
            np.float32,
            copy=True,
        ),
    }

    for key, value in runtime.ssm.items():
        arrays[f"ssm_{int(key)}"] = cp.asnumpy(value).astype(
            np.float32,
            copy=True,
        )
    for key, value in runtime.conv.items():
        arrays[f"conv_{int(key)}"] = cp.asnumpy(value).astype(
            np.float32,
            copy=True,
        )

    n_kv = int(runtime.n_kv)
    max_ctx = int(runtime.max_ctx)
    head_dim = int(runtime.head_dim)
    for layer in runtime.attn_layers:
        index = int(layer)
        arrays[f"k_{index}"] = cp.asnumpy(
            runtime.kc[index]
            .reshape(n_kv, max_ctx, head_dim)[:, :position, :]
        ).astype(np.float32, copy=True)
        arrays[f"v_{index}"] = cp.asnumpy(
            runtime.vc[index]
            .reshape(n_kv, max_ctx, head_dim)[:, :position, :]
        ).astype(np.float32, copy=True)

    return arrays


def nrmse(a, b):
    aa = np.asarray(a, np.float64)
    bb = np.asarray(b, np.float64)
    return float(
        np.linalg.norm(aa - bb)
        / max(np.linalg.norm(aa), 1e-30)
    )


def compare_npz(parent_path, candidate_path):
    with np.load(parent_path) as parent, np.load(
        candidate_path
    ) as candidate:
        keys = sorted(set(parent.files) & set(candidate.files))

        ssm = max(
            nrmse(parent[key], candidate[key])
            for key in keys
            if key.startswith("ssm_")
        )
        conv = max(
            nrmse(parent[key], candidate[key])
            for key in keys
            if key.startswith("conv_")
        )
        kv = max(
            [
                nrmse(parent[key], candidate[key])
                for key in keys
                if key.startswith("k_")
                or key.startswith("v_")
            ]
            or [0.0]
        )
        logits = nrmse(
            parent["logits"],
            candidate["logits"],
        )
        ids_exact = bool(
            np.array_equal(
                parent["ids"],
                candidate["ids"],
            )
        )
        deterministic = bool(
            np.array_equal(
                candidate["ids"],
                candidate["ids_repeat"],
            )
        )
        finite = all(
            np.isfinite(candidate[key]).all()
            for key in keys
        )

    state = {
        "max_ssm_nrmse": ssm,
        "max_conv_nrmse": conv,
        "max_kv_nrmse": kv,
        "logits_nrmse": logits,
    }
    gates = {
        "ids_exact": ids_exact,
        "candidate_deterministic_ids": deterministic,
        "ssm": ssm <= 5e-5,
        "conv": conv <= 1e-5,
        "kv": kv <= 5e-6,
        "logits": logits <= 5e-4,
        "finite": bool(finite),
    }
    return state, gates


def robust_cv(values):
    array = np.asarray(values, np.float64)
    center = float(np.median(array))
    return float(
        1.4826
        * np.median(np.abs(array - center))
        / max(abs(center), 1e-30)
    )
