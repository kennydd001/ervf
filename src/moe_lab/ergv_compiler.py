"""Restricted exact-reduction graph virtualization (ERGV) prototype.

The module deliberately models only the 256-virtual-accumulator reduction used
by the local Q8/Q5 GEMV kernels.  It separates the logical ordered arithmetic
graph from a physical CUDA schedule, then verifies the two mechanically before
emitting CUDA source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np


Family = Literal["q8", "q5"]
Stage = Literal["lane_local", "cross_warp_shared", "warp_shuffle"]
WIDTHS = (4, 8, 16, 32, 64)
VIRTUAL_ACCUMULATORS = 256
REDUCTION_STRIDES = (128, 64, 32, 16, 8, 4, 2, 1)
FMA_POLICY = "cuda-default-contract-same-reference-and-candidate"
FINAL_CAST = "round_bf16_rne"


@dataclass(frozen=True)
class SourceLoad:
    """One logical source work-item in its required serial order."""

    work_index: int
    scalar_columns: tuple[int, ...]


@dataclass(frozen=True)
class LogicalAccumulator:
    accumulator_id: int
    source_order: tuple[SourceLoad, ...]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    opcode: Literal["accumulator", "ordered_add", "round"]
    inputs: tuple[str, ...] = ()
    accumulator_id: int | None = None
    rounding: str | None = None


@dataclass(frozen=True)
class ExactReductionIR:
    family: Family
    columns: int
    virtual_accumulators: int
    accumulators: tuple[LogicalAccumulator, ...]
    nodes: tuple[GraphNode, ...]
    root: str
    fma_policy: str
    final_cast: str

    def node_map(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    def canonical_payload(self) -> dict:
        return {
            "family": self.family,
            "columns": self.columns,
            "virtual_accumulators": self.virtual_accumulators,
            "fma_policy": self.fma_policy,
            "final_cast": self.final_cast,
            "accumulators": [
                {
                    "id": acc.accumulator_id,
                    "source_order": [
                        [load.work_index, list(load.scalar_columns)]
                        for load in acc.source_order
                    ],
                }
                for acc in self.accumulators
            ],
            "nodes": [
                {
                    "id": node.node_id,
                    "opcode": node.opcode,
                    "inputs": list(node.inputs),
                    "accumulator_id": node.accumulator_id,
                    "rounding": node.rounding,
                }
                for node in self.nodes
            ],
            "root": self.root,
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LaneAssignment:
    accumulator_id: int
    physical_lane: int
    virtual_slot: int


@dataclass(frozen=True)
class ScheduledAdd:
    node_id: str
    left: str
    right: str
    stride: int
    logical_index: int
    physical_lane: int
    stage: Stage


@dataclass(frozen=True)
class ReductionSchedule:
    family: Family
    columns: int
    width: int
    rows_per_block: int
    virtual_accumulators_per_lane: int
    lane_mapping: tuple[LaneAssignment, ...]
    nodes: tuple[GraphNode, ...]
    scheduled_adds: tuple[ScheduledAdd, ...]
    root: str
    source_orders: tuple[tuple[SourceLoad, ...], ...]
    fma_policy: str
    final_cast: str
    vector_load_bytes: int = 1
    scale_broadcast: bool = False
    activation_staging: bool = False
    register_cap: int | None = None

    def node_map(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reasons: tuple[str, ...]
    compared_nodes: int
    reference_digest: str


@dataclass(frozen=True)
class KernelChoice:
    family: Family
    projection: str
    width: int
    rows_per_block: int
    vector_load_bytes: int = 1
    scale_broadcast: bool = False
    activation_staging: bool = False
    register_cap: int | None = None


@dataclass(frozen=True)
class ManualAudit:
    name: str
    widths: tuple[int, ...]
    structural_contract_pass: bool
    reasons: tuple[str, ...]


def _source_loads(family: Family, columns: int, accumulator_id: int) -> tuple[SourceLoad, ...]:
    if columns <= 0:
        raise ValueError("columns must be positive")
    if family == "q8":
        return tuple(
            SourceLoad(work_index=column, scalar_columns=(column,))
            for column in range(accumulator_id, columns, VIRTUAL_ACCUMULATORS)
        )
    if columns % 8:
        raise ValueError("Q5 columns must be divisible by eight")
    packs = columns // 8
    return tuple(
        SourceLoad(work_index=pack, scalar_columns=tuple(range(pack * 8, pack * 8 + 8)))
        for pack in range(accumulator_id, packs, VIRTUAL_ACCUMULATORS)
    )


def build_exact_reduction_ir(family: Family, columns: int) -> ExactReductionIR:
    if family not in ("q8", "q5"):
        raise ValueError(f"unsupported family: {family}")
    accumulators = tuple(
        LogicalAccumulator(index, _source_loads(family, columns, index))
        for index in range(VIRTUAL_ACCUMULATORS)
    )
    nodes: list[GraphNode] = [
        GraphNode(f"ref_acc_{index:03d}", "accumulator", accumulator_id=index)
        for index in range(VIRTUAL_ACCUMULATORS)
    ]
    active = [node.node_id for node in nodes]
    for stride in REDUCTION_STRIDES:
        next_active: list[str] = []
        for index in range(stride):
            node_id = f"ref_add_s{stride:03d}_{index:03d}"
            nodes.append(
                GraphNode(node_id, "ordered_add", (active[index], active[index + stride]))
            )
            next_active.append(node_id)
        active = next_active
    root = "ref_round_bf16"
    nodes.append(GraphNode(root, "round", (active[0],), rounding=FINAL_CAST))
    return ExactReductionIR(
        family=family,
        columns=columns,
        virtual_accumulators=VIRTUAL_ACCUMULATORS,
        accumulators=accumulators,
        nodes=tuple(nodes),
        root=root,
        fma_policy=FMA_POLICY,
        final_cast=FINAL_CAST,
    )


def _stage_for(width: int, stride: int) -> Stage:
    if stride >= width:
        return "lane_local"
    if width == 64 and stride == 32:
        return "cross_warp_shared"
    return "warp_shuffle"


def schedule_exact_reduction(
    ir: ExactReductionIR,
    width: int,
    *,
    vector_load_bytes: int = 1,
    scale_broadcast: bool = False,
    activation_staging: bool = False,
    register_cap: int | None = None,
) -> ReductionSchedule:
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}")
    if ir.virtual_accumulators != VIRTUAL_ACCUMULATORS:
        raise ValueError("restricted compiler requires 256 virtual accumulators")
    if vector_load_bytes not in (1, 4, 8, 16):
        raise ValueError("unsupported vector load width")

    lane_mapping = tuple(
        LaneAssignment(index, index % width, index // width)
        for index in range(VIRTUAL_ACCUMULATORS)
    )
    nodes: list[GraphNode] = [
        GraphNode(f"w{width}_acc_{index:03d}", "accumulator", accumulator_id=index)
        for index in range(VIRTUAL_ACCUMULATORS)
    ]
    active = [node.node_id for node in nodes]
    scheduled: list[ScheduledAdd] = []
    for stride in REDUCTION_STRIDES:
        next_active: list[str] = []
        stage = _stage_for(width, stride)
        for index in range(stride):
            node_id = f"w{width}_add_s{stride:03d}_{index:03d}"
            left, right = active[index], active[index + stride]
            nodes.append(GraphNode(node_id, "ordered_add", (left, right)))
            scheduled.append(
                ScheduledAdd(
                    node_id=node_id,
                    left=left,
                    right=right,
                    stride=stride,
                    logical_index=index,
                    physical_lane=index % width,
                    stage=stage,
                )
            )
            next_active.append(node_id)
        active = next_active
    root = f"w{width}_round_bf16"
    nodes.append(GraphNode(root, "round", (active[0],), rounding=ir.final_cast))
    return ReductionSchedule(
        family=ir.family,
        columns=ir.columns,
        width=width,
        rows_per_block=VIRTUAL_ACCUMULATORS // width,
        virtual_accumulators_per_lane=VIRTUAL_ACCUMULATORS // width,
        lane_mapping=lane_mapping,
        nodes=tuple(nodes),
        scheduled_adds=tuple(scheduled),
        root=root,
        source_orders=tuple(acc.source_order for acc in ir.accumulators),
        fma_policy=ir.fma_policy,
        final_cast=ir.final_cast,
        vector_load_bytes=vector_load_bytes,
        scale_broadcast=scale_broadcast,
        activation_staging=activation_staging,
        register_cap=register_cap,
    )


def verify_graph_isomorphism(
    reference: ExactReductionIR, candidate: ReductionSchedule
) -> VerificationResult:
    reasons: list[str] = []
    if (reference.family, reference.columns) != (candidate.family, candidate.columns):
        reasons.append("family/column contract differs")
    if reference.fma_policy != candidate.fma_policy:
        reasons.append("FMA policy differs")
    if reference.final_cast != candidate.final_cast:
        reasons.append("final cast differs")
    expected_orders = tuple(acc.source_order for acc in reference.accumulators)
    if expected_orders != candidate.source_orders:
        reasons.append("source load order differs")

    expected_mapping = tuple(
        LaneAssignment(index, index % candidate.width, index // candidate.width)
        for index in range(VIRTUAL_ACCUMULATORS)
    )
    if candidate.lane_mapping != expected_mapping:
        reasons.append("virtual-to-physical lane mapping differs")

    cross_warp = [item for item in candidate.scheduled_adds if item.stage == "cross_warp_shared"]
    if candidate.width == 64:
        if len(cross_warp) != 32 or any(item.stride != 32 for item in cross_warp):
            reasons.append("width 64 lacks the exact 32-node stride-32 cross-warp phase")
    elif cross_warp:
        reasons.append("cross-warp phase present below width 64")

    ref_nodes = reference.node_map()
    candidate_nodes = candidate.node_map()
    visited: set[tuple[str, str]] = set()

    def compare(reference_id: str, candidate_id: str) -> None:
        pair = (reference_id, candidate_id)
        if pair in visited:
            return
        visited.add(pair)
        left = ref_nodes.get(reference_id)
        right = candidate_nodes.get(candidate_id)
        if left is None or right is None:
            reasons.append("graph references a missing node")
            return
        if left.opcode != right.opcode:
            reasons.append(f"opcode mismatch at {reference_id}")
            return
        if left.opcode == "accumulator":
            if left.accumulator_id != right.accumulator_id:
                reasons.append(f"accumulator identity mismatch at {reference_id}")
            return
        if left.opcode == "round" and left.rounding != right.rounding:
            reasons.append(f"rounding mismatch at {reference_id}")
        if len(left.inputs) != len(right.inputs):
            reasons.append(f"arity mismatch at {reference_id}")
            return
        for ref_input, candidate_input in zip(left.inputs, right.inputs, strict=True):
            compare(ref_input, candidate_input)

    compare(reference.root, candidate.root)
    if len(visited) != len(reference.nodes):
        reasons.append(
            f"reachable-node count differs: {len(visited)} != {len(reference.nodes)}"
        )
    return VerificationResult(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        compared_nodes=len(visited),
        reference_digest=reference.digest(),
    )


def mutate_schedule_node(
    schedule: ReductionSchedule, node_id: str, *, inputs: tuple[str, ...]
) -> ReductionSchedule:
    """Test helper: return an immutable schedule with one changed graph node."""

    changed = tuple(
        replace(node, inputs=inputs) if node.node_id == node_id else node
        for node in schedule.nodes
    )
    return replace(schedule, nodes=changed)


def evaluate_reference_tree(ir: ExactReductionIR, leaf_values: Sequence[np.float32]) -> np.float32:
    values = np.asarray(leaf_values, dtype=np.float32)
    if values.shape != (VIRTUAL_ACCUMULATORS,):
        raise ValueError("expected 256 leaf values")
    active = values.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for stride in REDUCTION_STRIDES:
            active = np.asarray(
                [np.float32(active[index] + active[index + stride]) for index in range(stride)],
                dtype=np.float32,
            )
    return np.float32(active[0])


def evaluate_physical_schedule(
    schedule: ReductionSchedule, leaf_values: Sequence[np.float32]
) -> np.float32:
    values = np.asarray(leaf_values, dtype=np.float32)
    if values.shape != (VIRTUAL_ACCUMULATORS,):
        raise ValueError("expected 256 leaf values")
    width = schedule.width
    partial = np.empty((width, VIRTUAL_ACCUMULATORS // width), dtype=np.float32)
    for assignment in schedule.lane_mapping:
        partial[assignment.physical_lane, assignment.virtual_slot] = values[
            assignment.accumulator_id
        ]
    with np.errstate(over="ignore", invalid="ignore"):
        for stride in REDUCTION_STRIDES:
            if stride < width:
                break
            pairs_per_lane = stride // width
            for lane in range(width):
                old = partial[lane].copy()
                for index in range(pairs_per_lane):
                    partial[lane, index] = np.float32(old[index] + old[index + pairs_per_lane])
        lane_values = partial[:, 0].copy()
        if width == 64:
            before = lane_values.copy()
            for lane in range(32):
                lane_values[lane] = np.float32(before[lane] + before[lane + 32])
            first_shuffle = 16
        else:
            first_shuffle = width // 2
        offset = first_shuffle
        while offset:
            before = lane_values.copy()
            for lane in range(offset):
                lane_values[lane] = np.float32(before[lane] + before[lane + offset])
            offset //= 2
    return np.float32(lane_values[0])


def _extract_define_widths(source: str, macro: str) -> set[int]:
    return {int(value) for value in re.findall(rf"{re.escape(macro)}\((\d+)\)", source)}


def audit_manual_p7_source(source: str) -> ManualAudit:
    reasons: list[str] = []
    q8 = _extract_define_widths(source, "DEFINE_Q8")
    gate = _extract_define_widths(source, "DEFINE_Q5_GATE")
    down = _extract_define_widths(source, "DEFINE_Q5_DOWN")
    widths = tuple(sorted(q8 & gate & down))
    if widths != (8, 16, 32):
        reasons.append(f"manual P7 width set differs: {widths}")
    required = (
        "const int VIRTUAL = 256 / WIDTH;",
        "int tid = lane + WIDTH * virtual_index;",
        "for (int stride = 128; stride >= WIDTH; stride >>= 1)",
        "partial[index] += partial[index + stride / WIDTH]",
        "for (int offset = WIDTH / 2; offset > 0; offset >>= 1)",
        "round_bf16(value)",
    )
    for marker in required:
        if marker not in source:
            reasons.append(f"manual P7 marker missing: {marker}")
    return ManualAudit("P7", widths, not reasons, tuple(reasons))


def audit_manual_n1c_source(source: str) -> ManualAudit:
    reasons: list[str] = []
    q8 = _extract_define_widths(source, "DEFINE_Q8_N1C")
    gate = _extract_define_widths(source, "DEFINE_Q5_GATE_N1C")
    down = _extract_define_widths(source, "DEFINE_Q5_DOWN_N1C")
    ordinary = q8 & gate & down
    has_64 = all(
        marker in source
        for marker in ("q8_n1c_64(", "q5_gate_up_n1c_64(", "q5_down_n1c_64(")
    )
    widths = tuple(sorted(ordinary | ({64} if has_64 else set())))
    if widths != WIDTHS:
        reasons.append(f"manual N1C width set differs: {widths}")
    required_64 = (
        "partial[0] += partial[2];",
        "partial[1] += partial[3];",
        "partial[0] += partial[1];",
        "value += scratch[threadIdx.x + 32]",
        "for (int offset = 16; offset > 0; offset >>= 1)",
    )
    for marker in required_64:
        if marker not in source:
            reasons.append(f"manual N1C width-64 marker missing: {marker}")
    return ManualAudit("N1C", widths, not reasons, tuple(reasons))


def n1c_frozen_choices() -> tuple[KernelChoice, ...]:
    widths: Mapping[tuple[Family, str], int] = {
        ("q8", "head"): 16,
        ("q8", "k"): 64,
        ("q8", "o"): 16,
        ("q8", "q"): 16,
        ("q8", "router"): 64,
        ("q8", "v"): 64,
        ("q5", "gate_up"): 8,
        ("q5", "down"): 8,
    }
    return tuple(
        KernelChoice(family, projection, width, VIRTUAL_ACCUMULATORS // width)
        for (family, projection), width in widths.items()
    )


def _q8_load_body() -> str:
    return """        for (int col = tid; col < cols; col += 256) {
            float scale = bf16_to_float(scales[row * groups + (col >> 7)]);
            float weight = round_bf16(((float)codes[(long long)row * cols + col]) * scale);
            sum += weight * x[col];
        }"""


def _q5_load_body() -> str:
    return """        for (int pack = tid; pack < packs; pack += 256) {
            const unsigned char* source = packed + ((long long)row * packs + pack) * 5LL;
            unsigned long long word = ((unsigned long long)source[0])
                | ((unsigned long long)source[1] << 8)
                | ((unsigned long long)source[2] << 16)
                | ((unsigned long long)source[3] << 24)
                | ((unsigned long long)source[4] << 32);
            int column = pack << 3;
            float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                int code = ((word >> (item * 5)) & 31ULL) - 15;
                float weight = round_bf16(((float)code) * scale);
                sum += weight * x[column + item];
            }
        }"""


def generate_cuda_row_reducer(ir: ExactReductionIR, width: int) -> str:
    """Emit a standalone row reducer/pre-reducer from a verified schedule."""

    schedule = schedule_exact_reduction(ir, width)
    verification = verify_graph_isomorphism(ir, schedule)
    if not verification.passed:
        raise ValueError(f"refusing to emit non-isomorphic schedule: {verification.reasons}")
    family = ir.family
    args = (
        "const float* x, const signed char* codes, const unsigned short* scales, "
        "int row, int cols, int lane, unsigned mask"
        if family == "q8"
        else "const float* x, const unsigned char* packed, const unsigned short* scales, "
        "int row, int cols, int lane, unsigned mask"
    )
    setup = "int groups = cols >> 7;" if family == "q8" else "int packs = cols >> 3;\n    int groups = cols >> 7;"
    load_body = _q8_load_body() if family == "q8" else _q5_load_body()
    name = f"ergv_{family}_row_w{width}"
    virtual = VIRTUAL_ACCUMULATORS // width
    if width <= 32:
        return f"""template<int ERGV_SENTINEL = 0>
__device__ __forceinline__ float {name}({args}) {{
    const int WIDTH = {width};
    const int VIRTUAL = {virtual};
    float partial[VIRTUAL];
    {setup}
    #pragma unroll
    for (int virtual_index = 0; virtual_index < VIRTUAL; ++virtual_index) {{
        int tid = lane + WIDTH * virtual_index;
        float sum = 0.0f;
{load_body}
        partial[virtual_index] = sum;
    }}
    #pragma unroll
    for (int stride = 128; stride >= WIDTH; stride >>= 1) {{
        #pragma unroll
        for (int index = 0; index < stride / WIDTH; ++index)
            partial[index] += partial[index + stride / WIDTH];
    }}
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(mask, value, offset, WIDTH);
    return value;
}}
"""
    return f"""template<int ERGV_SENTINEL = 0>
__device__ __forceinline__ float {name}_pre64({args}) {{
    float partial[4];
    {setup}
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 4; ++virtual_index) {{
        int tid = lane + 64 * virtual_index;
        float sum = 0.0f;
{load_body}
        partial[virtual_index] = sum;
    }}
    partial[0] += partial[2];
    partial[1] += partial[3];
    partial[0] += partial[1];
    return partial[0];
}}
// Width 64 completion contract: store each pre64 value in shared memory,
// add lane+32 into lanes 0..31, then shuffle offsets 16,8,4,2,1.
"""


def generate_cuda_source(specs: Iterable[tuple[ExactReductionIR, int]]) -> str:
    parts = [
        "// Generated by the restricted ERGV compiler.\n",
        "// Requires bf16_to_float() and round_bf16() from the enclosing runtime.\n\n",
    ]
    # Reducers take ``cols`` at runtime, so one family/width definition covers
    # every compatible matrix shape.  Deduplicating by columns would emit the
    # same C++ symbol twice when a source contains (for example) Q5 gate/up and
    # down shapes.
    seen: set[tuple[Family, int]] = set()
    for ir, width in specs:
        key = (ir.family, width)
        if key in seen:
            continue
        seen.add(key)
        parts.append(generate_cuda_row_reducer(ir, width))
        parts.append("\n")
    return "".join(parts)


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


__all__ = [
    "ExactReductionIR",
    "FMA_POLICY",
    "FINAL_CAST",
    "KernelChoice",
    "ManualAudit",
    "REDUCTION_STRIDES",
    "ReductionSchedule",
    "SourceLoad",
    "VerificationResult",
    "WIDTHS",
    "audit_manual_n1c_source",
    "audit_manual_p7_source",
    "build_exact_reduction_ir",
    "evaluate_physical_schedule",
    "evaluate_reference_tree",
    "generate_cuda_row_reducer",
    "generate_cuda_source",
    "mutate_schedule_node",
    "n1c_frozen_choices",
    "schedule_exact_reduction",
    "source_sha256",
    "verify_graph_isomorphism",
]
