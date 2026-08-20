from __future__ import annotations

import json
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase24_sres_grouped import GroupedScaleResidentKernels
from s100_phase28_common import RESULTS, phase28_gate
from s100_phase28_kernels import (
    Phase28MirrorlessKernels,
    ROUTES,
    GROUPS,
    MAXM,
    NCHUNKS,
    ROW_TILE,
)

OUT = RESULTS / "S100_PHASE28_PREFLIGHT.json"

ROWS = 256
INTER = 256
NPANEL = INTER // 16
ROWHALF = ROWS // 2
PANEL_STRIDE = ROWS + 16 * ROWHALF
PANEL_BYTES_TEST = NPANEL * PANEL_STRIDE
PLANE_BYTES_TEST = NPANEL * ROWS


def group_fixture():
    # 4 + 3 + 2 + fifteen M1 groups = all 24 routes.
    counts = [4, 3, 2] + [1] * 15

    group_count = np.zeros(GROUPS, np.int32)
    group_ids = np.full(GROUPS, -1, np.int32)
    group_refs = np.full(GROUPS * MAXM, -1, np.int32)
    route_group = np.full(ROUTES, -1, np.int32)
    ids = np.empty(ROUTES, np.int32)
    slots = np.empty(ROUTES, np.int32)

    route = 0
    for group, count in enumerate(counts):
        group_count[group] = count
        group_ids[group] = group

        for member in range(count):
            group_refs[group * MAXM + member] = route
            route_group[route] = group
            ids[route] = group
            slots[route] = group
            route += 1

    if route != ROUTES:
        raise AssertionError(route)

    return {
        "counts": counts,
        "active_groups": len(counts),
        "group_count": group_count,
        "group_ids": group_ids,
        "group_refs": group_refs,
        "route_group": route_group,
        "ids": ids,
        "slots": slots,
    }


def sparsity_fixture(fixture):
    rng = np.random.default_rng(20260820)

    route_masks = np.zeros((ROUTES, NPANEL), np.uint32)
    route_plist = np.zeros((ROUTES, NPANEL), np.int32)
    route_pcount = np.zeros(ROUTES, np.int32)
    activation = np.zeros((ROUTES, INTER), np.float32)

    for route in range(ROUTES):
        panel_count = 7 + route % 6
        panels = np.sort(
            rng.choice(
                NPANEL,
                size=panel_count,
                replace=False,
            )
        )
        route_pcount[route] = panel_count
        route_plist[route, :panel_count] = panels

        for panel in panels:
            columns = np.sort(
                rng.choice(
                    16,
                    size=3 + (route + int(panel)) % 5,
                    replace=False,
                )
            )
            mask = 0
            for column in columns:
                column = int(column)
                mask |= 1 << column
                activation[
                    route,
                    int(panel) * 16 + column,
                ] = np.float32(
                    0.005 * (route + 1)
                    + 0.0002 * (int(panel) + 1)
                    + 0.00001 * (column + 1)
                )
            route_masks[route, int(panel)] = mask

    union_masks = np.zeros((GROUPS, NPANEL), np.uint32)
    union_plist = np.zeros((GROUPS, NPANEL), np.int32)
    union_pcount = np.zeros(GROUPS, np.int32)
    union_nz = np.zeros((GROUPS, INTER), np.int32)
    union_nzc = np.zeros(GROUPS, np.int32)

    for group in range(fixture["active_groups"]):
        count = int(fixture["group_count"][group])

        for member in range(count):
            route = int(
                fixture["group_refs"][group * MAXM + member]
            )
            union_masks[group] |= route_masks[route]

        panels = np.nonzero(union_masks[group])[0]
        union_pcount[group] = len(panels)
        union_plist[group, : len(panels)] = panels

        columns = []
        for panel in panels:
            mask = int(union_masks[group, panel])
            for column in range(16):
                if mask & (1 << column):
                    columns.append(int(panel) * 16 + column)

        union_nzc[group] = len(columns)
        union_nz[group, : len(columns)] = columns

    expected_panel_chunk = np.full(
        (ROUTES, NPANEL),
        -1,
        np.int8,
    )
    for route in range(ROUTES):
        count = int(route_pcount[route])
        for pi in range(count):
            panel = int(route_plist[route, pi])
            expected_panel_chunk[route, panel] = pi % NCHUNKS

    return {
        "route_masks": route_masks,
        "route_plist": route_plist,
        "route_pcount": route_pcount,
        "activation": activation,
        "union_masks": union_masks,
        "union_plist": union_plist,
        "union_pcount": union_pcount,
        "union_nz": union_nz,
        "union_nzc": union_nzc,
        "expected_panel_chunk": expected_panel_chunk,
    }


def main() -> int:
    payload = {
        "kind": "s100_phase28_preflight",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "synthetic partial-buffer exactness; no model throughput"
        ),
    }

    try:
        phase28_gate()
        import cupy as cp

        fixture = group_fixture()
        sparse = sparsity_fixture(fixture)

        mirrorless = Phase28MirrorlessKernels()
        parent = GroupedScaleResidentKernels()

        rng = np.random.default_rng(28)

        down_host = rng.integers(
            0,
            256,
            size=fixture["active_groups"] * PANEL_BYTES_TEST,
            dtype=np.uint8,
        )

        # The published Phase27 correction established that real routed scale
        # bytes occupy the finite Lightning range 62..124.
        planes_host = rng.integers(
            62,
            125,
            size=GROUPS * PLANE_BYTES_TEST,
            dtype=np.uint8,
        )

        globals_host = np.ones(GROUPS * 2, np.float32)
        globals_host[0::2] = np.linspace(
            0.75,
            1.25,
            GROUPS,
            dtype=np.float32,
        )

        down = cp.asarray(down_host)
        planes = cp.asarray(planes_host)
        group_count = cp.asarray(fixture["group_count"])
        group_ids = cp.asarray(fixture["group_ids"])
        group_refs = cp.asarray(fixture["group_refs"])
        route_group = cp.asarray(fixture["route_group"])
        ids = cp.asarray(fixture["ids"])
        slots = cp.asarray(fixture["slots"])

        route_masks = cp.asarray(sparse["route_masks"])
        route_plist = cp.asarray(sparse["route_plist"])
        route_pcount = cp.asarray(sparse["route_pcount"])
        activation = cp.asarray(sparse["activation"])
        union_masks = cp.asarray(sparse["union_masks"])
        union_plist = cp.asarray(sparse["union_plist"])
        union_pcount = cp.asarray(sparse["union_pcount"])
        union_nz = cp.asarray(sparse["union_nz"])
        union_nzc = cp.asarray(sparse["union_nzc"])
        globals_dev = cp.asarray(globals_host)

        # The same lookup tables are passed to parent and candidates; finite
        # exact partial equality is the criterion.
        e2 = cp.asarray(
            np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        )
        e4 = cp.asarray(
            np.linspace(0.125, 1.875, 256, dtype=np.float32)
        )

        mirror = cp.zeros(
            GROUPS * PANEL_BYTES_TEST,
            cp.uint8,
        )

        nan = np.float32(np.nan)
        reference = cp.full(
            ROUTES * NCHUNKS * ROWS,
            nan,
            cp.float32,
        )
        direct = cp.full_like(reference, nan)
        group_chunk = cp.full_like(reference, nan)
        allchunks_v4 = cp.full_like(reference, nan)
        allchunks_v16 = cp.full_like(reference, nan)
        panel_chunk = cp.empty(
            (ROUTES, NPANEL),
            cp.int8,
        )

        # Exact Phase24 parent using synthetic record strides.
        parent.gather_k(
            (GROUPS, 32),
            (256,),
            (
                down,
                group_ids,
                group_count,
                union_nz,
                union_nzc,
                mirror,
                np.uint64(PANEL_BYTES_TEST),
                np.int32(ROWS),
                np.int32(INTER),
            ),
        )
        parent.down_k(
            (
                (ROWS + 127) // 128,
                ROUTES * NCHUNKS,
            ),
            (128,),
            (
                mirror,
                planes,
                slots,
                ids,
                route_group,
                globals_dev,
                activation,
                route_plist,
                route_masks,
                route_pcount,
                e2,
                e4,
                reference,
                np.uint64(PANEL_BYTES_TEST),
                np.uint64(PLANE_BYTES_TEST),
                np.int32(ROWS),
                np.int32(INTER),
                np.int32(NCHUNKS),
            ),
        )

        # Direct exact zero-mirror control.
        mirrorless.f["mirrorless_direct_route"](
            (
                (ROWS + 127) // 128,
                ROUTES * NCHUNKS,
            ),
            (128,),
            (
                down,
                planes,
                slots,
                ids,
                globals_dev,
                activation,
                route_plist,
                route_masks,
                route_pcount,
                e2,
                e4,
                direct,
                np.uint64(PANEL_BYTES_TEST),
                np.uint64(PLANE_BYTES_TEST),
                np.int32(ROWS),
                np.int32(INTER),
                np.int32(NCHUNKS),
            ),
        )

        for multiplicity in (1, 2, 3, 4):
            mirrorless.f[
                f"mirrorless_group_chunk_m{multiplicity}_v16"
            ](
                (
                    ROWS // ROW_TILE,
                    GROUPS * NCHUNKS,
                ),
                (ROW_TILE,),
                (
                    down,
                    planes,
                    slots,
                    group_ids,
                    group_count,
                    group_refs,
                    globals_dev,
                    activation,
                    route_plist,
                    route_masks,
                    route_pcount,
                    e2,
                    e4,
                    group_chunk,
                    np.uint64(PANEL_BYTES_TEST),
                    np.uint64(PLANE_BYTES_TEST),
                    np.int32(ROWS),
                    np.int32(INTER),
                    np.int32(NCHUNKS),
                ),
            )

        mirrorless.build_panel_chunk(
            route_plist,
            route_pcount,
            panel_chunk,
            INTER,
            NCHUNKS,
        )

        for vector_bytes, output in (
            (4, allchunks_v4),
            (16, allchunks_v16),
        ):
            for multiplicity in (1, 2, 3, 4):
                mirrorless.f[
                    "mirrorless_allchunks_"
                    f"m{multiplicity}_v{vector_bytes}"
                ](
                    (
                        ROWS // ROW_TILE,
                        GROUPS,
                    ),
                    (ROW_TILE,),
                    (
                        down,
                        planes,
                        slots,
                        group_ids,
                        group_count,
                        group_refs,
                        globals_dev,
                        activation,
                        route_masks,
                        union_plist,
                        union_pcount,
                        panel_chunk,
                        e2,
                        e4,
                        output,
                        np.uint64(PANEL_BYTES_TEST),
                        np.uint64(PLANE_BYTES_TEST),
                        np.int32(ROWS),
                        np.int32(INTER),
                        np.int32(NCHUNKS),
                    ),
                )

        cp.cuda.get_current_stream().synchronize()

        bit_exact = {
            "direct_route": bool(
                cp.asnumpy(cp.array_equal(reference, direct))
            ),
            "group_chunk_v16": bool(
                cp.asnumpy(
                    cp.array_equal(reference, group_chunk)
                )
            ),
            "group_allchunks_v4": bool(
                cp.asnumpy(
                    cp.array_equal(reference, allchunks_v4)
                )
            ),
            "group_allchunks_v16": bool(
                cp.asnumpy(
                    cp.array_equal(reference, allchunks_v16)
                )
            ),
        }

        finite = {
            "reference": bool(
                cp.asnumpy(cp.isfinite(reference).all())
            ),
            "direct_route": bool(
                cp.asnumpy(cp.isfinite(direct).all())
            ),
            "group_chunk_v16": bool(
                cp.asnumpy(cp.isfinite(group_chunk).all())
            ),
            "group_allchunks_v4": bool(
                cp.asnumpy(cp.isfinite(allchunks_v4).all())
            ),
            "group_allchunks_v16": bool(
                cp.asnumpy(cp.isfinite(allchunks_v16).all())
            ),
        }

        panel_chunk_exact = bool(
            np.array_equal(
                cp.asnumpy(panel_chunk),
                sparse["expected_panel_chunk"],
            )
        )

        alignment = {
            "down_device_pointer_mod16": int(
                down.data.ptr
            ) % 16,
            "panel_stride_mod16": PANEL_STRIDE % 16,
            "rowhalf_mod16": ROWHALF % 16,
            "row_tile_pair_offset_mod16": (
                ROW_TILE // 2
            ) % 16,
        }
        naturally_aligned = all(
            value == 0 for value in alignment.values()
        )

        green = bool(
            all(bit_exact.values())
            and all(finite.values())
            and panel_chunk_exact
            and naturally_aligned
        )

        payload.update(
            {
                "status": "measured",
                "group_count_fixture": fixture["counts"],
                "bit_exact": bit_exact,
                "finite": finite,
                "panel_chunk_exact": panel_chunk_exact,
                "alignment": alignment,
                "naturally_aligned_16": naturally_aligned,
                "kernel_attributes": mirrorless.attributes(),
                "PREFLIGHT_GREEN": green,
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "PREFLIGHT_GREEN": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "bit_exact": payload.get("bit_exact"),
                "finite": payload.get("finite"),
                "panel_chunk_exact": payload.get(
                    "panel_chunk_exact"
                ),
                "alignment": payload.get("alignment"),
                "PREFLIGHT_GREEN": payload.get(
                    "PREFLIGHT_GREEN"
                ),
                "error": (
                    payload.get("error") or {}
                ).get("message"),
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0 if payload.get("PREFLIGHT_GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
