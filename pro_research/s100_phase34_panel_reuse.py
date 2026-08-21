from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
import traceback

import numpy as np

from common import REPO, utc_now, write_json_atomic
from s100_phase21_common import load_trace, prefill_to, release
from s100_phase25_common import expected_for_h8
from s100_phase32_common import make_candidate


RESULTS = REPO / "pro_research" / "results" / "s100_phase34"
OUT = RESULTS / "S100_PHASE34_PANEL_REUSE.json"
TRACE = RESULTS / "S100_PHASE34_PANEL_KEYS.npz"
CONTEXT = 1024
BLOCKS = 16
PANEL_BYTES = 32 * 1024
CAPACITY_MIB = (32, 64, 96, 128)


class PanelTracer:
    def __init__(self, base):
        self.base = base
        self.cp = base.cp
        self.block = -1
        self.records = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        result = self.base(layer, normed, out, collect_stats)
        cp = self.cp
        cp.cuda.get_current_stream().synchronize()
        self.base.shared_stream.synchronize()
        group_count = cp.asnumpy(self.base.group_count)
        active = np.flatnonzero(group_count > 0)
        group_ids = cp.asnumpy(self.base.group_ids)
        counts = cp.asnumpy(self.base.union_pcount)
        panels = cp.asnumpy(self.base.union_plist).reshape(
            self.base.group_count.size, self.base.npanel
        )
        keys = []
        for group in active.tolist():
            expert = int(group_ids[group])
            count = int(counts[group])
            if count < 0 or count > self.base.npanel:
                raise RuntimeError(f"invalid union panel count {count}")
            for panel in panels[group, :count].tolist():
                keys.append((int(layer), expert, int(panel)))
        if len(keys) != len(set(keys)):
            raise RuntimeError(
                f"duplicate transfer key in block={self.block} layer={layer}"
            )
        self.records.append(
            {"block": self.block, "layer": int(layer), "keys": keys}
        )
        return result


def simulate_lru(stream, capacity_entries: int, first_block_accesses: int) -> dict:
    cache = OrderedDict()
    hits = 0
    misses = 0
    steady_hits = 0
    steady_accesses = 0
    for block, key in stream:
        hit = key in cache
        if hit:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            cache[key] = None
            if len(cache) > capacity_entries:
                cache.popitem(last=False)
        if block > 0:
            steady_accesses += 1
            steady_hits += int(hit)
    accesses = hits + misses
    return {
        "capacity_entries": capacity_entries,
        "accesses": accesses,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(accesses, 1),
        "steady_state_accesses": steady_accesses,
        "steady_state_hits": steady_hits,
        "steady_state_hit_rate": steady_hits / max(steady_accesses, 1),
        "baseline_host_bytes": accesses * PANEL_BYTES,
        "host_bytes_avoided": hits * PANEL_BYTES,
        "remaining_host_bytes": misses * PANEL_BYTES,
        "first_block_accesses": first_block_accesses,
    }


def main() -> int:
    payload = {
        "kind": "s100_phase34_panel_reuse",
        "status": "started",
        "context": CONTEXT,
        "blocks": BLOCKS,
        "panel_payload_bytes": PANEL_BYTES,
        "started_utc": utc_now(),
        "claim_boundary": "synchronized exact route/panel diagnostic; not throughput",
    }
    runtime = None
    try:
        import cupy as cp

        tokens = load_trace()["tokens"]
        runtime, graph, keep = make_candidate(CONTEXT, "dense_m8")
        tracer = PanelTracer(graph.gmoe)
        graph.gmoe = tracer
        prefill_to(runtime, tokens, CONTEXT)
        graph.prepare_after_prefill()
        blocks = []
        all_exact = True
        for block in range(BLOCKS):
            tracer.block = block
            pos = int(runtime.pos)
            drafts, expected = expected_for_h8(tokens, pos)
            graph.tok_dev[...] = cp.asarray(drafts)
            graph.set_pos_from_host()
            graph.body()
            cp.cuda.get_current_stream().synchronize()
            runtime.copy_stream.synchronize()
            graph.gmoe.shared_stream.synchronize()
            got = cp.asnumpy(graph.ids_dev).astype(np.int32)
            exact = bool(np.array_equal(got, expected))
            all_exact = all_exact and exact
            if not exact:
                raise RuntimeError(
                    f"token mismatch block={block} pos={pos} got={got.tolist()}"
                )
            runtime.pos += 8
            blocks.append(
                {
                    "block": block,
                    "pos": pos,
                    "ids": got.tolist(),
                    "exact": exact,
                }
            )

        rows = []
        stream = []
        per_block = defaultdict(int)
        per_layer = defaultdict(list)
        for record in tracer.records:
            block = int(record["block"])
            layer = int(record["layer"])
            keys = record["keys"]
            per_block[block] += len(keys)
            per_layer[layer].append((block, set(keys)))
            for key in keys:
                rows.append((block, *key))
                stream.append((block, key))

        rows_np = np.asarray(rows, np.int32)
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            TRACE,
            block=rows_np[:, 0],
            layer=rows_np[:, 1],
            expert=rows_np[:, 2],
            panel=rows_np[:, 3],
        )
        digest = hashlib.sha256(rows_np.tobytes()).hexdigest()

        layer_reuse = []
        for layer, visits in sorted(per_layer.items()):
            intersections = []
            denominators = []
            for (_, previous), (_, current) in zip(visits, visits[1:]):
                intersections.append(len(previous & current))
                denominators.append(len(current))
            layer_reuse.append(
                {
                    "layer": layer,
                    "mean_keys_per_window": float(
                        np.mean([len(keys) for _, keys in visits])
                    ),
                    "previous_window_hit_rate": (
                        sum(intersections) / max(sum(denominators), 1)
                    ),
                }
            )

        first_block_accesses = int(per_block[0])
        simulations = {}
        for mib in CAPACITY_MIB:
            entries = (mib * 2**20) // PANEL_BYTES
            simulations[str(mib)] = simulate_lru(
                stream, entries, first_block_accesses
            )

        implementation_open = bool(
            all_exact
            and any(
                result["steady_state_hit_rate"] >= 0.20
                and result["host_bytes_avoided"] >= 64 * 2**20
                for result in simulations.values()
            )
        )
        payload.update(
            {
                "status": "measured",
                "all_tokens_exact": all_exact,
                "block_results": blocks,
                "layer_calls": len(tracer.records),
                "total_panel_accesses": len(stream),
                "unique_panel_keys": len({key for _, key in stream}),
                "panel_accesses_per_block": {
                    str(block): int(per_block[block]) for block in range(BLOCKS)
                },
                "trace_npz": str(TRACE),
                "trace_sha256": digest,
                "lru": simulations,
                "per_layer_previous_window_reuse": layer_reuse,
                "PERSISTENT_PANEL_CACHE_IMPLEMENTATION_OPEN": implementation_open,
                "completed_utc": utc_now(),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )
    finally:
        if runtime is not None:
            try:
                release(runtime)
            except Exception:
                pass
    write_json_atomic(OUT, payload, archive=True)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "all_tokens_exact": payload.get("all_tokens_exact"),
                "total_panel_accesses": payload.get("total_panel_accesses"),
                "unique_panel_keys": payload.get("unique_panel_keys"),
                "lru": payload.get("lru"),
                "implementation_open": payload.get(
                    "PERSISTENT_PANEL_CACHE_IMPLEMENTATION_OPEN"
                ),
                "error": (payload.get("error") or {}).get("message"),
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
