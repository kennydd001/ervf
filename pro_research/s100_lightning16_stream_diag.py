from __future__ import annotations

from collections import defaultdict
import json
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    RESULTS, assert_lightning, case_manifest, ensure_results,
    feed_prompt, load_trace, normalize_eager_moe,
    prompt_manifest, reset_eager,
)
from s100_lightning16_native import NativeEngine

OUT = RESULTS / "S100_LIGHTNING16_STREAM_DIAG.json"
PATHS = ("legacy", "context_first", "sync_control")
PROMPTS_USED = 2
TOKENS_USED = 8

def metric(cp, reference, candidate):
    diff = candidate - reference
    rn = float(cp.linalg.norm(reference).item())
    cn = float(cp.linalg.norm(candidate).item())
    dot = float(cp.dot(
        reference.reshape(-1), candidate.reshape(-1)
    ).item())
    return {
        "nrmse": float(cp.linalg.norm(diff).item() / max(rn, 1e-30)),
        "cosine": dot / max(rn * cn, 1e-30),
        "max_abs": float(cp.max(cp.abs(diff)).item()),
        "finite": bool(cp.isfinite(candidate).all().item()),
    }

class Shadow:
    def __init__(self, rt, path, records):
        import cupy as cp
        self.rt = rt
        self.cp = cp
        self.original = rt.k.mv_bf16
        self.engine = NativeEngine(path)
        self.records = {row["pointer"]: row for row in records}
        self.scratch = {}
        self.rows = []
        self.active = False

    def __call__(self, out, weight, x, rows, cols):
        record = self.records.get(int(weight.data.ptr))
        if record is None or not self.active:
            return self.original(out, weight, x, rows, cols)

        scratch = self.scratch.get(record["case"])
        if scratch is None:
            scratch = self.cp.empty(rows, self.cp.float32)
            self.scratch[record["case"]] = scratch

        producer_stream = int(self.cp.cuda.get_current_stream().ptr)
        self.engine.run(
            weight, x.reshape(1, cols), scratch.reshape(1, rows),
            int(rows), int(cols), 1, 2,
        )
        self.original(out, weight, x, rows, cols)
        self.cp.cuda.get_current_stream().synchronize()
        row = metric(self.cp, out, scratch)
        row.update({
            "case": record["case"],
            "family": record["family"],
            "layer": record["layer"],
            "producer_stream_ptr": producer_stream,
            "torch_external_stream_ptr": int(
                getattr(
                    self.engine._stream(self.cp),
                    "cuda_stream",
                    producer_stream,
                )
            ),
        })
        self.rows.append(row)
        return None

def aggregate(rows):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    cases = {}
    for case, values in by_case.items():
        cases[case] = {
            "calls": len(values),
            "family": values[0]["family"],
            "layer": values[0]["layer"],
            "max_nrmse": max(v["nrmse"] for v in values),
            "mean_nrmse": float(np.mean([v["nrmse"] for v in values])),
            "min_cosine": min(v["cosine"] for v in values),
            "max_abs": max(v["max_abs"] for v in values),
            "stream_pointer_equal": all(
                v["producer_stream_ptr"]
                == v["torch_external_stream_ptr"]
                for v in values
            ),
            "finite": all(v["finite"] for v in values),
        }
    return {
        "calls": len(rows),
        "max_nrmse": max((v["nrmse"] for v in rows), default=None),
        "mean_nrmse": float(np.mean([
            v["nrmse"] for v in rows
        ])) if rows else None,
        "min_cosine": min((v["cosine"] for v in rows), default=None),
        "cases": cases,
    }

def run_path(path):
    import cupy as cp
    from transformers import AutoTokenizer
    from common import require_model_dir

    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False
    normalize_eager_moe(rt)
    records = case_manifest(rt)
    shadow = Shadow(rt, path, records)
    rt.k.mv_bf16 = shadow

    metadata, trace = load_trace("calibration")
    targets = trace["target_ids"]
    manifest = prompt_manifest()
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    for pi, record in enumerate(
        metadata["prompt_records"][:PROMPTS_USED]
    ):
        prompt = manifest[record["id"]]
        ids = tokenizer.encode(
            prompt["prompt"], add_special_tokens=False
        )
        reset_eager(rt)
        shadow.active = False
        feed_prompt(rt, ids)
        shadow.active = True
        # Advance the parent through frozen Lightning targets. Each step
        # exercises all K/V/O calls on actual decode activations.
        for ti in range(TOKENS_USED):
            rt.step(int(targets[pi, ti]))
        shadow.active = False

    result = aggregate(shadow.rows)
    result["path"] = path
    result["terms"] = 2
    result["torch_mm_style"] = shadow.engine.mm.style
    bundle.restore_combined()
    bundle.restore_sel()
    return result

def sentinel():
    import cupy as cp

    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False
    records = case_manifest(rt)
    representatives = {}
    for family in ("k", "v", "o"):
        representatives[family] = next(
            row for row in records if row["family"] == family
        )
    by_pointer = {
        int(rt.layer[row["layer"]][row["key"]].data.ptr): row
        for row in representatives.values()
    }
    results = {}

    for path in PATHS:
        engine = NativeEngine(path)
        rows = []
        for family, record in representatives.items():
            weight = rt.layer[record["layer"]][record["key"]]
            source = cp.random.standard_normal(
                record["cols"], dtype=cp.float32
            )
            produced = cp.empty_like(source)
            reference = cp.empty(record["rows"], cp.float32)
            candidate = cp.empty(record["rows"], cp.float32)
            for rep in range(32):
                # Enqueue a fresh producer immediately before the consumer.
                cp.multiply(
                    source, np.float32(1.0 + rep * 1e-5),
                    out=produced,
                )
                engine.run(
                    weight, produced.reshape(1, -1),
                    candidate.reshape(1, -1),
                    record["rows"], record["cols"], 1, 2,
                )
                rt.k.mv_bf16(
                    reference, weight, produced,
                    record["rows"], record["cols"],
                )
                cp.cuda.get_current_stream().synchronize()
                m = metric(cp, reference, candidate)
                m.update({"family": family, "rep": rep})
                rows.append(m)
        results[path] = aggregate(rows)

    bundle.restore_combined()
    bundle.restore_sel()
    return results

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning16_stream_diag",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        ident = assert_lightning()
        sentinel_rows = sentinel()
        real = {path: run_path(path) for path in PATHS}
        legacy = real["legacy"]["max_nrmse"]
        context = real["context_first"]["max_nrmse"]
        sync = real["sync_control"]["max_nrmse"]
        fixed = min(context, sync)
        confirmed = bool(
            legacy is not None and fixed is not None
            and legacy >= 10.0 * max(fixed, 1e-12)
            and fixed <= 1e-4
        )
        recommended = (
            "context_first"
            if context <= sync * 1.25 else "sync_control"
        )
        payload.update({
            "status": "measured",
            "identity": ident,
            "sentinel": sentinel_rows,
            "real_activation_shadow": real,
            "STREAM_HANDSHAKE_BUG_CONFIRMED": confirmed,
            "recommended_handoff": recommended,
            "FIXED_NATIVE_SHADOW_GREEN": bool(
                fixed is not None and fixed <= 1e-4
            ),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "STREAM_HANDSHAKE_BUG_CONFIRMED": payload.get(
            "STREAM_HANDSHAKE_BUG_CONFIRMED"
        ),
        "FIXED_NATIVE_SHADOW_GREEN": payload.get(
            "FIXED_NATIVE_SHADOW_GREEN"
        ),
        "recommended_handoff": payload.get("recommended_handoff"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
