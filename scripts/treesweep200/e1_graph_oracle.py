"""E1 phase 1 — reproduce the N1 graph-resident oracle, and re-measure it on top
of ERVF.

Gates:
  G-E1-R1  reproduce N1's removable fraction (23.7%) within 5 percentage points
  G-E1-E1  report what remains of the issue overhead once ERVF has removed a
           chunk of the actual work -- this is what a graph-resident token could
           still buy on the current best exact kernels

Method is N1's: capture one token's kernel sequence as a CUDA graph with FROZEN
routes (capture forbids synchronisation) and replay it. Same kernels, same
arguments, same bytes; only the issuer changes. Semantically wrong after the
first token, so it is used strictly as a timing oracle.
"""

from __future__ import annotations

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
    LightningRuntime, UP_CODE, UP_SCALE)

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
N1_REMOVABLE = 0.237
GATE_TOL = 0.05


class GraphRuntime(LightningRuntime):
    frozen_routes = None
    emb_row = None
    _cap = None

    def _moe_cached(self, i, out):
        if self.frozen_routes is not None:
            return self._moe_frozen(i, out)
        idx, w = super()._moe_cached(i, out)
        if self._cap is not None:
            self._cap[i] = (np.asarray(idx, dtype=np.int64).copy(),
                            np.asarray(w, dtype=np.float64).copy())
        return idx, w

    def freeze(self, token_id):
        cp = self.cp
        self._cap = {}
        self.step(token_id)
        self.step(token_id)          # second pass: every expert now resident
        self.frozen_routes = dict(self._cap)
        self._cap = None
        row = self.embed_host[token_id * self.hidden:(token_id + 1) * self.hidden]
        self.emb_row = cp.asarray(row)
        cp.cuda.Device(0).synchronize()

    def _moe_frozen(self, i, out):
        d, f = self.layer[i], self.fused
        self._route_device(i)
        out.fill(0)
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
                raise RuntimeError("expert %d not resident; prewarm first" % e)
            f.gemv_into(act_moe,
                        c["codes"][slot * UP_CODE:(slot + 1) * UP_CODE],
                        c["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE],
                        self.normed, bank["g_up"][e], self.moe_inter, self.hidden,
                        apply_relu2=True)
            f.down_masked_into(self.tmp, bank["down_ptr"][e], act_moe, self.mstate,
                               bank["g_dn"][e], self.hidden, self.moe_inter)
            f.accumulate_into(out, self.tmp, float(w[s]), self.hidden)
        return idx, w

    def body(self, token_id):
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


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(v):
    return float(np.percentile(np.asarray(v, dtype=np.float64), 50))


def main() -> int:
    import cupy as cp

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = GraphRuntime(MODEL_DIR, contexts_max=8192, embed_on_host=True, fp8_kv=True)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rng = np.random.default_rng(11)
    varied = [int(v) for v in rng.integers(1000, 60000, size=4096)]

    arms = {}
    for label, use_ervf in (("baseline", False), ("ervf", True)):
        rt.fused.use_ervf = use_ervf
        rt.frozen_routes = None
        rt.reset()
        for j in range(64):
            rt.step(varied[j % len(varied)])
        cp.cuda.Device(0).synchronize()

        eager = []
        for _ in range(24):
            t0 = time.perf_counter_ns()
            rt.step(varied[0])
            cp.cuda.Device(0).synchronize()
            eager.append((time.perf_counter_ns() - t0) / 1e6)
        t_eager = p50(eager)

        rt.freeze(varied[1])
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            rt.body(varied[1])
            graph = stream.end_capture()
        for _ in range(5):
            graph.launch()
        cp.cuda.Device(0).synchronize()
        gs = []
        for _ in range(24):
            t0 = time.perf_counter_ns()
            graph.launch()
            cp.cuda.Device(0).synchronize()
            gs.append((time.perf_counter_ns() - t0) / 1e6)
        t_graph = p50(gs)
        rt.frozen_routes = None
        removable = 1.0 - t_graph / t_eager
        arms[label] = {"use_ervf": use_ervf, "eager_ms": t_eager,
                       "graph_ms": t_graph, "removable": removable,
                       "eager_raw": eager, "graph_raw": gs}
        print("  %-9s eager %7.3f -> graph %7.3f ms | removable %+.1f%%"
              % (label, t_eager, t_graph, removable * 100), flush=True)
    rt.fused.use_ervf = False

    dev = abs(arms["baseline"]["removable"] - N1_REMOVABLE)
    gates = {
        "G_E1_R1_reproduce_n1": {
            "n1_removable": N1_REMOVABLE,
            "measured": arms["baseline"]["removable"],
            "abs_deviation_pp": dev, "tolerance_pp": GATE_TOL,
            "passed": bool(dev <= GATE_TOL)},
        "G_E1_E1_on_top_of_ervf": {
            "removable_with_ervf": arms["ervf"]["removable"],
            "eager_ms_with_ervf": arms["ervf"]["eager_ms"],
            "graph_ms_with_ervf": arms["ervf"]["graph_ms"],
            "residual_ms": arms["ervf"]["eager_ms"] - arms["ervf"]["graph_ms"]},
    }

    payload = {
        "kind": "treesweep200_e1_graph_oracle", "registry": "TREESWEEP200",
        "phase": "E1_PHASE1", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": 72, "max_ctx": 8192, "context": 64,
                   "samples": 24, "n1_reference": N1_REMOVABLE},
        "arms": arms, "gates": gates,
        "claim_boundary": (
            "CUDA-graph replay of one token's kernel sequence with FROZEN routes; "
            "after the first token that is semantically wrong and it is used "
            "strictly as a timing oracle for how much of a token is issue "
            "overhead rather than arithmetic. It is an UPPER BOUND on what any "
            "design that moves issuing to the device could recover, not an "
            "achievable runtime figure. No graph-resident runtime exists: V1 "
            "closed the host-read route for device-side routing and the "
            "device-driven alternative is unbuilt."),
    }
    (OUT_DIR / "E1_GRAPH_ORACLE.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n  G-E1-R1 reproduce N1 (%.1f%% +-%.0fpp): %s (dev %.1fpp)"
          % (N1_REMOVABLE * 100, GATE_TOL * 100, dev <= GATE_TOL, dev * 100))
    print("  residual issue overhead with ERVF on: %.3f ms (%.1f%%)"
          % (arms["ervf"]["eager_ms"] - arms["ervf"]["graph_ms"],
             arms["ervf"]["removable"] * 100))
    print("\nwritten E1_GRAPH_ORACLE.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
