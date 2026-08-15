from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p12r_residency_lifetime_staging as p12r


P7_RUNNER = ROOT / "scripts/streamq5_moe/run_p7c_ervf_end_to_end.py"
PREREG = ROOT / "reports/streamq5_moe/P12R2_PINNED_WINDOW_STREAMING_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/streamq5_moe/p12r2_pinned_window_streaming.json"


MAPPED_CODE = '''class PinnedWindowExpertBank:
    def __init__(self, bank):
        self.layers = []
        self.hashes = {}
        bank_dir = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
        for layer in range(LAYERS):
            path = bank_dir / f"layer_{layer:02d}.q5bin"
            mapped = np.memmap(path, dtype=np.uint8, mode="r", shape=(LAYER_BYTES,))
            observed = hashlib.sha256(memoryview(mapped)).hexdigest()
            if observed != bank["manifests"][str(layer)]["artifact_sha256"]:
                raise ValueError("P12R2 mapped bank hash mismatch")
            self.hashes[str(layer)] = observed
            self.layers.append(mapped)
            if layer % 8 == 7:
                print(json.dumps({"mapped_layers": layer + 1}), flush=True)
        self.memories = [cp.cuda.alloc_pinned_memory(EXPERT_BYTES) for _ in range(8)]
        self.windows = [np.frombuffer(memory, dtype=np.uint8, count=EXPERT_BYTES) for memory in self.memories]
        self.events = [cp.cuda.Event() for _ in range(8)]
        self.used = [False] * 8
        self.cursor = 0

    def copy(self, stream, cache, layer_bases, layer, expert, slot):
        index = self.cursor % 8
        self.cursor += 1
        if self.used[index]:
            self.events[index].synchronize()
        begin = expert * EXPERT_BYTES
        np.copyto(self.windows[index], self.layers[layer][begin:begin + EXPERT_BYTES])
        cp.cuda.runtime.memcpyAsync(
            cache.ptr + (layer_bases[layer] + slot) * EXPERT_BYTES,
            self.memories[index].ptr, EXPERT_BYTES,
            cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
        )
        self.events[index].record(stream)
        self.used[index] = True


def map_expert_bank(bank):
    started = time.perf_counter()
    mapped = PinnedWindowExpertBank(bank)
    return mapped, mapped.hashes, (time.perf_counter() - started) * 1000.0


def copy_expert_mapped(stream, bank, cache, layer_bases, layer, expert, slot):
    bank.copy(stream, cache, layer_bases, layer, expert, slot)


class Runtime:'''


def load_runtime_class_pinned_window():
    source = P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    additions = {
        p12r.OLD_PIN: p12r.NEW_PIN,
        p12r.OLD_COPY: p12r.NEW_COPY,
        "class Runtime:": MAPPED_CODE,
        "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
        "copy_expert(": "copy_expert_mapped(",
    }
    pairs = ", ".join(repr(old) + ": " + repr(new) for old, new in additions.items())
    source = source.replace(marker, "replacements.update({" + pairs + "})\n" + marker)
    old_exec = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    new_exec = 'exec(compile(source, str(source_path), "exec"), globals())'
    if old_exec not in source: raise RuntimeError("P7 import transform target missing")
    namespace = {"__name__": "p12r2_runtime", "__file__": str(P7_RUNNER)}
    exec(compile(source.replace(old_exec, new_exec), str(P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


if __name__ == "__main__":
    p12.PREREG = PREREG
    p12.OUTPUT = OUTPUT
    p12.load_runtime_class = load_runtime_class_pinned_window
    p12.__file__ = __file__
    p12.main()
    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text(encoding="utf-8"))
        result["kind"] = "streamq5_moe_p12r2_pinned_window_streaming"
        result["pinned_window_streaming"] = {
            "mapped_expert_bank_bytes": 18647875584,
            "pinned_expert_window_bytes": 8 * 3035136,
            "persistent_q8_host_bytes": 316026880,
            "maximum_q8_stage_bytes": 8519680,
            "superseded_failures": [
                "reports/streamq5_moe/p12_32g_allocation_failure.json",
                "reports/streamq5_moe/p12r_32g_allocation_failure.json"
            ]
        }
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"p12r2_output": str(OUTPUT), "overall_pass": result["overall_pass"]}), flush=True)
