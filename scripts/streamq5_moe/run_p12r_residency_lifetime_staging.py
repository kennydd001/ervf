from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12


P7_RUNNER = ROOT / "scripts/streamq5_moe/run_p7c_ervf_end_to_end.py"
P12R_PREREG = ROOT / "reports/streamq5_moe/P12R_RESIDENCY_LIFETIME_STAGING_PREREGISTRATION.md"
P12R_OUTPUT = ROOT / "reports/streamq5_moe/p12r_residency_lifetime_staging.json"


OLD_PIN = '''def pin_q8_bank(bank):
    total = bank["aggregate"]["bytes"]
    memory = cp.cuda.alloc_pinned_memory(total)
    host = np.frombuffer(memory, dtype=np.uint8, count=total)
    host_offsets = {}
    cursor = 0
    digest = hashlib.sha256()
    for index, record in enumerate(bank["records"]):
        path = ROOT / record["artifact"]
        target = memoryview(host[cursor:cursor + record["bytes"]])
        with path.open("rb") as handle:
            if handle.readinto(target) != record["bytes"]:
                raise RuntimeError("short P6 bank read")
        raw = memoryview(host[cursor:cursor + record["bytes"]])
        if hashlib.sha256(raw).hexdigest() != record["artifact_sha256"]:
            raise ValueError("P6 pinned record hash mismatch")
        digest.update(raw)
        host_offsets[index] = cursor
        cursor += record["bytes"]
    if cursor != total:
        raise RuntimeError("P6 pinned byte mismatch")
    return memory, host, host_offsets, digest.hexdigest()
'''


NEW_PIN = '''def pin_q8_bank(bank):
    total = bank["aggregate"]["host_embedding_bytes"]
    memory = cp.cuda.alloc_pinned_memory(total)
    host = np.frombuffer(memory, dtype=np.uint8, count=total)
    host_offsets = {}
    cursor = 0
    digest = hashlib.sha256()
    for index, record in enumerate(bank["records"]):
        path = ROOT / record["artifact"]
        if record["residency"] == "host":
            target = memoryview(host[cursor:cursor + record["bytes"]])
            with path.open("rb") as handle:
                if handle.readinto(target) != record["bytes"]:
                    raise RuntimeError("short P12R host record read")
            raw = memoryview(host[cursor:cursor + record["bytes"]])
            if hashlib.sha256(raw).hexdigest() != record["artifact_sha256"]:
                raise ValueError("P12R host record hash mismatch")
            digest.update(raw)
            host_offsets[index] = cursor
            cursor += record["bytes"]
        else:
            host_offsets[index] = None
            record_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 2**20), b""):
                    record_digest.update(chunk); digest.update(chunk)
            if record_digest.hexdigest() != record["artifact_sha256"]:
                raise ValueError("P12R device source hash mismatch")
    if cursor != total:
        raise RuntimeError("P12R compact host byte mismatch")
    return memory, host, host_offsets, digest.hexdigest()
'''


OLD_COPY = '''        device_cursor = 0
        for index, record in enumerate(self.bank["records"]):
            self.record_by_key[(record["layer"], record["name"])] = (index, record)
            if record["residency"] == "device":
                self.device_offsets[(record["layer"], record["name"])] = device_cursor
                cp.cuda.runtime.memcpyAsync(
                    self.trunk_memory.ptr + device_cursor,
                    self.q8_pinned.ptr + self.q8_host_offsets[index],
                    record["bytes"], cp.cuda.runtime.memcpyHostToDevice, self.compute.ptr,
                )
                device_cursor += record["bytes"]
        if device_cursor != self.bank["aggregate"]["device_bytes"]:
            raise RuntimeError("device trunk byte mismatch")
'''


NEW_COPY = '''        device_cursor = 0
        stage_bytes = max(record["bytes"] for record in self.bank["records"] if record["residency"] == "device")
        self.q8_stage_bytes = stage_bytes
        stage_memory = cp.cuda.alloc_pinned_memory(stage_bytes)
        stage = np.frombuffer(stage_memory, dtype=np.uint8, count=stage_bytes)
        for index, record in enumerate(self.bank["records"]):
            self.record_by_key[(record["layer"], record["name"])] = (index, record)
            if record["residency"] == "device":
                self.device_offsets[(record["layer"], record["name"])] = device_cursor
                target = memoryview(stage[:record["bytes"]])
                with (ROOT / record["artifact"]).open("rb") as handle:
                    if handle.readinto(target) != record["bytes"]:
                        raise RuntimeError("short P12R staging read")
                if hashlib.sha256(target).hexdigest() != record["artifact_sha256"]:
                    raise ValueError("P12R staging hash mismatch")
                cp.cuda.runtime.memcpyAsync(
                    self.trunk_memory.ptr + device_cursor, stage_memory.ptr,
                    record["bytes"], cp.cuda.runtime.memcpyHostToDevice, self.compute.ptr,
                )
                self.compute.synchronize()
                device_cursor += record["bytes"]
        del stage
        del stage_memory
        if device_cursor != self.bank["aggregate"]["device_bytes"]:
            raise RuntimeError("device trunk byte mismatch")
'''


def load_runtime_class_compact():
    source = P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    if marker not in source: raise RuntimeError("P7 replacements marker missing")
    injected = (
        "replacements.update({" + repr(OLD_PIN) + ": " + repr(NEW_PIN) + ", "
        + repr(OLD_COPY) + ": " + repr(NEW_COPY) + "})\n" + marker
    )
    source = source.replace(marker, injected)
    old_exec = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    new_exec = 'exec(compile(source, str(source_path), "exec"), globals())'
    if old_exec not in source: raise RuntimeError("P7 import transform target missing")
    namespace = {"__name__": "p12r_runtime", "__file__": str(P7_RUNNER)}
    exec(compile(source.replace(old_exec, new_exec), str(P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


if __name__ == "__main__":
    p12.PREREG = P12R_PREREG
    p12.OUTPUT = P12R_OUTPUT
    p12.load_runtime_class = load_runtime_class_compact
    p12.__file__ = __file__
    p12.main()
    if P12R_OUTPUT.exists():
        result = json.loads(P12R_OUTPUT.read_text(encoding="utf-8"))
        result["kind"] = "streamq5_moe_p12r_residency_lifetime_staging"
        result["residency_lifetime_staging"] = {
            "persistent_q8_host_bytes": 316026880,
            "maximum_q8_stage_bytes": 8519680,
            "superseded_p12_failure": "reports/streamq5_moe/p12_32g_allocation_failure.json",
        }
        P12R_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"p12r_output": str(P12R_OUTPUT), "overall_pass": result["overall_pass"]}), flush=True)
