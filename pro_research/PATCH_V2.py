"""Patch the already-expanded PRO research pack after the first manual smoke.

This is intentionally surgical and fail-closed.  It modifies only the local
expanded PRO files, never the model/runtime or closed research reports.

Fixes:
1. Windows PowerShell 5.1 + ErrorActionPreference=Stop treated CuPy's harmless
   stderr UserWarning as a terminating PowerShell error even when Python could
   continue.  Native exit code is now authoritative.
2. Graph smoke no longer applies the frozen >=500-sample performance gate to a
   16-token technical smoke.  Correctness/VRAM remain mandatory; timing remains
   recorded as diagnostic data.
3. Dense smoke similarly tests compilation + bit exactness + causal wiring, not
   tiny-sample performance gates.
4. FP8 generalized ERVF now preserves the production gemv_fp8_tensor's exact
   virtual-thread MAC assignment: one uchar4 vector per virtual tid, four FMAs
   in the original order, rather than scalar-stride assignment.
5. Smoke orchestration accepts graph status smoke_pass and can then exercise the
   epoch smoke.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

PRO = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"already patched: {path.name} [{marker}]")
        return False
    if old not in text:
        raise RuntimeError(
            f"Refusing to patch {path}: expected source block was not found. "
            "The local file may differ from the installed PRO v1 source."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path.name} [{marker}]")
    return True


def patch_ps1() -> None:
    path = PRO / "INSTALL_AND_RUN.ps1"
    old = r'''switch ($Mode) {
    'install' {
        & $Python -m compileall -q $ProDir
        if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed.' }
        & $Python (Join-Path $ProDir 'ervf_dense.py') --selftest
        if ($LASTEXITCODE -ne 0) { throw 'ERVF reduction-tree selftest failed.' }
        Write-Host 'PRO pack installed and CPU selftest passed.' -ForegroundColor Green
        Write-Host 'Next: .\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke'
    }
    'smoke'  { & $Python (Join-Path $ProDir 'run_all.py') smoke }
    'full'   { & $Python (Join-Path $ProDir 'run_all.py') full }
    'graph'  { & $Python (Join-Path $ProDir 'run_all.py') graph }
    'dense'  { & $Python (Join-Path $ProDir 'run_all.py') dense }
    'epoch'  { & $Python (Join-Path $ProDir 'run_all.py') epoch }
    'verify' { & $Python (Join-Path $ProDir 'run_all.py') verify }
    'report' { & $Python (Join-Path $ProDir 'run_all.py') report }
}

if ($LASTEXITCODE -ne 0) {
    throw "PRO runner exited with code $LASTEXITCODE. Read pro_research\results\logs."
}
'''
    new = r'''function Invoke-ProPython([string[]]$Arguments) {
    # Windows PowerShell 5.1 promotes native stderr lines to ErrorRecords.
    # CuPy can emit a CUDA_PATH UserWarning on stderr while the process itself
    # remains usable. Do not let ErrorActionPreference=Stop create a false
    # failure; the native process exit code remains authoritative.
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments
        $nativeRc = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($nativeRc -ne 0) {
        throw "PRO Python command exited with code $nativeRc. Read pro_research\results\logs."
    }
}

switch ($Mode) {
    'install' {
        Invoke-ProPython @('-m', 'compileall', '-q', $ProDir)
        Invoke-ProPython @((Join-Path $ProDir 'ervf_dense.py'), '--selftest')
        Write-Host 'PRO pack installed and CPU selftest passed.' -ForegroundColor Green
        Write-Host 'Next: .\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke'
    }
    'smoke'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'smoke') }
    'full'   { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'full') }
    'graph'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'graph') }
    'dense'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'dense') }
    'epoch'  { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'epoch') }
    'verify' { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'verify') }
    'report' { Invoke-ProPython @((Join-Path $ProDir 'run_all.py'), 'report') }
}
'''
    replace_once(path, old, new, "Windows PowerShell 5.1 promotes")


def patch_run_all() -> None:
    path = PRO / "run_all.py"
    old = '''        if result_status("PRO_G0_E1F22_GRAPH_AB.json") == "pass":\n            rc |= run_step("epoch_smoke", [str(PRO / "epoch_graph.py"), "--mode", "smoke"])\n'''
    new = '''        if result_status("PRO_G0_E1F22_GRAPH_AB.json") in {"pass", "smoke_pass"}:\n            rc |= run_step("epoch_smoke", [str(PRO / "epoch_graph.py"), "--mode", "smoke"])\n'''
    replace_once(path, old, new, 'in {"pass", "smoke_pass"}')


def patch_graph() -> None:
    path = PRO / "graph_e1f22.py"
    old = '''        payload["gates"] = gates\n        mandatory = [gates["G_E1F22_PAR"]["passed"], gates["G_E1F22_DET"]["passed"], gates["G_E1F22_S1"]["passed"], gates["G_E1F22_VRAM"]["passed"]]\n        if not args.skip_control:\n            mandatory.append(bool(gates["G_E1F22_CTL"]["passed"]))\n        payload["status"] = "pass" if all(mandatory) else "gate_failed"\n'''
    new = '''        payload["gates"] = gates\n        if args.mode == "smoke":\n            # Smoke is a technical/correctness check, not a speed claim. Sixteen\n            # tokens cannot adjudicate the frozen >=500-sample S1 performance gate.\n            # Keep S1 in the JSON as diagnostic data, but do not let it turn a\n            # working graph into a false smoke failure.\n            smoke_mandatory = [\n                bool(gates["G_E1F22_PAR"]["passed"]),\n                bool(gates["G_E1F22_DET"]["passed"]),\n                bool(gates["G_E1F22_VRAM"]["passed"]),\n            ]\n            if not args.skip_control:\n                smoke_mandatory.append(bool(gates["G_E1F22_CTL"]["passed"]))\n            payload["status"] = "smoke_pass" if all(smoke_mandatory) else "smoke_gate_failed"\n        else:\n            mandatory = [\n                bool(gates["G_E1F22_PAR"]["passed"]),\n                bool(gates["G_E1F22_DET"]["passed"]),\n                bool(gates["G_E1F22_S1"]["passed"]),\n                bool(gates["G_E1F22_VRAM"]["passed"]),\n            ]\n            if not args.skip_control:\n                mandatory.append(bool(gates["G_E1F22_CTL"]["passed"]))\n            payload["status"] = "pass" if all(mandatory) else "gate_failed"\n'''
    replace_once(path, old, new, "smoke_mandatory")

    old2 = '''    print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))\n    return 0 if payload["status"] in {"pass", "gate_failed"} else 2\n'''
    new2 = '''    console = {"status": payload["status"], "output": str(OUT)}\n    if "gates" in payload:\n        console["gates"] = {\n            name: gate.get("passed") for name, gate in payload["gates"].items()\n        }\n        s1 = payload["gates"].get("G_E1F22_S1", {})\n        console["timing"] = {\n            "eager_p50_ms": s1.get("eager_p50_ms"),\n            "graph_p50_ms": s1.get("graph_p50_ms"),\n            "gain_ms": s1.get("gain_ms"),\n        }\n    print(json.dumps(console, indent=2))\n    return 0 if payload["status"] in {"pass", "gate_failed", "smoke_pass", "smoke_gate_failed"} else 2\n'''
    replace_once(path, old2, new2, 'console["gates"]')


def patch_dense() -> None:
    path = PRO / "ervf_dense.py"
    old = '''    const unsigned char* w = W + (size_t)row * cols;\n\n    float acc[PRO_VIRTUAL];\n    #pragma unroll\n    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;\n    #pragma unroll\n    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {\n        const int tid = lane + PRO_WIDTH * vi;\n        for (int k = tid; k < cols; k += 256)\n            acc[vi] = fmaf(lut[w[k]], sx[k], acc[vi]);\n    }\n    const float v = pro_reduce_exact(acc);\n    if (lane == 0) out[row] = v * wscale;\n'''
    new = '''    const unsigned char* w = W + (size_t)row * cols;\n    const uchar4* __restrict__ w4 = reinterpret_cast<const uchar4*>(w);\n    const int nvec = cols >> 2;\n\n    float acc[PRO_VIRTUAL];\n    #pragma unroll\n    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) acc[vi] = 0.0f;\n    #pragma unroll\n    for (int vi = 0; vi < PRO_VIRTUAL; ++vi) {\n        // Production gemv_fp8_tensor assigns one uchar4 vector to each\n        // virtual tid. Preserve that assignment and its four-FMA order exactly.\n        const int tid = lane + PRO_WIDTH * vi;\n        float a = 0.0f;\n        for (int vec = tid; vec < nvec; vec += 256) {\n            const uchar4 q = w4[vec];\n            const int k = vec << 2;\n            a = fmaf(lut[q.x], sx[k],     a);\n            a = fmaf(lut[q.y], sx[k + 1], a);\n            a = fmaf(lut[q.z], sx[k + 2], a);\n            a = fmaf(lut[q.w], sx[k + 3], a);\n        }\n        for (int k = (nvec << 2) + tid; k < cols; k += 256)\n            a = fmaf(lut[w[k]], sx[k], a);\n        acc[vi] = a;\n    }\n    const float v = pro_reduce_exact(acc);\n    if (lane == 0) out[row] = v * wscale;\n'''
    replace_once(path, old, new, "one uchar4 vector to each")

    old2 = '''        can_integrate = exact and no_regression and gmean is not None and gmean >= 1.25\n        payload["integration_opened"] = bool(can_integrate and not args.micro_only)\n'''
    new2 = '''        full_micro_gate = exact and no_regression and gmean is not None and gmean >= 1.25\n        # Smoke validates compilation + bit exactness + full-model wiring. It\n        # deliberately does not use a tiny timing sample to open/close the frozen\n        # performance gate. Full mode remains fail-closed on the 1.25x micro gate.\n        can_integrate = exact if args.mode == "smoke" else full_micro_gate\n        payload["integration_opened"] = bool(can_integrate and not args.micro_only)\n'''
    replace_once(path, old2, new2, "full_micro_gate")

    old3 = '''        micro_gates = payload["microbench"]["gates"]\n        if not all(micro_gates.values()):\n            payload["status"] = "micro_gate_failed"\n        elif args.micro_only:\n            payload["status"] = "micro_pass"\n        elif "integration" not in payload:\n            payload["status"] = "integration_not_opened"\n        elif all(payload["integration"]["gates"].values()):\n            payload["status"] = "pass"\n        else:\n            payload["status"] = "integration_gate_failed"\n'''
    new3 = '''        micro_gates = payload["microbench"]["gates"]\n        if args.mode == "smoke":\n            # Technical smoke only: exact kernel outputs and exact causal rollout\n            # are mandatory. Timing gates remain recorded but are not claims.\n            rollout_exact = (\n                "integration" in payload\n                and bool(payload["integration"]["gates"]["all_rollouts_identical"])\n            )\n            payload["status"] = "smoke_pass" if exact and rollout_exact else "smoke_gate_failed"\n        elif not all(micro_gates.values()):\n            payload["status"] = "micro_gate_failed"\n        elif args.micro_only:\n            payload["status"] = "micro_pass"\n        elif "integration" not in payload:\n            payload["status"] = "integration_not_opened"\n        elif all(payload["integration"]["gates"].values()):\n            payload["status"] = "pass"\n        else:\n            payload["status"] = "integration_gate_failed"\n'''
    replace_once(path, old3, new3, "rollout_exact")

    old4 = '''    print(json.dumps({"status": payload["status"], "output": str(OUT)}, indent=2))\n    return 0 if payload["status"] in {"pass", "micro_pass", "micro_gate_failed", "integration_gate_failed"} else 2\n'''
    new4 = '''    console = {"status": payload["status"], "output": str(OUT)}\n    if "microbench" in payload:\n        console["micro"] = {\n            "all_bit_exact": payload["microbench"]["gates"]["all_bit_exact"],\n            "gmean_speedup": payload["microbench"].get("geometric_mean_speedup"),\n            "weighted_speedup": payload["microbench"].get("weighted_speedup_model"),\n        }\n    if "integration" in payload:\n        console["integration"] = {\n            "all_rollouts_identical": payload["integration"]["gates"]["all_rollouts_identical"],\n            "gain_ms": payload["integration"].get("gain_ms"),\n            "tok_s": payload["integration"].get("tok_s"),\n        }\n    print(json.dumps(console, indent=2))\n    return 0 if payload["status"] in {\n        "pass", "micro_pass", "micro_gate_failed", "integration_gate_failed",\n        "smoke_pass", "smoke_gate_failed"\n    } else 2\n'''
    replace_once(path, old4, new4, 'console["micro"]')


def main() -> int:
    targets = [
        PRO / "INSTALL_AND_RUN.ps1",
        PRO / "run_all.py",
        PRO / "graph_e1f22.py",
        PRO / "ervf_dense.py",
    ]
    before = {p.name: sha(p) for p in targets}
    patch_ps1()
    patch_run_all()
    patch_graph()
    patch_dense()
    after = {p.name: sha(p) for p in targets}

    print("\nSHA-256 before -> after")
    for name in before:
        print(f"  {name}: {before[name]} -> {after[name]}")

    print("\nCompiling patched Python sources...")
    cp = subprocess.run([sys.executable, "-m", "compileall", "-q", str(PRO)])
    if cp.returncode:
        raise SystemExit(cp.returncode)

    print("Running CPU ERVF reduction-tree selftest...")
    st = subprocess.run([sys.executable, str(PRO / "ervf_dense.py"), "--selftest"])
    if st.returncode:
        raise SystemExit(st.returncode)

    print("\nPATCH_V2 PASS")
    print("Next command:")
    print(r"  .\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
