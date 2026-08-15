#!/usr/bin/env python3
"""NC19I2 static preflight.  No compiler, payload, Driver, runtime, or device."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
SCRIPT = Path(__file__).resolve()
CONTRACT = SCRIPT.with_name("het_next_l0_ph1_nvidia_nc19i2_compile_contract.py")
RUNNER = SCRIPT.with_name("run_het_next_l0_ph1_nvidia_nc19i2_compile_only.py")
VERIFIER = SCRIPT.with_name("verify_het_next_l0_ph1_nvidia_nc19i2_compile_only.py")
KERNEL = SCRIPT.with_name("het_next_l0_ph1_nvidia_n5_kernels.cu")
MANIFEST = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_corrected_fixture_manifest.json"
DESIGN_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19_design_lock.json"
SOURCE_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i2_source_lock.json"
LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i2_preflight_lock.json"
OUT = REPORTS / "het_next_l0_ph1_nvidia_nc19i2_static_preflight_result.json"
FAILURES = REPORTS / "het_next_l0_ph1_nvidia_nc19i2_static_preflight_failures"
QUARANTINE = REPORTS / "het_next_l0_ph1_nvidia_nc19i2_static_preflight_quarantine"
ACK = "ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I2_STATIC_PREFLIGHT_ONCE"
MANIFEST_SHA = "e5254c911e5e5997427977df02a5afba3435931e6e8caa7168fb7e2d641a4a90"
AUDIT_SHA = "578df903f3dd975c41d3eeb68b8f70674ebd4d5f8f85fc180b6ce791ae957ab5"
CHECK_NAMES = [
    "authorization_anchor", "direct_bindings", "fixture_manifest", "manifest_classifier_1106",
    "static_ast", "kernel_structure", "abi_exact", "fake_success", "fake_failure_matrix",
    "environment_matrix", "cache_history", "transaction_matrix", "failure_durability",
    "topology_matrix", "verifier_positive", "verifier_negative", "verifier_mutations",
    "no_payload_driver_device",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsha(path: Path) -> str:
    return sha(path.read_bytes())


def load_absolute(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FakeAdapter:
    def __init__(self, fault_op=None, fault_kind=None):
        self.fault_op, self.fault_kind = fault_op, fault_kind
        self.handle = 0
        self.ptx = (b".version 9.0\n.target sm_120\n.address_size 64\n"
                    b".visible .entry q5_linear() {}\n"
                    b".visible .entry bf16_lut_activation() {}\n\0")
        self.cubin = fake_elf()

    def call(self, op, **kwargs):
        if op == self.fault_op and self.fault_kind == "exception":
            raise OSError("injected")
        if op == self.fault_op and self.fault_kind in {"zero_size","oversize","bad_nul"}:
            raise ValueError("injected_" + self.fault_kind)
        code = 7 if op == self.fault_op and self.fault_kind == "code" else 0
        row = {"code": code}
        if op == "nvrtcCreateProgram":
            self.handle = 0x123456789ABCDEF0 if self.fault_kind != "null" or op != self.fault_op else 0
            row["handle"] = self.handle
        elif op == "nvrtcGetProgramLogSize": row["size"] = (0 if self.fault_kind=="zero_size" and op==self.fault_op else 1)
        elif op == "nvrtcGetProgramLog": row["log"] = (b"bad\0nul\0" if self.fault_kind=="bad_nul" and op==self.fault_op else b"\0")
        elif op == "nvrtcGetPTXSize": row["size"] = (17*2**20 if self.fault_kind=="oversize" and op==self.fault_op else len(self.ptx))
        elif op == "nvrtcGetPTX": row["ptx"] = (self.ptx[:-1] if self.fault_kind=="bad_nul" and op==self.fault_op else self.ptx)
        elif op == "nvrtcGetCUBINSize": row["size"] = (33*2**20 if self.fault_kind=="oversize" and op==self.fault_op else len(self.cubin))
        elif op == "nvrtcGetCUBIN": row["cubin"] = self.cubin
        elif op == "nvrtcDestroyProgram": row["handle"] = 0
        return row


def fake_elf() -> bytes:
    """Create a bounded ELF64 fixture with exactly the two function symbols."""
    import struct
    names=b"\0.shstrtab\0.strtab\0.symtab\0"; strings=b"\0q5_linear\0bf16_lut_activation\0"
    sym=b"\0"*24
    for offset in (1,11):
        sym += struct.pack("<IBBHQQ",offset,0x12,0,1,0,0)
    ehsize=64; shoff=ehsize+len(names)+len(strings)+len(sym); total=shoff+4*64
    raw=bytearray(total); raw[:6]=b"\x7fELF\x02\x01"
    struct.pack_into("<HHIQQQIHHHHHH",raw,16,1,190,1,0,0,shoff,0,64,0,0,64,4,1)
    noff=ehsize; soff=noff+len(names); yoff=soff+len(strings)
    raw[noff:soff]=names; raw[soff:yoff]=strings; raw[yoff:shoff]=sym
    def sh(i,name,typ,off,size,link=0,entsize=0):
        struct.pack_into("<IIQQQQIIQQ",raw,shoff+i*64,name,typ,0,0,off,size,link,0,1,entsize)
    sh(1,1,3,noff,len(names)); sh(2,11,3,soff,len(strings)); sh(3,19,2,yoff,len(sym),2,24)
    return bytes(raw)


def check_manifest() -> tuple[bool, dict]:
    raw = MANIFEST.read_bytes()
    if len(raw) > 8*2**20 or sha(raw) != MANIFEST_SHA or raw[:3] != b"\xef\xbb\xbf":
        return False, {}
    manifest = json.loads(raw.decode("utf-8-sig"))
    names = [x["name"] for x in manifest["cases"]]
    good = (manifest.get("kind") == "het_next_l0_ph1_nvidia_nc19i1_corrected_fixture_manifest"
            and manifest.get("fixture_semantics",{}).get("affected_cases_observed") == 10
            and manifest.get("fixture_semantics",{}).get("corrected_raw_rows") == 20
            and len(names) == len(set(names)) == 1106)
    for case in manifest["cases"]:
        entries = case["observed_entries"]
        ordered=[{key:row[key] for key in manifest["observed_entry_fields"]} for row in entries]
        digest = sha(json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode())
        good &= sum(x["size"] for x in entries) == case["observed_total_bytes"]
        if case["input_integrity_policy"]=="nominal_raw_schema_exact": good &= digest == case["observed_tree_digest"]
    return bool(good), manifest


def static_ast() -> bool:
    trees = {p:ast.parse(p.read_text(encoding="utf-8")) for p in (CONTRACT,RUNNER,VERIFIER,SCRIPT)}
    runner_calls = [n for n in ast.walk(trees[RUNNER]) if isinstance(n,ast.Call)]
    subprocess_calls = [n for n in runner_calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="subprocess" and n.func.attr=="run"]
    forbidden = []
    for path,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.split(".")[0].casefold() in {"cupy","torch","numpy","pyopencl","numba"}:
                        forbidden.append((path,alias.name))
    allowed={'base64','copy','hashlib','json','os','re','shutil','time','uuid','pathlib','ctypes','struct','__future__'}
    observed=[]
    for node in ast.walk(trees[CONTRACT]):
        if isinstance(node,ast.Import): observed.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom): observed.append((node.module or '').split('.')[0])
    return len(subprocess_calls)==1 and not forbidden and set(observed)<=allowed and "pathlib" in observed


def no_device_callgraph() -> bool:
    forbidden_modules={"cupy","torch","pycuda","nvcuda","cuda","cudart","pyopencl"}
    forbidden_calls={"cuInit","cuDeviceGet","cuCtxCreate","cudaSetDevice","cudaMalloc","cudaLaunchKernel"}
    for path in (CONTRACT,RUNNER,VERIFIER,SCRIPT):
        tree=ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import) and any(alias.name.split('.')[0].casefold() in forbidden_modules for alias in node.names): return False
            if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0].casefold() in forbidden_modules: return False
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ""
                if name in forbidden_calls: return False
        if path==RUNNER:
            # Compile-only permits exactly kernel32 and NVRTC module names, never Driver/runtime loaders.
            literals={n.value.casefold() for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
            if any(value.endswith(("nvcuda.dll","cudart64_130.dll")) for value in literals): return False
    return True


def kernel_structure() -> bool:
    text = KERNEL.read_text(encoding="utf-8")
    entries = re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\((.*?)\)', text, re.S)
    names = [x[0] for x in entries]
    return names == ["q5_linear","bf16_lut_activation"] and text.count("q5_linear<<<")==0 and all(token in text for token in (
        "cg::tiled_partition<8>", "int pack = lane + 8 * virtual_index",
        "int column = pack * 8", "blockIdx.x * 32", "field < 8", "& 31ULL",
        "fmaf(", "tile.shfl_down(value, distance)",
        "atomicAdd(&counters[row], 1U)", "lut[(unsigned)gate_word]",
    )) and "0xff" not in text.lower()


def abi_surface() -> bool:
    tree=ast.parse(RUNNER.read_text(encoding="utf-8")); text=RUNNER.read_text(encoding="utf-8")
    abi_keys=[]; kernel_keys=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ABI" for t in node.targets) and isinstance(node.value,ast.Dict):
            abi_keys=[k.value for k in node.value.keys if isinstance(k,ast.Constant)]
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="table" for t in node.targets) and isinstance(node.value,ast.Dict):
            kernel_keys=[k.value for k in node.value.keys if isinstance(k,ast.Constant)]
    exact_kernel=["AddDllDirectory","RemoveDllDirectory","LoadLibraryExW","FreeLibrary","GetProcAddress","GetModuleHandleW","GetModuleFileNameW","GetCommandLineW","LocalFree"]
    signatures=("C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int",
                "C.c_void_p, C.c_int, C.POINTER(C.c_char_p)",
                "C.POINTER(C.c_void_p)], C.c_int", "C.CFUNCTYPE(result, *args)")
    return abi_keys==[
        "nvrtcVersion","nvrtcCreateProgram","nvrtcCompileProgram","nvrtcGetProgramLogSize",
        "nvrtcGetProgramLog","nvrtcGetPTXSize","nvrtcGetPTX","nvrtcGetCUBINSize",
        "nvrtcGetCUBIN","nvrtcDestroyProgram"] and kernel_keys==exact_kernel and all(x in text for x in signatures) and "C.WINFUNCTYPE" not in text


def fake_compile(contract) -> tuple[bool,bool]:
    snapshots=[]
    success = contract.compile_with_adapter(FakeAdapter(), b"extern \"C\" __global__ void x(){}", snapshots.append)
    checks = contract.validate_compile_evidence(success)
    success_ok = len(success["ledger"])==10 and len(snapshots)==10 and all(checks.values())
    fault_ok = True
    for op in contract.NVRTC_OPS[:-1]:
        for kind in ("code","exception"):
            evidence = contract.compile_with_adapter(FakeAdapter(op,kind), b"x")
            fault_ok &= len(evidence["ledger"])==10 and [x["op"] for x in evidence["ledger"]]==list(contract.NVRTC_OPS) and evidence["primary"]["state"]=="failure"
    null = contract.compile_with_adapter(FakeAdapter("nvrtcCreateProgram","null"), b"x")
    fault_ok &= null["ledger"][1]["attempted"] and null["ledger"][-1]["attempted"] is False
    for op,kind in (("nvrtcDestroyProgram","code"),("nvrtcDestroyProgram","exception"),
                    ("nvrtcGetProgramLogSize","zero_size"),("nvrtcGetPTXSize","oversize"),
                    ("nvrtcGetCUBINSize","oversize"),("nvrtcGetPTX","bad_nul")):
        evidence=contract.compile_with_adapter(FakeAdapter(op,kind),b"x")
        fault_ok &= len(evidence["ledger"])==10 and evidence["primary"]["state"]=="failure"
    mutated = json.loads(json.dumps({"ledger":success["ledger"],"primary":success["primary"],"artifacts":{}},default=str))
    fault_ok &= not all(contract.validate_compile_evidence(mutated).values())
    return bool(success_ok), bool(fault_ok)


def environment_matrix(contract) -> bool:
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); private=root/"private"; private.mkdir()
        for s in contract.ENV_SUBDIRS.values(): (private/s).mkdir()
        for present_mask in range(1<<len(contract.ENV_KEYS)):
            mapping={key:f"old-{i}" for i,key in enumerate(contract.ENV_KEYS) if present_mask&(1<<i)}
            before=dict(mapping); captured=contract.capture_environment(mapping)
            applied=contract.apply_private_environment(mapping,private)
            if len({applied[k] for k in contract.ENV_SUBDIRS})!=4 or not all(Path(applied[k]).is_absolute() for k in contract.ENV_SUBDIRS): return False
            rows=contract.restore_environment(mapping,captured)
            if mapping!=before or not all(x["code"]==0 for x in rows): return False
    return True


def transaction_matrix(contract) -> bool:
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); out=root/"bundle"; kind="fixture"
        result={"kind":"fixture","revision":"NC19I2"}; bundle=contract.build_bundle(result,{"x.bin":b"x"},kind)
        if not contract.verify_bundle_bytes(bundle,kind): return False
        contract.publish_transaction(out,bundle,kind,lambda p: True)
        if not out.is_dir(): return False
        try: contract.publish_transaction(out,bundle,kind,lambda p: True); return False
        except FileExistsError: pass
        debris=root/"debris.inprogress.1.0123456789abcdef"; debris.mkdir()
        row=contract.recover_inprogress(root,"debris",root/"quarantine")
        if not row["recovered"] or row["attempt_consumed"]: return False
        bad=dict(bundle); bad["result.json"]+=b"x"
        if contract.verify_bundle_bytes(bad,kind): return False
        # Production writer failures: no canonical partial and no hidden temp.
        target=root/"atomic.bin"; original_link=contract.os.link
        try:
            contract.os.link=lambda *_: (_ for _ in ()).throw(OSError("prelink"))
            try: contract.atomic_create(target,b"x"); return False
            except OSError: pass
        finally: contract.os.link=original_link
        if target.exists() or list(root.glob("atomic.bin.inprogress.*")): return False
        postlink=root/"postlink.bin"; original_flush=contract.flush_directory
        try:
            contract.flush_directory=lambda p: (_ for _ in ()).throw(OSError("postlink_flush"))
            try: contract.atomic_create(postlink,b"x"); return False
            except OSError: pass
        finally: contract.flush_directory=original_flush
        if postlink.exists() or list(root.glob("postlink.bin.inprogress.*")): return False
        reject=root/"reject"; contract.publish_transaction.__name__ == "publish_transaction"
        try: contract.publish_transaction(reject,bundle,kind,lambda _:False); return False
        except RuntimeError: pass
        if reject.exists() or list(root.glob("reject.inprogress.*")): return False
        promoted=root/"promoted"
        try:
            def fail_after_rename(path):
                if Path(path)==root and promoted.exists(): raise OSError("postrename_flush")
                return original_flush(path)
            contract.flush_directory=fail_after_rename
            try: contract.publish_transaction(promoted,bundle,kind,lambda _:True); return False
            except OSError: pass
        finally: contract.flush_directory=original_flush
        if promoted.exists() or list(root.glob("promoted.inprogress.*")) or list(root.glob("promoted.rollback.*")): return False
        failure_root=root/"failures"
        one=contract.write_incidental_failure(failure_root,{"kind":"fixture_failure","revision":"NC19I2","status":"incidental_failure"})
        two=contract.write_incidental_failure(failure_root,{"kind":"fixture_failure","revision":"NC19I2","status":"incidental_failure"})
        if one==two or not (one/"failure.json").is_file() or not (two/"failure.json").is_file(): return False
        failed_root=root/"failure_postrename"
        try:
            def fail_failure_flush(path):
                if Path(path)==failed_root and any(p.name.startswith("attempt.") and ".inprogress." not in p.name for p in failed_root.iterdir()):
                    raise OSError("failure_postrename")
                return original_flush(path)
            contract.flush_directory=fail_failure_flush
            try: contract.write_incidental_failure(failed_root,{"kind":"fixture_failure","revision":"NC19I2","status":"incidental_failure"}); return False
            except OSError: pass
        finally: contract.flush_directory=original_flush
        if any(p.is_dir() and p.name.startswith("attempt.") for p in failed_root.iterdir()): return False
        if len(list(failed_root.glob("writer_failure.*.json")))!=1: return False
    return True


def verifier_production_mutations(contract, verifier) -> tuple[bool,bool,bool]:
    source=KERNEL.read_bytes(); adapter=FakeAdapter(); evidence=contract.compile_with_adapter(adapter,source)
    artifacts={"source.cu":source,"build.log":evidence["artifacts"]["log"],
               "ptx.bin":evidence["artifacts"]["ptx"],"cubin.bin":evidence["artifacts"]["cubin"]}
    compile_record=dict(evidence); compile_record["artifacts"]={
        n:{"bytes":len(v),"sha256":sha(v)} for n,v in evidence["artifacts"].items()}
    entries=[{"path":p,"type":"dir","size":0,"mtime_ns":0,"sha256":None}
             for p in (".","cuda_cache","nvrtc_cache","temp","tmp")]
    tree=contract.cache_tree_digest(entries)
    history=[{"stage":stage,"entries":entries,"tree_digest":tree}
             for stage in ("pre_load",*contract.NVRTC_OPS,"post_release")]
    preflight_raw=b'{"pass":true}'
    fixture_source_lock={"revision":"NC19I2","bindings":[{"path":"scripts/streamq5_moe/verify_het_next_l0_ph1_nvidia_nc19i2_compile_only.py","bytes":VERIFIER.stat().st_size,"sha256":fsha(VERIFIER)}]}
    fixture_preflight_lock={"revision":"NC19I2","bindings":[]}
    fixture_auth={"execution_open":True,"preflight_result_sha256":sha(preflight_raw),"bindings":[]}
    trust={"auth_lock":fixture_auth,"preflight_lock":fixture_preflight_lock,
           "source_lock":fixture_source_lock,"preflight_result":{"pass":True},
           "preflight_result_sha256":sha(preflight_raw),"provenance_ok":True}
    result={
        "kind":"het_next_l0_ph1_nvidia_nc19i2_compile_only","revision":"NC19I2",
        "status":"compile_positive","terminal_valid":True,"positive":True,
        "authorization":{"auth_lock":fixture_auth,"ack":"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I2_COMPILE_ONLY_ONCE"},
        "invocation":{"direct":True,"raw":"base -I -B script ACK","native_argv":[str((ROOT/".venv/Scripts/python.exe").resolve()),"-I","-B",str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i2_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I2_COMPILE_ONLY_ONCE"],"parse_error":None,"orig_argv":[str((ROOT/".venv/Scripts/python.exe").resolve()),"-I","-B",str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i2_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I2_COMPILE_ONLY_ONCE"],"sys_argv":[str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i2_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I2_COMPILE_ONLY_ONCE"],"sys_executable":str((ROOT/".venv/Scripts/python.exe").resolve()),"base_executable":str((ROOT/".venv/Scripts/python.exe").resolve()),"prefix":str((ROOT/".venv").resolve()),"base_prefix":str(ROOT.resolve()),"name":"__main__","spec_is_none":True,"package":None,"file":str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i2_compile_only.py").resolve())},
        "source_identity":{"path":str(KERNEL.resolve()),"bytes":len(source),"sha256":sha(source)},
        "toolchain_identity":{"nvrtc":{"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll").resolve()),"bytes":1,"sha256":"c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"},"builtins":{"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll").resolve()),"bytes":1,"sha256":"82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"},"header":{"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/include/nvrtc.h").resolve()),"bytes":1,"sha256":"316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21"},"python":{"path":str((ROOT/".venv/Scripts/python.exe").resolve()),"bytes":1,"sha256":"0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"}},
        "options":list(contract.OPTIONS),"program_name":"het_next_l0_ph1_nvidia_nc19i2.cu",
        "create_operands":{"source_bytes":6173,"source_buffer_bytes":6174,"source_terminal_nul":1,"program_name_bytes":len(contract.PROGRAM_NAME),"program_name_buffer_bytes":len(contract.PROGRAM_NAME)+1,"num_headers":0,"headers":None,"include_names":None},
        "compile":compile_record,"compile_checks":contract.validate_compile_evidence(evidence),
        "loader":{"calling_convention":"WinDLL_kernel32_plus_cdecl_CFUNCTYPE","load_flags":0x1100,"nvrtc_version":[13,3],"nvrtc_abi_names":list(contract.NVRTC_OPS),"kernel32_abi":{"AddDllDirectory":{"argtypes":["c_wchar_p"],"restype":"c_void_p"},"RemoveDllDirectory":{"argtypes":["c_void_p"],"restype":"c_long"},"LoadLibraryExW":{"argtypes":["c_wchar_p","c_void_p","c_ulong"],"restype":"c_void_p"},"FreeLibrary":{"argtypes":["c_void_p"],"restype":"c_long"},"GetProcAddress":{"argtypes":["c_void_p","c_char_p"],"restype":"c_void_p"},"GetModuleHandleW":{"argtypes":["c_wchar_p"],"restype":"c_void_p"},"GetModuleFileNameW":{"argtypes":["c_void_p","LP_c_wchar","c_ulong"],"restype":"c_ulong"}},"invocation_abi":{"GetCommandLineW":{"argtypes":[],"restype":"c_wchar_p"},"LocalFree":{"argtypes":["c_void_p"],"restype":"c_void_p"}},"nvrtc_abi":{"nvrtcVersion":{"argtypes":["LP_c_int","LP_c_int"],"restype":"c_int"},"nvrtcCreateProgram":{"argtypes":["LP_c_void_p","c_char_p","c_char_p","c_int","LP_c_char_p","LP_c_char_p"],"restype":"c_int"},"nvrtcCompileProgram":{"argtypes":["c_void_p","c_int","LP_c_char_p"],"restype":"c_int"},"nvrtcGetProgramLogSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetProgramLog":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetPTXSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetPTX":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetCUBINSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetCUBIN":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcDestroyProgram":{"argtypes":["LP_c_void_p"],"restype":"c_int"}},"modules":{"before":{"nvrtc64_130_0.dll":0,"nvrtc-builtins64_133.dll":0},"after_load":{"nvrtc64_130_0.dll":{"handle":2,"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll").resolve()),"bytes":1,"sha256":"c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"},"nvrtc-builtins64_133.dll":{"handle":0,"path":"","bytes":1,"sha256":"82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"}},"during_compile":{"nvrtc64_130_0.dll":{"handle":2,"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll").resolve()),"bytes":1,"sha256":"c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"},"nvrtc-builtins64_133.dll":{"handle":3,"path":str((ROOT/".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll").resolve()),"bytes":1,"sha256":"82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"}},"post_release":{"nvrtc64_130_0.dll":0,"nvrtc-builtins64_133.dll":0},"module":2,"cookie":1,"flags":0x1100}},
        "ownership":[{"resource":"dll_directory_cookie","identity":1,"registered":True,"code":0},{"resource":"nvrtc_hmodule","identity":2,"registered":True,"code":0},{"resource":"nvrtc_program","identity":0x123456789ABCDEF0,"registered":True,"code":0}],
        "cleanup":[{"resource":"nvrtc_program","identity":0x123456789ABCDEF0,"attempted":True,"code":0,"owned_before":True,"identity_after":0},{"resource":"wrappers","attempted":True,"code":0},{"resource":"nvrtc_hmodule","identity":2,"attempted":True,"code":0},{"resource":"dll_directory_cookie","identity":1,"attempted":True,"code":0},{"resource":"postrelease_module_check","attempted":True,"code":0,"modules":{"nvrtc64_130_0.dll":0,"nvrtc-builtins64_133.dll":0}}],
        "cache":{"private_root":"C:/fixture/private","environment_original":{k:{"present":False,"value":None} for k in contract.ENV_KEYS},"environment_applied":{"CUDA_CACHE_DISABLE":"1","CUDA_CACHE_MAXSIZE":"0","CUDA_CACHE_PATH":"C:/fixture/private/cuda_cache","TMP":"C:/fixture/private/tmp","TEMP":"C:/fixture/private/temp","NVRTC_CACHE_PATH":"C:/fixture/private/nvrtc_cache"},"environment_restore":[{"key":k,"attempted":True,"code":0} for k in reversed(contract.ENV_KEYS)],"history":history,"history_digest":contract.cache_history_digest(history)},
        "exclusions":{"payload_bytes_read":0,"nvcuda_driver_calls":0,"cuda_runtime_calls":0,"device_calls":0},
        "terminal_adjudication":{"terminal":"compile_positive","terminal_valid":True,"next_invocation_allowed":False,"attempt_consumed":True},
    }
    def write_candidate(directory, row, files):
        payload={"result.json":verifier.canonical(row),**files}
        kind="het_next_l0_ph1_nvidia_nc19i2_compile_only"+("_negative" if row["status"]=="compile_valid_negative" else "")
        manifest=verifier.canonical({"kind":kind+"_manifest","revision":"NC19I2","files":[{"name":n,"bytes":len(v),"sha256":sha(v)} for n,v in sorted(payload.items())]})
        commit=verifier.canonical({"kind":kind+"_commit","revision":"NC19I2","state":"complete","result_sha256":sha(payload["result.json"]),"manifest_sha256":sha(manifest)})
        for p in directory.iterdir(): p.unlink()
        for n,v in {**payload,"manifest.json":manifest,"commit.json":commit}.items(): (directory/n).write_bytes(v)
    with tempfile.TemporaryDirectory() as td:
        directory=Path(td); write_candidate(directory,result,artifacts)
        baseline=verifier.verify(directory,trust)
        positive_ok=(set(baseline)==set(verifier.EXPECTED_CHECKS) and all(baseline.values()))
        negative_evidence=contract.compile_with_adapter(FakeAdapter("nvrtcCompileProgram","code"),source)
        negative=json.loads(json.dumps(result)); negative["kind"]="het_next_l0_ph1_nvidia_nc19i2_compile_only_negative"
        negative["status"]="compile_valid_negative"; negative["positive"]=False
        negative["terminal_adjudication"]={"terminal":"compile_valid_negative","terminal_valid":True,"next_invocation_allowed":False,"attempt_consumed":True}
        negative["compile"]={**negative_evidence,"artifacts":{n:{"bytes":len(v),"sha256":sha(v)} for n,v in negative_evidence["artifacts"].items()}}
        negative["compile_checks"]=contract.validate_compile_evidence(negative_evidence)
        negative_files={"source.cu":source,"build.log":negative_evidence["artifacts"]["log"]}
        write_candidate(directory,negative,negative_files)
        negative_checks=verifier.verify(directory,trust)
        negative_ok=(set(negative_checks)==set(verifier.EXPECTED_CHECKS) and all(negative_checks.values()))
        mutations=[
            ("status",lambda r,f:r.__setitem__("status","incidental_failure")),
            ("ledger",lambda r,f:r["compile"]["ledger"][2].__setitem__("code",7)),
            ("source",lambda r,f:f.__setitem__("source.cu",f["source.cu"]+b"x")),
            ("ptx",lambda r,f:f.__setitem__("ptx.bin",f["ptx.bin"].replace(b"sm_120",b"sm_121"))),
            ("cubin",lambda r,f:f.__setitem__("cubin.bin",b"BAD"+f["cubin.bin"][3:])),
            ("cleanup",lambda r,f:r["cleanup"][1].__setitem__("code",5)),
            ("history",lambda r,f:r["cache"]["history"].pop()),
            ("exclusion",lambda r,f:r["exclusions"].__setitem__("device_calls",1)),
            ("authorization",lambda r,f:r["authorization"]["auth_lock"].__setitem__("execution_open",False)),
            ("invocation",lambda r,f:r["invocation"].__setitem__("direct",False)),
            ("toolchain",lambda r,f:r["toolchain_identity"]["nvrtc"].__setitem__("sha256","0"*64)),
        ]
        for _,mutation in mutations:
            row=json.loads(json.dumps(result)); files=dict(artifacts); mutation(row,files); write_candidate(directory,row,files)
            if all(verifier.verify(directory,trust).values()): return positive_ok,negative_ok,False
        negative_bad=json.loads(json.dumps(negative)); negative_bad["compile_checks"]["handle_chain"]=False
        write_candidate(directory,negative_bad,negative_files)
        mutation_ok=not all(verifier.verify(directory,trust).values())
    return positive_ok,negative_ok,mutation_ok


def source_lock_contract(manifest) -> bool:
    authority=manifest["nc19_observed_source_lock_authority"]
    raw=base64.b64decode(authority["raw_base64"],validate=True)
    if len(raw)!=authority["bytes"] or sha(raw)!=authority["sha256"]: return False
    doc=json.loads(raw.decode("utf-8"))
    if doc!=authority["document"]: return False
    bootstrap=authority["bootstrap_document"]
    if bootstrap.get("source_lock_sha256")!=authority["sha256"] or bootstrap.get("source_lock_bytes")!=authority["bytes"]: return False
    descriptor=manifest["descriptors"][-1]; expected=descriptor["expected_absent_by_stage"]["implementation_freeze"]["paths"]
    required=descriptor["required_present_by_stage"]["implementation_freeze"]
    return doc["expected_absent"]==sorted(expected) and len(expected)==len(set(expected))==100 and len(required)==len(set(required))==57 and not(set(expected)&set(required)) and authority["path"] not in expected and len(doc["bindings"]["source_identity_entries"])==32


def source_bindings() -> bool:
    lock=json.loads(SOURCE_LOCK.read_text()); rows=lock["bindings"]
    if lock["revision"]!="NC19I2" or not rows or len(rows)!=len({x["path"] for x in rows}): return False
    for row in rows:
        path=ROOT/row["path"]
        if not path.is_file() or path.stat().st_size!=row["bytes"] or fsha(path)!=row["sha256"]: return False
    return True


def anchored_authorization(token: str) -> tuple[bool,dict]:
    try:
        raw=LOCK.read_bytes(); lock=json.loads(raw)
        rows=lock["bindings"]
        good=(token==ACK and lock["kind"]=="het_next_l0_ph1_nvidia_nc19i2_preflight_lock"
              and lock["revision"]=="NC19I2" and lock["preflight_open"] is True
              and lock["preflight_token"]==ACK and rows and len(rows)==len({x["path"] for x in rows}))
        for row in rows:
            path=ROOT/row["path"]
            good &= path.is_file() and path.stat().st_size==row["bytes"] and fsha(path)==row["sha256"]
        return bool(good),{"lock_bytes":len(raw),"lock_sha256":sha(raw),"binding_count":len(rows)}
    except Exception as exc:
        return False,{"error":f"{type(exc).__name__}:{exc}"}


def classifier_matrix(contract, manifest) -> tuple[bool,dict]:
    mismatches=[]
    for case in manifest["cases"]:
        actual=contract.evaluate_fixture_case(case,manifest)
        expected=case["expected_result"]
        projection={key:actual.get(key) for key in expected}
        if projection!=expected:
            mismatches.append({"name":case["name"],"expected":expected,"actual":projection})
    return not mismatches,{"executed":len(manifest["cases"]),"mismatches":mismatches[:8]}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("token"); args=parser.parse_args()
    auth,anchor=anchored_authorization(args.token)
    if not auth: return 2
    checks={name:False for name in CHECK_NAMES}; checks["authorization_anchor"]=auth
    try:
        checks["direct_bindings"]=(source_bindings() and fsha(REPORTS/"HET_NEXT_L0_PH1_NVIDIA_NC19_INDEPENDENT_DESIGN_AUDIT_2026-08-14.md")==AUDIT_SHA)
        checks["fixture_manifest"],manifest=check_manifest()
        before=set(sys.modules); contract=load_absolute(CONTRACT,"nc19i2_contract_fixture"); after=set(sys.modules)
        checks["fixture_manifest"] &= source_lock_contract(manifest)
        checks["manifest_classifier_1106"],classifier_detail=classifier_matrix(contract,manifest)
        checks["static_ast"]=(contract.__file__==str(CONTRACT) and static_ast()
                              and not any(name.casefold().startswith(("cupy","torch","numpy","nvcuda")) for name in after-before))
        checks["kernel_structure"]=kernel_structure()
        checks["abi_exact"]=abi_surface() and len(contract.NVRTC_OPS)==10
        checks["fake_success"],checks["fake_failure_matrix"]=fake_compile(contract)
        checks["environment_matrix"]=environment_matrix(contract)
        entries=[{"path":p,"type":"dir","size":0,"mtime_ns":0,"sha256":None} for p in (".","cuda_cache","nvrtc_cache","temp","tmp")]
        history=[{"stage":s,"entries":entries,"tree_digest":contract.cache_tree_digest(entries)} for s in ("pre_load",*contract.NVRTC_OPS,"post_release")]
        checks["cache_history"]=(len(history)==12 and contract.cache_history_digest(history)==contract.cache_history_digest(history))
        descriptor=next(x for x in manifest["descriptors"] if x["revision"]=="NC19")
        checks["topology_matrix"]=(len(contract.paths_for_revision(descriptor))==157 and checks["manifest_classifier_1106"])
        checks["transaction_matrix"]=transaction_matrix(contract)
        checks["failure_durability"]=checks["transaction_matrix"]
        verifier=load_absolute(VERIFIER,"nc19i2_independent_fixture")
        positive_ok,negative_ok,mutation_ok=verifier_production_mutations(contract,verifier)
        checks["verifier_positive"]=positive_ok; checks["verifier_negative"]=negative_ok
        checks["verifier_mutations"]=mutation_ok
        checks["no_payload_driver_device"]=checks["static_ast"] and no_device_callgraph()
    except Exception:
        pass
    result={"kind":"het_next_l0_ph1_nvidia_nc19i2_static_preflight_result","revision":"NC19I2","check_names":CHECK_NAMES,"checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(CHECK_NAMES),"authorization_anchor":anchor,"device_opened":False,"compiler_loaded":False,"driver_loaded":False,"payload_bytes_read":0}
    if OUT.exists(): return 3
    raw=contract.canonical(result) if 'contract' in locals() else (json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
    contract.atomic_create(OUT,raw)
    return 0 if result["pass"] else 3


if __name__=="__main__":
    raise SystemExit(main())
