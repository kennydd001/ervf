#!/usr/bin/env python3
"""NC19I0 static preflight.  No compiler, payload, Driver, runtime, or device."""
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
CONTRACT = SCRIPT.with_name("het_next_l0_ph1_nvidia_nc19i0_compile_contract.py")
RUNNER = SCRIPT.with_name("run_het_next_l0_ph1_nvidia_nc19i0_compile_only.py")
VERIFIER = SCRIPT.with_name("verify_het_next_l0_ph1_nvidia_nc19i0_compile_only.py")
KERNEL = SCRIPT.with_name("het_next_l0_ph1_nvidia_n5_kernels.cu")
MANIFEST = REPORTS / "het_next_l0_ph1_nvidia_nc19_source_lock_absent_set_fixture_manifest.json"
DESIGN_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19_design_lock.json"
SOURCE_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_source_lock.json"
LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_preflight_lock.json"
OUT = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_static_preflight_result.json"
FAILURES = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_static_preflight_failures"
QUARANTINE = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_static_preflight_quarantine"
ACK = "ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_STATIC_PREFLIGHT_ONCE"
MANIFEST_SHA = "481914c80b5dc8970c2217d8e08c783dc97aabe6a07d1d073fc45d8851709018"
AUDIT_SHA = "e8cad87b9fc8ede45df3d7cd41df6d0802d178cb65c3bf7814a36df4ab59275e"
CHECK_NAMES = [
    "authorization", "source_bindings", "design_audit", "manifest_identity",
    "manifest_cases", "source_lock_contract", "shared_contract_identity",
    "import_inert", "ast_surface", "kernel_structure", "fake_compile_success",
    "fake_compile_faults", "environment_matrix", "topology_matrix",
    "transaction_matrix", "verifier_production_mutations", "no_payload_device",
    "output_absent",
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
        self.cubin = b"\x7fELF" + b"\0" * 32 + b"q5_linear\0bf16_lut_activation\0"

    def call(self, op, **kwargs):
        if op == self.fault_op and self.fault_kind == "exception":
            raise OSError("injected")
        code = 7 if op == self.fault_op and self.fault_kind == "code" else 0
        row = {"code": code}
        if op == "nvrtcCreateProgram":
            self.handle = 0x123456789ABCDEF0 if self.fault_kind != "null" or op != self.fault_op else 0
            row["handle"] = self.handle
        elif op == "nvrtcGetProgramLog": row["log"] = b"\0"
        elif op == "nvrtcGetPTX": row["ptx"] = self.ptx
        elif op == "nvrtcGetCUBIN": row["cubin"] = self.cubin
        elif op == "nvrtcDestroyProgram": row["handle"] = 0
        return row


def check_manifest() -> tuple[bool, dict]:
    raw = MANIFEST.read_bytes()
    if len(raw) > 8*2**20 or sha(raw) != MANIFEST_SHA or raw[:3] != b"\xef\xbb\xbf":
        return False, {}
    manifest = json.loads(raw.decode("utf-8-sig"))
    names = [x["name"] for x in manifest["cases"]]
    good = len(names) == len(set(names)) == 1106
    for case in manifest["cases"]:
        entries = case["observed_entries"]
        digest = sha(json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode())
        good &= digest == case["observed_tree_digest"] and sum(x["size"] for x in entries) == case["observed_total_bytes"]
        for row in entries:
            if row["node_type"] == "file" and row["content_base64_or_null"] is not None:
                data = base64.b64decode(row["content_base64_or_null"], validate=True)
                good &= len(data) == row["size"] and sha(data) == row["sha256_or_null"]
                if row["parse_status"] == "valid_json":
                    good &= json.loads(data) == row["schema_key_values"]
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
    contract_imports = [n for n in ast.walk(trees[CONTRACT]) if isinstance(n,(ast.Import,ast.ImportFrom))]
    return len(subprocess_calls)==1 and not forbidden and all(a.name.split('.')[0] in {
        'base64','hashlib','json','os','re','shutil','time','uuid','pathlib','ctypes','__future__','annotations'
    } for n in contract_imports for a in n.names)


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


def fake_compile(contract) -> tuple[bool,bool]:
    success = contract.compile_with_adapter(FakeAdapter(), b"extern \"C\" __global__ void x(){}")
    checks = contract.validate_compile_evidence(success)
    success_ok = len(success["ledger"])==10 and all(checks.values())
    fault_ok = True
    for op in contract.NVRTC_OPS[:-1]:
        for kind in ("code","exception"):
            evidence = contract.compile_with_adapter(FakeAdapter(op,kind), b"x")
            fault_ok &= len(evidence["ledger"])==10 and [x["op"] for x in evidence["ledger"]]==list(contract.NVRTC_OPS) and evidence["primary"]["state"]=="failure"
    null = contract.compile_with_adapter(FakeAdapter("nvrtcCreateProgram","null"), b"x")
    fault_ok &= null["ledger"][1]["attempted"] and null["ledger"][-1]["attempted"] is False
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
        result={"kind":"fixture","revision":"NC19I0"}; bundle=contract.build_bundle(result,{"x.bin":b"x"},kind)
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
    return True


def verifier_production_mutations(contract, verifier) -> bool:
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
    result={
        "kind":"het_next_l0_ph1_nvidia_nc19i0_compile_only","revision":"NC19I0",
        "status":"compile_positive","terminal_valid":True,"positive":True,
        "authorization":{"auth_lock":{"execution_open":True},"ack":"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_COMPILE_ONLY_ONCE"},
        "invocation":{"direct":True,"raw":"base -I -B script ACK","native_argv":["base","-I","-B",str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i0_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_COMPILE_ONLY_ONCE"],"parse_error":None,"orig_argv":["base","-I","-B",str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i0_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_COMPILE_ONLY_ONCE"],"sys_argv":[str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i0_compile_only.py").resolve()),"ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_COMPILE_ONLY_ONCE"],"sys_executable":str((ROOT/".venv/Scripts/python.exe").resolve()),"base_executable":"base","prefix":str((ROOT/".venv").resolve()),"base_prefix":"base","name":"__main__","spec_is_none":True,"package":None,"file":str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i0_compile_only.py").resolve())},
        "source_identity":{"sha256":sha(source)},
        "toolchain_identity":{"nvrtc":{"sha256":"c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"},"builtins":{"sha256":"82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"},"header":{"sha256":"316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21"}},
        "options":list(contract.OPTIONS),"program_name":"het_next_l0_ph1_nvidia_nc19i0.cu",
        "create_operands":{"source_bytes":6173,"source_buffer_bytes":6174,"source_terminal_nul":1,"program_name_bytes":len(contract.PROGRAM_NAME),"program_name_buffer_bytes":len(contract.PROGRAM_NAME)+1,"num_headers":0,"headers":None,"include_names":None},
        "compile":compile_record,"compile_checks":contract.validate_compile_evidence(evidence),
        "loader":{"calling_convention":"WinDLL_kernel32_plus_cdecl_CFUNCTYPE","load_flags":0x1100,"nvrtc_version":[13,3],"nvrtc_abi_names":list(contract.NVRTC_OPS),"kernel32_abi":{"AddDllDirectory":{"argtypes":["c_wchar_p"],"restype":"c_void_p"},"RemoveDllDirectory":{"argtypes":["c_void_p"],"restype":"c_long"},"LoadLibraryExW":{"argtypes":["c_wchar_p","c_void_p","c_ulong"],"restype":"c_void_p"},"FreeLibrary":{"argtypes":["c_void_p"],"restype":"c_long"},"GetProcAddress":{"argtypes":["c_void_p","c_char_p"],"restype":"c_void_p"},"GetModuleHandleW":{"argtypes":["c_wchar_p"],"restype":"c_void_p"}},"invocation_abi":{"GetCommandLineW":{"argtypes":[],"restype":"c_wchar_p"},"LocalFree":{"argtypes":["c_void_p"],"restype":"c_void_p"}},"nvrtc_abi":{"nvrtcVersion":{"argtypes":["LP_c_int","LP_c_int"],"restype":"c_int"},"nvrtcCreateProgram":{"argtypes":["LP_c_void_p","c_char_p","c_char_p","c_int","LP_c_char_p","LP_c_char_p"],"restype":"c_int"},"nvrtcCompileProgram":{"argtypes":["c_void_p","c_int","LP_c_char_p"],"restype":"c_int"},"nvrtcGetProgramLogSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetProgramLog":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetPTXSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetPTX":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetCUBINSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetCUBIN":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcDestroyProgram":{"argtypes":["LP_c_void_p"],"restype":"c_int"}},"modules":{}},
        "ownership":[{"resource":"dll_directory_cookie","identity":1,"registered":True},{"resource":"nvrtc_hmodule","identity":2,"registered":True},{"resource":"nvrtc_program","identity":0x123456789ABCDEF0,"registered":True}],
        "cleanup":[{"resource":"wrappers","attempted":True,"code":0},{"resource":"nvrtc_hmodule","attempted":True,"code":0},{"resource":"dll_directory_cookie","attempted":True,"code":0},{"resource":"postrelease_module_check","attempted":True,"code":0}],
        "cache":{"private_root":"C:/fixture/private","environment_original":{k:{"present":False,"value":None} for k in contract.ENV_KEYS},"environment_applied":{},"environment_restore":[{"key":k,"attempted":True,"code":0} for k in reversed(contract.ENV_KEYS)],"history":history,"history_digest":contract.cache_history_digest(history)},
        "exclusions":{"payload_bytes_read":0,"nvcuda_driver_calls":0,"cuda_runtime_calls":0,"device_calls":0},
    }
    def write_candidate(directory, row, files):
        payload={"result.json":verifier.canonical(row),**files}
        manifest=verifier.canonical({"kind":"het_next_l0_ph1_nvidia_nc19i0_compile_only_manifest","revision":"NC19I0","files":[{"name":n,"bytes":len(v),"sha256":sha(v)} for n,v in sorted(payload.items())]})
        commit=verifier.canonical({"kind":"het_next_l0_ph1_nvidia_nc19i0_compile_only_commit","revision":"NC19I0","state":"complete","result_sha256":sha(payload["result.json"]),"manifest_sha256":sha(manifest)})
        for p in directory.iterdir(): p.unlink()
        for n,v in {**payload,"manifest.json":manifest,"commit.json":commit}.items(): (directory/n).write_bytes(v)
    with tempfile.TemporaryDirectory() as td:
        directory=Path(td); write_candidate(directory,result,artifacts)
        baseline=verifier.verify(directory)
        if set(baseline)!=set(verifier.EXPECTED_CHECKS) or not all(baseline.values()): return False
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
            if all(verifier.verify(directory).values()): return False
    return True


def source_lock_contract(manifest) -> bool:
    authority=manifest["nc19_observed_source_lock_authority"]; doc=authority["document"]
    descriptor=manifest["descriptors"][-1]; expected=descriptor["expected_absent_by_stage"]["implementation_freeze"]["paths"]
    required=descriptor["required_present_by_stage"]["implementation_freeze"]
    return doc["expected_absent"]==sorted(expected) and len(expected)==len(set(expected))==100 and len(required)==len(set(required))==57 and not(set(expected)&set(required)) and authority["path"] not in expected and len(doc["bindings"]["source_identity_entries"])==32


def source_bindings() -> bool:
    lock=json.loads(SOURCE_LOCK.read_text()); rows=lock["bindings"]["source_identity_entries"]
    if lock["revision"]!="NC19I0" or len(rows)!=len({x["path"] for x in rows}): return False
    for row in rows:
        path=ROOT/row["path"]
        if not path.is_file() or path.stat().st_size!=row["bytes"] or fsha(path)!=row["sha256"]: return False
    return True


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("token"); args=parser.parse_args()
    try: lock=json.loads(LOCK.read_text()); auth=lock["preflight_open"] is True and lock["preflight_token"]==args.token==ACK
    except Exception: auth=False
    if not auth: return 2
    checks={name:False for name in CHECK_NAMES}; checks["authorization"]=auth
    try:
        checks["source_bindings"]=source_bindings()
        checks["design_audit"]=fsha(REPORTS/"HET_NEXT_L0_PH1_NVIDIA_NC19_INDEPENDENT_DESIGN_AUDIT_2026-08-14.md")==AUDIT_SHA
        checks["manifest_identity"],manifest=check_manifest(); checks["manifest_cases"]=checks["manifest_identity"]
        checks["source_lock_contract"]=source_lock_contract(manifest)
        before=set(sys.modules); contract=load_absolute(CONTRACT,"nc19i0_contract_fixture"); after=set(sys.modules)
        checks["shared_contract_identity"]=contract.__file__==str(CONTRACT) and len(contract.EXPORTS)==11 and all(x.__module__=="nc19i0_contract_fixture" for x in contract.EXPORTS)
        checks["import_inert"]=not any(name.casefold().startswith(("cupy","torch","numpy","nvcuda")) for name in after-before)
        checks["ast_surface"]=static_ast(); checks["kernel_structure"]=kernel_structure()
        checks["fake_compile_success"],checks["fake_compile_faults"]=fake_compile(contract)
        checks["environment_matrix"]=environment_matrix(contract)
        descriptor=manifest["descriptors"][-1]; checks["topology_matrix"]=len(contract.paths_for_revision(descriptor))==157
        checks["transaction_matrix"]=transaction_matrix(contract)
        verifier=load_absolute(VERIFIER,"nc19i0_independent_fixture")
        checks["verifier_production_mutations"]=verifier_production_mutations(contract,verifier)
        combined=(RUNNER.read_text()+CONTRACT.read_text()+VERIFIER.read_text()).casefold()
        checks["no_payload_device"]=checks["ast_surface"] and not any(x in combined for x in ("safetensors","model.safetensors","cuinit(","nvcuda.dll"))
        checks["output_absent"]=not any(p.exists() for p in (OUT,FAILURES,QUARANTINE,REPORTS/"het_next_l0_ph1_nvidia_nc19i0_compile_only",REPORTS/"het_next_l0_ph1_nvidia_nc19i0_compile_only_negative"))
    except Exception:
        pass
    result={"kind":"het_next_l0_ph1_nvidia_nc19i0_static_preflight_result","revision":"NC19I0","check_names":CHECK_NAMES,"checks":checks,"pass":all(checks.values()),"passed":sum(checks.values()),"total":len(CHECK_NAMES),"device_opened":False,"compiler_loaded":False,"driver_loaded":False,"payload_bytes_read":0}
    if OUT.exists(): return 3
    raw=contract.canonical(result) if 'contract' in locals() else (json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
    with OUT.open("xb") as h: h.write(raw); h.flush(); os.fsync(h.fileno())
    return 0 if result["pass"] else 3


if __name__=="__main__":
    raise SystemExit(main())
