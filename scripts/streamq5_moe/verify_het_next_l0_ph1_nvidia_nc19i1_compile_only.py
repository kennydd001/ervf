#!/usr/bin/env python3
"""Independent NC19I1 compile artifact verifier; imports no candidate module."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
SCRIPT = Path(__file__).resolve()
LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_verifier_lock.json"
SOURCE_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_source_lock.json"
PREFLIGHT_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_preflight_lock.json"
AUTH_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_authorization_bootstrap_lock.json"
PREFLIGHT_RESULT = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_static_preflight_result.json"
POSITIVE = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_compile_only"
NEGATIVE = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_compile_only_negative"
OUT = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_independent_verification"
FAILURES = REPORTS / "het_next_l0_ph1_nvidia_nc19i1_independent_verification_failures"
TOKEN = "ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I1_VERIFY_ONCE"
PRECOMMIT = "NC19I1_PRECOMMIT_INDEPENDENT_VERIFY"
SOURCE_SHA = "9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781"
OPS = ["nvrtcVersion", "nvrtcCreateProgram", "nvrtcCompileProgram",
       "nvrtcGetProgramLogSize", "nvrtcGetProgramLog", "nvrtcGetPTXSize",
       "nvrtcGetPTX", "nvrtcGetCUBINSize", "nvrtcGetCUBIN", "nvrtcDestroyProgram"]
OPTIONS = ["--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true",
           "--ftz=false", "--gpu-architecture=sm_120", "--device-as-default-execution-space"]
EXPECTED_CHECKS = [
    "bundle", "result_schema", "terminal", "source", "options", "create_operands",
    "ledger", "program_identity", "log", "ptx", "ptx_entries", "cubin",
    "artifact_manifest", "loader", "toolchain", "ownership", "cleanup",
    "environment", "cache_history", "cache_containment", "exclusions",
    "no_driver_runtime", "authorization", "invocation", "provenance",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory: Path, kind: str) -> tuple[bool, dict[str, bytes]]:
    try:
        entries = list(directory.iterdir())
        if any(not p.is_file() for p in entries):
            return False, {}
        if any(p.stat().st_size > 64*2**20 for p in entries) or sum(p.stat().st_size for p in entries) > 64*2**20:
            return False, {}
        raw = {p.name: p.read_bytes() for p in entries}
        allowed = {"result.json", "manifest.json", "commit.json", "source.cu",
                   "build.log", "ptx.bin", "cubin.bin"}
        if not {"result.json", "manifest.json", "commit.json", "source.cu"} <= set(raw) or not set(raw) <= allowed:
            return False, raw
        if sum(map(len, raw.values())) > 64 * 2**20:
            return False, raw
        manifest = json.loads(raw["manifest.json"]); commit = json.loads(raw["commit.json"])
        payload = {k: v for k, v in raw.items() if k not in {"manifest.json", "commit.json"}}
        expected = {"kind": kind + "_manifest", "revision": "NC19I1",
                    "files": [{"name": n, "bytes": len(v), "sha256": sha(v)}
                              for n, v in sorted(payload.items())]}
        good = manifest == expected and commit == {
            "kind": kind + "_commit", "revision": "NC19I1", "state": "complete",
            "result_sha256": sha(raw["result.json"]),
            "manifest_sha256": sha(raw["manifest.json"]),
        }
        return good, raw
    except Exception:
        return False, {}


def _history_digest(history) -> str:
    return sha(json.dumps(history, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode())


def _tree_digest(entries) -> str:
    raw=b"".join((f"{x['path']}\0{x['type']}\0{x['size']}\0{x['mtime_ns']}\0{x['sha256'] or ''}\n").encode()
                 for x in entries)
    return sha(raw)


def _elf_symbols(raw: bytes) -> set[str] | None:
    import struct
    try:
        if not (1<len(raw)<=32*2**20 and raw[:6]==b"\x7fELF\x02\x01"): return None
        shoff=struct.unpack_from("<Q",raw,40)[0]; entsize,count,strndx=struct.unpack_from("<HHH",raw,58)
        if entsize!=64 or not(0<count<=4096) or strndx>=count or shoff+count*64>len(raw): return None
        sections=[struct.unpack_from("<IIQQQQIIQQ",raw,shoff+i*64) for i in range(count)]
        symbols=set()
        for sec in sections:
            if sec[1] not in (2,11): continue
            if sec[6]>=count or sec[9]!=24 or sec[4]+sec[5]>len(raw): return None
            strings_hdr=sections[sec[6]]; strings=raw[strings_hdr[4]:strings_hdr[4]+strings_hdr[5]]
            for pos in range(sec[4],sec[4]+sec[5],24):
                noff,info=struct.unpack_from("<IB",raw,pos)
                if noff and (info&15)==2:
                    end=strings.index(0,noff); symbols.add(strings[noff:end].decode("utf-8","strict"))
        return symbols
    except (IndexError,UnicodeError,ValueError,struct.error): return None


def _binding_rows(lock: dict) -> list[dict]:
    rows=lock.get("bindings",[])
    return rows if isinstance(rows,list) else []


def _rehash_lock(lock: dict) -> bool:
    rows=_binding_rows(lock)
    if not rows or len(rows)!=len({x.get("path") for x in rows}): return False
    for row in rows:
        if set(row)!={"path","bytes","sha256"}: return False
        path=ROOT/row["path"]
        if not path.is_file() or path.stat().st_size!=row["bytes"] or fsha(path)!=row["sha256"]: return False
    return True


def verify(directory: Path, fixture_trust: dict | None = None) -> dict:
    directory=Path(directory).resolve()
    if fixture_trust is None and directory not in {POSITIVE.resolve(),NEGATIVE.resolve()} and ".inprogress." not in directory.name:
        return {name:False for name in EXPECTED_CHECKS}
    try:
        preliminary=json.loads((directory/"result.json").read_text())
        negative=preliminary.get("status")=="compile_valid_negative"
    except Exception:
        negative=False
    kind = "het_next_l0_ph1_nvidia_nc19i1_compile_only" + ("_negative" if negative else "")
    bundle_ok, raw = verify_bundle(directory, kind)
    checks = {name: False for name in EXPECTED_CHECKS}
    checks["bundle"] = bundle_ok
    try:
        result = json.loads(raw["result.json"])
        evidence = result["compile"]; ledger = evidence["ledger"]
        source=raw["source.cu"]; log=raw.get("build.log"); ptx=raw.get("ptx.bin"); cubin=raw.get("cubin.bin")
        checks["result_schema"] = set(result) == {
            "kind", "revision", "status", "terminal_valid", "positive", "authorization",
            "invocation", "source_identity", "toolchain_identity", "options", "program_name",
            "create_operands", "compile", "compile_checks", "loader", "ownership", "cleanup",
            "cache", "exclusions",
        } and result["kind"] == kind and result["revision"] == "NC19I1"
        checks["terminal"] = ((not negative and result["status"] == "compile_positive" and result["positive"] is True and evidence["primary"] == {"state":"none","value":None}) or (negative and result["status"]=="compile_valid_negative" and result["positive"] is False and evidence["primary"]["state"]=="failure")) and result["terminal_valid"] is True
        expected_source=str((ROOT/"scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu").resolve())
        checks["source"] = len(source)==6173 and sha(source)==SOURCE_SHA and result["source_identity"]=={"path":expected_source,"bytes":6173,"sha256":SOURCE_SHA}
        checks["options"] = result["options"] == OPTIONS
        checks["create_operands"] = result["program_name"] == "het_next_l0_ph1_nvidia_nc19i1.cu" and result["create_operands"] == {
            "source_bytes":6173,"source_buffer_bytes":6174,"source_terminal_nul":1,
            "program_name_bytes":len(b"het_next_l0_ph1_nvidia_nc19i1.cu"),
            "program_name_buffer_bytes":len(b"het_next_l0_ph1_nvidia_nc19i1.cu")+1,
            "num_headers":0,"headers":None,"include_names":None,
        }
        checks["ledger"] = len(ledger) == 10 and [x["sequence"] for x in ledger] == list(range(10)) and [x["op"] for x in ledger] == OPS and ((not negative and all(x["attempted"] is True and x["code"] == 0 for x in ledger)) or (negative and (any(x["attempted"] and x["code"] not in (0,"not_attempted") for x in ledger) or str(evidence["primary"].get("value","")).startswith("artifact_contract:")) and all(x["attempted"] or x["code"]=="not_attempted" for x in ledger)))
        nonzero = {x["handle_after"] for x in ledger if x["handle_after"]}
        checks["program_identity"] = len(nonzero) == 1 and ledger[1]["handle_before"] == 0 and ledger[1]["handle_after"] != 0 and ledger[-1]["handle_before"] == ledger[1]["handle_after"] and ledger[-1]["handle_after"] == 0
        checks["log"] = log is not None and 1 <= len(log) <= 4*2**20 and log.endswith(b"\0") and b"\0" not in log[:-1]
        checks["ptx"] = (negative and ptx is None) or (ptx is not None and 1 < len(ptx) <= 16*2**20 and ptx.endswith(b"\0") and b"\0" not in ptx[:-1])
        text = ptx[:-1].decode("utf-8", "strict") if ptx is not None and checks["ptx"] else ""
        checks["ptx_entries"] = (negative and ptx is None) or (text.count(".version")==1 and text.count(".target sm_120")==1 and text.count(".address_size 64")==1 and len(re.findall(r"(?m)^\s*\.visible\s+\.entry\s+q5_linear\b", text))==1 and len(re.findall(r"(?m)^\s*\.visible\s+\.entry\s+bf16_lut_activation\b", text))==1 and len(re.findall(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+",text))==2 and not any(x in text.lower() for x in (".ftz", "approx", ".extern .func")))
        checks["cubin"] = (negative and cubin is None) or (cubin is not None and _elf_symbols(cubin)=={"q5_linear","bf16_lut_activation"})
        expected_artifacts = {n:v for n,v in (("log",log),("ptx",ptx),("cubin",cubin)) if v is not None}
        checks["artifact_manifest"] = evidence["artifacts"] == {
            n:{"bytes":len(v),"sha256":sha(v)} for n,v in expected_artifacts.items()
        }
        loader = result["loader"]
        expected_kernel={"AddDllDirectory":{"argtypes":["c_wchar_p"],"restype":"c_void_p"},"RemoveDllDirectory":{"argtypes":["c_void_p"],"restype":"c_long"},"LoadLibraryExW":{"argtypes":["c_wchar_p","c_void_p","c_ulong"],"restype":"c_void_p"},"FreeLibrary":{"argtypes":["c_void_p"],"restype":"c_long"},"GetProcAddress":{"argtypes":["c_void_p","c_char_p"],"restype":"c_void_p"},"GetModuleHandleW":{"argtypes":["c_wchar_p"],"restype":"c_void_p"},"GetModuleFileNameW":{"argtypes":["c_void_p","LP_c_wchar","c_ulong"],"restype":"c_ulong"}}
        expected_nvrtc={"nvrtcVersion":{"argtypes":["LP_c_int","LP_c_int"],"restype":"c_int"},"nvrtcCreateProgram":{"argtypes":["LP_c_void_p","c_char_p","c_char_p","c_int","LP_c_char_p","LP_c_char_p"],"restype":"c_int"},"nvrtcCompileProgram":{"argtypes":["c_void_p","c_int","LP_c_char_p"],"restype":"c_int"},"nvrtcGetProgramLogSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetProgramLog":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetPTXSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetPTX":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcGetCUBINSize":{"argtypes":["c_void_p","LP_c_ulonglong"],"restype":"c_int"},"nvrtcGetCUBIN":{"argtypes":["c_void_p","c_void_p"],"restype":"c_int"},"nvrtcDestroyProgram":{"argtypes":["LP_c_void_p"],"restype":"c_int"}}
        checks["loader"] = loader["calling_convention"] == "WinDLL_kernel32_plus_cdecl_CFUNCTYPE" and loader["load_flags"] == 0x1100 and loader["nvrtc_version"] == [13,3] and loader["kernel32_abi"]==expected_kernel and loader["invocation_abi"]=={"GetCommandLineW":{"argtypes":[],"restype":"c_wchar_p"},"LocalFree":{"argtypes":["c_void_p"],"restype":"c_void_p"}} and loader["nvrtc_abi"]==expected_nvrtc and loader["nvrtc_abi_names"]==OPS
        ti = result["toolchain_identity"]
        expected_toolchain={"nvrtc":(".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll","c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"),"builtins":(".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll","82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"),"header":(".venv/Lib/site-packages/nvidia/cu13/include/nvrtc.h","316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21")}
        checks["toolchain"] = all(ti[k]["path"]==str((ROOT/p).resolve()) and ti[k]["bytes"]>0 and ti[k]["sha256"]==h for k,(p,h) in expected_toolchain.items()) and ti["python"]["path"]==str((ROOT/".venv/Scripts/python.exe").resolve())
        ownership = result["ownership"]
        checks["ownership"] = [x["resource"] for x in ownership] == ["dll_directory_cookie","nvrtc_hmodule","nvrtc_program"] and all(x["identity"] and x["registered"] for x in ownership) and loader["modules"]["during"]["nvrtc64_130_0.dll"]["handle"]==ownership[1]["identity"]
        cleanup = result["cleanup"]
        checks["cleanup"] = [x["resource"] for x in cleanup] == ["nvrtc_program","wrappers","nvrtc_hmodule","dll_directory_cookie","postrelease_module_check"] and all(x["attempted"] and x["code"] == 0 for x in cleanup)
        cache = result["cache"]; history = cache["history"]
        checks["environment"] = list(cache["environment_original"]) == ["CUDA_CACHE_DISABLE","CUDA_CACHE_MAXSIZE","CUDA_CACHE_PATH","TMP","TEMP","NVRTC_CACHE_PATH"] and [x["key"] for x in cache["environment_restore"]] == list(reversed(list(cache["environment_original"]))) and all(x["code"] == 0 for x in cache["environment_restore"])
        checks["cache_history"] = len(history) == 12 and [x["stage"] for x in history] == ["pre_load", *OPS, "post_release"] and cache["history_digest"] == _history_digest(history) and all(h["tree_digest"]==_tree_digest(h["entries"]) for h in history)
        exact_dirs={".","cuda_cache","tmp","temp","nvrtc_cache"}
        checks["cache_containment"] = all(len(h["entries"])==5 and {e["path"] for e in h["entries"]}==exact_dirs and all(e["type"]=="dir" and e["sha256"] is None for e in h["entries"]) for h in history) and all(Path(v).is_absolute() and str(Path(v)).casefold().startswith(str(Path(cache["private_root"]).resolve()).casefold()) for k,v in cache["environment_applied"].items() if k in {"CUDA_CACHE_PATH","TMP","TEMP","NVRTC_CACHE_PATH"})
        checks["exclusions"] = result["exclusions"] == {"payload_bytes_read":0,"nvcuda_driver_calls":0,"cuda_runtime_calls":0,"device_calls":0}
        checks["no_driver_runtime"] = all("nvcuda" not in json.dumps(result[k]).lower() and "cudart" not in json.dumps(result[k]).lower() and "cupy" not in json.dumps(result[k]).lower() for k in ("loader","ownership","cleanup"))
        if fixture_trust is None:
            auth=json.loads(AUTH_LOCK.read_text()); plock=json.loads(PREFLIGHT_LOCK.read_text()); slock=json.loads(SOURCE_LOCK.read_text()); presult=json.loads(PREFLIGHT_RESULT.read_text())
            preflight_sha=fsha(PREFLIGHT_RESULT); provenance_ok=_rehash_lock(slock) and _rehash_lock(plock) and _rehash_lock(auth)
        else:
            auth=fixture_trust["auth_lock"]; plock=fixture_trust["preflight_lock"]; slock=fixture_trust["source_lock"]; presult=fixture_trust["preflight_result"]
            preflight_sha=fixture_trust["preflight_result_sha256"]; provenance_ok=fixture_trust["provenance_ok"] is True
        checks["authorization"] = (result["authorization"]["auth_lock"]==auth and auth["execution_open"] is True and result["authorization"]["ack"]=="ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I1_COMPILE_ONLY_ONCE" and auth["preflight_result_sha256"]==preflight_sha and presult["pass"] is True)
        inv=result["invocation"]; expected_script=str((ROOT/"scripts/streamq5_moe/run_het_next_l0_ph1_nvidia_nc19i1_compile_only.py").resolve()); expected_ack="ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I1_COMPILE_ONLY_ONCE"
        checks["invocation"] = inv["direct"] is True and inv["parse_error"] is None and inv["sys_argv"]==[expected_script,expected_ack] and inv["orig_argv"][1:]==["-I","-B",expected_script,expected_ack] and inv["native_argv"][1:]==["-I","-B",expected_script,expected_ack] and inv["name"]=="__main__" and inv["spec_is_none"] is True and inv["package"] in (None,"") and inv["file"]==expected_script
        current = {x["path"]:x for x in _binding_rows(slock)}
        checks["provenance"] = (slock["revision"]=="NC19I1" and provenance_ok
                                and current["scripts/streamq5_moe/verify_het_next_l0_ph1_nvidia_nc19i1_compile_only.py"]["sha256"]==fsha(SCRIPT))
    except Exception:
        pass
    return checks


def _atomic_output(path: Path, result: dict):
    raw = canonical(result); temp = path.with_name(path.name + f".inprogress.{os.getpid()}.{uuid.uuid4().hex[:16]}")
    temp.mkdir(parents=False)
    try:
        files = {"result.json":raw}
        manifest = canonical({"kind":"het_next_l0_ph1_nvidia_nc19i1_verification_manifest","revision":"NC19I1","files":[{"name":"result.json","bytes":len(raw),"sha256":sha(raw)}]})
        commit = canonical({"kind":"het_next_l0_ph1_nvidia_nc19i1_verification_commit","revision":"NC19I1","state":"complete","result_sha256":sha(raw),"manifest_sha256":sha(manifest)})
        for name,data in (("result.json",raw),("manifest.json",manifest),("commit.json",commit)):
            with (temp/name).open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        _flush_directory(temp)
        os.replace(temp,path)
        _flush_directory(path.parent)
    except Exception:
        if temp.exists():
            for p in temp.iterdir(): p.unlink()
            temp.rmdir()
        raise


def _flush_directory(path: Path):
    if os.name!="nt":
        fd=os.open(path,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return
    import ctypes as c
    k=c.WinDLL("kernel32",use_last_error=True)
    k.CreateFileW.argtypes=[c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_void_p,c.c_uint32,c.c_uint32,c.c_void_p]; k.CreateFileW.restype=c.c_void_p
    k.FlushFileBuffers.argtypes=[c.c_void_p]; k.FlushFileBuffers.restype=c.c_int32
    k.CloseHandle.argtypes=[c.c_void_p]; k.CloseHandle.restype=c.c_int32
    handle=k.CreateFileW(str(Path(path).resolve()),0x80000000,7,None,3,0x02000000,None)
    if handle in (None,c.c_void_p(-1).value): raise OSError(c.get_last_error(),"CreateFileW")
    try:
        if not k.FlushFileBuffers(handle): raise OSError(c.get_last_error(),"FlushFileBuffers")
    finally:
        if not k.CloseHandle(handle): raise OSError(c.get_last_error(),"CloseHandle")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", type=Path)
    parser.add_argument("--mode", choices=("precommit","postcommit"), required=True)
    parser.add_argument("--no-write", action="store_true"); parser.add_argument("--token", required=True)
    args = parser.parse_args(); candidate = (args.candidate or POSITIVE).resolve()
    if args.mode == "precommit":
        authorized = args.no_write and args.token == PRECOMMIT and len(sys.argv) == 8
    else:
        try:
            lock = json.loads(LOCK.read_text()); authorized = (not args.no_write and args.token == TOKEN and lock["verification_open"] is True and lock["verification_token"] == TOKEN)
        except Exception:
            authorized = False
    if not authorized:
        return 2
    checks = verify(candidate)
    result = {"kind":"het_next_l0_ph1_nvidia_nc19i1_independent_verification",
              "revision":"NC19I1","candidate":str(candidate),"mode":args.mode,
              "check_names":EXPECTED_CHECKS,"checks":checks,"pass":all(checks.values()),
              "passed":sum(checks.values()),"total":len(EXPECTED_CHECKS),
              "device_opened":False,"compiler_loaded":False,"payload_bytes_read":0}
    if args.mode == "precommit":
        sys.stdout.write(json.dumps(result,sort_keys=True,separators=(",",":")))
    elif result["pass"]:
        if OUT.exists(): return 3
        _atomic_output(OUT,result)
    else:
        failure={"kind":"het_next_l0_ph1_nvidia_nc19i1_verifier_protocol_negative",
                 "revision":"NC19I1","status":"verifier_protocol_negative",
                 "candidate":str(candidate),"check_names":EXPECTED_CHECKS,
                 "false_checks":[name for name,value in checks.items() if not value],
                 "device_opened":False,"compiler_loaded":False,"payload_bytes_read":0}
        FAILURES.mkdir(parents=True,exist_ok=True)
        target=FAILURES/f"attempt.{uuid.uuid4().hex}.json"
        with target.open("xb") as h: h.write(canonical(failure)); h.flush(); os.fsync(h.fileno())
        _flush_directory(FAILURES)
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
