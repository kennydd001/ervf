#!/usr/bin/env python3
"""Standalone Intel host-USM execution backend. Import is device-free."""
from __future__ import annotations

import ctypes as C
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPILED = ROOT / "reports/streamq5_moe/het_next_l0_ph1_intel_compile_r2a"
SOURCE = COMPILED / "intel_source.cl"
BINARY = COMPILED / "intel_program.bin"
SOURCE_SHA = "f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21"
BINARY_SHA = "8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629"
BINARY_BYTES = 186_352
CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DRIVER_VERSION, CL_DEVICE_EXTENSIONS = 0x102B, 0x102C, 0x102D, 0x1030
CL_DEVICE_PCI_BUS_INFO_KHR, CL_CONTEXT_PLATFORM = 0x410F, 0x1084
CL_PROGRAM_BUILD_LOG, CL_PROGRAM_BINARY_SIZES, CL_PROGRAM_BINARIES = 0x1183, 0x1165, 0x1166
CL_MEM_ALLOC_TYPE_INTEL, CL_MEM_ALLOC_BASE_PTR_INTEL, CL_MEM_ALLOC_SIZE_INTEL, CL_MEM_TYPE_HOST_INTEL = 0x419A, 0x419B, 0x419C, 0x4197

BUFFER_TABLE = (("gate_record",675840),("up_record",675840),("down_record",675840),("natural_input",4096),("silu_lut",131072),("gate",1024),("up",1024),("silu",1024),("activation",1024),("down",4096),("gate_counters",2048),("up_counters",2048),("activation_counters",2048),("down_counters",8192))
ARGUMENT_MAPS = (("gate_linear",("gate_record","natural_input","gate","gate_counters")),("up_linear",("up_record","natural_input","up","up_counters")),("activation",("gate","up","silu_lut","silu","activation","activation_counters")),("down_linear",("down_record","activation","down","down_counters")))
LAUNCHES = (("gate_linear",4096,256),("up_linear",4096,256),("activation",512,256),("down_linear",16384,256))
if len(BUFFER_TABLE)!=14 or sum(x[1] for x in BUFFER_TABLE)!=2_185_216 or sum(len(x[1]) for x in ARGUMENT_MAPS)!=18 or len(LAUNCHES)!=4:
    raise RuntimeError("frozen_cardinality")


def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def file_sha(path: Path) -> str: return sha(path.read_bytes())
def check(code: int, op: str) -> None:
    if int(code)!=0: raise RuntimeError(f"{op}:{int(code)}")


class PCI(C.Structure): _fields_=[("domain",C.c_uint),("bus",C.c_uint),("device",C.c_uint),("function",C.c_uint)]
class ExecutionFailure(RuntimeError):
    def __init__(self,message,evidence): super().__init__(message); self.evidence=evidence


class Backend:
    def __init__(self):
        self.lib=self.context=self.queue=self.program=None; self.kernels=[]; self.allocations=[]; self.ledger=[]; self.cleanup_errors=[]

    def bind(self):
        l=self.lib;p,u,z,i=C.c_void_p,C.c_uint,C.c_size_t,C.c_int
        l.clGetPlatformIDs.argtypes=[u,C.POINTER(p),C.POINTER(u)];l.clGetPlatformIDs.restype=i
        l.clGetDeviceIDs.argtypes=[p,C.c_ulonglong,u,C.POINTER(p),C.POINTER(u)];l.clGetDeviceIDs.restype=i
        l.clGetDeviceInfo.argtypes=[p,u,z,p,C.POINTER(z)];l.clGetDeviceInfo.restype=i
        l.clGetExtensionFunctionAddressForPlatform.argtypes=[p,C.c_char_p];l.clGetExtensionFunctionAddressForPlatform.restype=p
        l.clCreateContext.argtypes=[C.POINTER(C.c_ssize_t),u,C.POINTER(p),p,p,C.POINTER(i)];l.clCreateContext.restype=p
        l.clCreateCommandQueue.argtypes=[p,p,C.c_ulonglong,C.POINTER(i)];l.clCreateCommandQueue.restype=p
        l.clCreateProgramWithBinary.argtypes=[p,u,C.POINTER(p),C.POINTER(z),C.POINTER(C.POINTER(C.c_ubyte)),C.POINTER(i),C.POINTER(i)];l.clCreateProgramWithBinary.restype=p
        l.clBuildProgram.argtypes=[p,u,C.POINTER(p),C.c_char_p,p,p];l.clBuildProgram.restype=i
        l.clGetProgramBuildInfo.argtypes=[p,p,u,z,p,C.POINTER(z)];l.clGetProgramBuildInfo.restype=i
        l.clGetProgramInfo.argtypes=[p,u,z,p,C.POINTER(z)];l.clGetProgramInfo.restype=i
        l.clCreateKernel.argtypes=[p,C.c_char_p,C.POINTER(i)];l.clCreateKernel.restype=p
        l.clEnqueueNDRangeKernel.argtypes=[p,p,u,p,C.POINTER(z),C.POINTER(z),u,p,p];l.clEnqueueNDRangeKernel.restype=i
        l.clFinish.argtypes=[p];l.clFinish.restype=i
        for n in ("clReleaseKernel","clReleaseProgram","clReleaseCommandQueue","clReleaseContext"):
            getattr(l,n).argtypes=[p];getattr(l,n).restype=i

    def info(self,d,param):
        n=C.c_size_t();check(self.lib.clGetDeviceInfo(d,param,0,None,C.byref(n)),"info_size");b=C.create_string_buffer(n.value);check(self.lib.clGetDeviceInfo(d,param,n.value,b,None),"info");return b.value.decode()

    def select(self):
        n=C.c_uint();check(self.lib.clGetPlatformIDs(0,None,C.byref(n)),"platform_count");ps=(C.c_void_p*n.value)();check(self.lib.clGetPlatformIDs(n.value,ps,None),"platforms");found=[]
        for rp in ps:
            p=C.c_void_p(rp);m=C.c_uint()
            if self.lib.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,None,C.byref(m)):continue
            ds=(C.c_void_p*m.value)();check(self.lib.clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,m.value,ds,None),"devices")
            for rd in ds:
                d=C.c_void_p(rd);ext=self.info(d,CL_DEVICE_EXTENSIONS).split()
                if self.info(d,CL_DEVICE_NAME)=="Intel(R) Arc(TM) Pro 140T GPU (32GB)" and "cl_intel_unified_shared_memory" in ext:found.append((p,d,ext))
        if len(found)!=1:raise RuntimeError(f"intel_cardinality:{len(found)}")
        p,d,ext=found[0];pci=PCI();check(self.lib.clGetDeviceInfo(d,CL_DEVICE_PCI_BUS_INFO_KHR,C.sizeof(pci),C.byref(pci),None),"pci")
        ident={"name":self.info(d,CL_DEVICE_NAME),"vendor":self.info(d,CL_DEVICE_VENDOR),"driver":self.info(d,CL_DRIVER_VERSION),"pci":f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function}","extensions":ext}
        if ident["vendor"]!="Intel(R) Corporation" or ident["driver"]!="32.0.101.8517" or ident["pci"]!="0000:00:02.0":raise RuntimeError("identity")
        self.ledger.append({"op":"identity","identity":ident});return p,d,ident

    def extension(self,p,name,proto):
        a=self.lib.clGetExtensionFunctionAddressForPlatform(p,name.encode())
        if not a:raise RuntimeError("extension:"+name)
        return proto(a)

    def close(self):
        for name,pointer,_size,free in reversed(self.allocations):
            try:code=int(free(self.context,C.c_void_p(pointer)));self.ledger.append({"op":"release","name":"usm:"+name,"code":code});check(code,"release_usm")
            except Exception as exc:self.cleanup_errors.append(f"usm:{name}:{exc}")
        for name,handle in reversed(self.kernels):
            try:code=int(self.lib.clReleaseKernel(handle));self.ledger.append({"op":"release","name":"kernel:"+name,"code":code});check(code,"release_kernel")
            except Exception as exc:self.cleanup_errors.append(f"kernel:{name}:{exc}")
        for name,handle,fn in (("program",self.program,"clReleaseProgram"),("queue",self.queue,"clReleaseCommandQueue"),("context",self.context,"clReleaseContext")):
            if handle:
                try:code=int(getattr(self.lib,fn)(handle));self.ledger.append({"op":"release","name":name,"code":code});check(code,"release:"+name)
                except Exception as exc:self.cleanup_errors.append(f"{name}:{exc}")
        self.ledger.append({"op":"cleanup","cleanup_complete":not self.cleanup_errors,"errors":self.cleanup_errors,"release_attempts":sum(r.get("op")=="release" for r in self.ledger),"live_owned_resources":0 if not self.cleanup_errors else None})

    def run(self,records,input_bytes,lut,authorization):
        ev={"authorization":authorization,"ledger":self.ledger,"source_sha256":file_sha(SOURCE),"binary_sha256":file_sha(BINARY)};outputs={};identity={}
        try:
            if set(records)!={"gate","up","down"} or any(len(v)!=675840 for v in records.values()) or len(input_bytes)!=4096 or len(lut)!=131072:raise RuntimeError("payload_shape")
            if file_sha(SOURCE)!=SOURCE_SHA or BINARY.stat().st_size!=BINARY_BYTES or file_sha(BINARY)!=BINARY_SHA:raise RuntimeError("compile_artifact")
            self.lib=C.WinDLL("OpenCL.dll");self.bind();platform,device,identity=self.select();err=C.c_int();props=(C.c_ssize_t*3)(CL_CONTEXT_PLATFORM,int(platform.value),0);devices=(C.c_void_p*1)(device.value)
            self.context=self.lib.clCreateContext(props,1,devices,None,None,C.byref(err));check(err.value,"context");self.ledger.append({"op":"context_create","pointer":int(self.context)})
            self.queue=self.lib.clCreateCommandQueue(self.context,device,0,C.byref(err));check(err.value,"queue");self.ledger.append({"op":"queue_create","pointer":int(self.queue),"in_order":True})
            binary=BINARY.read_bytes();buf=(C.c_ubyte*len(binary)).from_buffer_copy(binary);bufp=C.cast(buf,C.POINTER(C.c_ubyte));bufs=(C.POINTER(C.c_ubyte)*1)(bufp);sizes=(C.c_size_t*1)(len(binary));status=C.c_int()
            self.program=self.lib.clCreateProgramWithBinary(self.context,1,devices,sizes,bufs,C.byref(status),C.byref(err));check(err.value,"program");check(status.value,"binary_status");self.ledger.append({"op":"program_create_binary","bytes":len(binary),"sha256":sha(binary)})
            check(self.lib.clBuildProgram(self.program,1,devices,b"",None,None),"build_binary")
            for name in ("gate_linear","up_linear","activation","down_linear"):
                k=self.lib.clCreateKernel(self.program,name.encode(),C.byref(err));check(err.value,"kernel");self.kernels.append((name,k));self.ledger.append({"op":"kernel_create","name":name,"pointer":int(k)})
            host_alloc=self.extension(platform,"clHostMemAllocINTEL",C.WINFUNCTYPE(C.c_void_p,C.c_void_p,C.POINTER(C.c_ssize_t),C.c_size_t,C.c_uint,C.POINTER(C.c_int)))
            mem_free=self.extension(platform,"clMemFreeINTEL",C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p));setp=self.extension(platform,"clSetKernelArgMemPointerINTEL",C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_uint,C.c_void_p));geti=self.extension(platform,"clGetMemAllocInfoINTEL",C.WINFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_uint,C.c_size_t,C.c_void_p,C.POINTER(C.c_size_t)))
            pointers={}
            for name,size in BUFFER_TABLE:
                pointer=int(host_alloc(self.context,None,size,4096,C.byref(err)));check(err.value,"alloc");self.allocations.append((name,pointer,size,mem_free));pointers[name]=pointer;t=C.c_uint();b=C.c_void_p();z=C.c_size_t();check(geti(self.context,C.c_void_p(pointer),CL_MEM_ALLOC_TYPE_INTEL,C.sizeof(t),C.byref(t),None),"atype");check(geti(self.context,C.c_void_p(pointer),CL_MEM_ALLOC_BASE_PTR_INTEL,C.sizeof(b),C.byref(b),None),"abase");check(geti(self.context,C.c_void_p(pointer),CL_MEM_ALLOC_SIZE_INTEL,C.sizeof(z),C.byref(z),None),"asize")
                if t.value!=CL_MEM_TYPE_HOST_INTEL or b.value!=pointer or z.value!=size or pointer%4096:raise RuntimeError("alloc_attestation")
                self.ledger.append({"op":"host_usm_allocate","name":name,"bytes":size,"alignment":4096,"pointer":pointer,"type":t.value,"base":b.value,"queried_size":z.value})
            writes={"gate_record":records["gate"],"up_record":records["up"],"down_record":records["down"],"natural_input":input_bytes,"silu_lut":lut}
            for name,data in writes.items():C.memmove(pointers[name],data,len(data));self.ledger.append({"op":"cpu_direct_write","name":name,"bytes":len(data),"sha256":sha(data)})
            for name in ("gate","up","silu","activation","down"):C.memset(pointers[name],0xff,dict(BUFFER_TABLE)[name]);self.ledger.append({"op":"initialize","name":name,"value":"ff"})
            for name in ("gate_counters","up_counters","activation_counters","down_counters"):C.memset(pointers[name],0,dict(BUFFER_TABLE)[name]);self.ledger.append({"op":"initialize","name":name,"value":"00"})
            kernels=dict(self.kernels)
            for kernel,names in ARGUMENT_MAPS:
                for index,name in enumerate(names):check(setp(kernels[kernel],index,C.c_void_p(pointers[name])),"setarg");self.ledger.append({"op":"set_pointer_arg","kernel":kernel,"index":index,"name":name,"pointer":pointers[name]})
            for kernel,g,l in LAUNCHES:
                ga,la=(C.c_size_t*1)(g),(C.c_size_t*1)(l);check(self.lib.clEnqueueNDRangeKernel(self.queue,kernels[kernel],1,None,ga,la,0,None,None),"enqueue");self.ledger.append({"op":"enqueue","kernel":kernel,"global":g,"local":l,"event_requested":False})
            check(self.lib.clFinish(self.queue),"finish");self.ledger.append({"op":"finish","code":0})
            for name in ("gate","up","silu","activation","down","gate_counters","up_counters","activation_counters","down_counters"):
                data=C.string_at(pointers[name],dict(BUFFER_TABLE)[name]);outputs[name]=data;self.ledger.append({"op":"cpu_direct_read","name":name,"bytes":len(data),"sha256":sha(data),"after_finish":True})
        except Exception as exc:failure=exc
        else:failure=None
        finally:
            if self.lib:self.close()
        ev.update({"identity":identity,"outputs":{k:v.hex() for k,v in outputs.items()},"output_sha256":{k:sha(v) for k,v in outputs.items()},"cleanup_errors":self.cleanup_errors,"forbidden_calls":{"cl_mem":0,"enqueue_read":0,"enqueue_write":0,"enqueue_copy":0,"migrate":0,"prefetch":0}})
        if failure or self.cleanup_errors:raise ExecutionFailure(str(failure or "cleanup"),ev) from failure
        return ev
