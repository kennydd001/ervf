#!/usr/bin/env python3
"""No-device/no-payload static preflight for PH1 Intel execution R2."""
from __future__ import annotations
import ast,ctypes as C,hashlib,json,struct,tempfile,zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe'
RUN=S/'run_het_next_l0_ph1_intel_execution_r2.py';BACK=S/'het_next_l0_ph1_intel_execution_r2_backend.py';COMMON=S/'het_next_l0_ph1_intel_execution_r2_common.py';VER=S/'verify_het_next_l0_ph1_intel_execution_r2.py';R0B=S/'het_next_l0_ph1_intel_execution_r0_backend.py';R0R=S/'run_het_next_l0_ph1_intel_execution_r0.py';PR=R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R2_PREREGISTRATION_2026-08-14.md';LOCK=R/'het_next_l0_ph1_intel_execution_r2_lock.json';OUT=R/'het_next_l0_ph1_intel_execution_r2';RESULT=R/'het_next_l0_ph1_intel_execution_r2_static_preflight.json'
def fs(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def synthetic_controls(common):
 old=(common.SPECS,common.INPUT,common.LUT_SHA)
 try:
  records={};specs=[]
  for ordinal,name in enumerate(('gate','up','down')):
   rows,cols=((512,2048) if name!='down' else (2048,512));word=sum(15<<(5*i) for i in range(8));word=(word&~31)|16;code5=word.to_bytes(5,'little');codes=bytearray(code5*(rows*cols//8));sc=(b'\x80\x3f')*(rows*cols//128);crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=common.HEADER.pack(b'SQ5M',1,0,50,ordinal,5,rows,cols,128,len(codes),len(sc),crc,bytes(28));rec=head+bytes(codes)+sc+bytes(4032);spec=(name,ordinal,(rows,cols),(0,0),'unused',hashlib.sha256(codes).hexdigest(),hashlib.sha256(sc).hexdigest(),hashlib.sha256(rec).hexdigest());specs.append(spec);records[name]=rec
  inp=bytes(4096);lut=bytes(131072);common.SPECS=tuple(specs);common.INPUT=(0,4096,hashlib.sha256(inp).hexdigest());common.LUT_SHA=hashlib.sha256(lut).hexdigest();rows=common.controls(records,inp,lut)
  return len(rows)==22 and all(x['pass'] and all(v==0 for v in x['predevice_counts'].values()) for x in rows) and [(x['record'],x['control'],x['observed']) for x in rows]==[(n,c,e) for n in ('gate','up','down') for c,e in (('truncation','size'),('wrong_projection','identity'),('stale_crc','crc'),('code_mutation','canonical_digest'),('scale_mutation','canonical_digest'),('field31','field31'),('wrong_input','input_digest'))]+[('global','wrong_lut_digest','lut_digest')]
 finally:common.SPECS,common.INPUT,common.LUT_SHA=old
def cleanup_faults(back):
 class Lib:
  pass
 for fail in range(21):
  b=back.Backend();b.lib=Lib();b.context=303;b.queue=302;b.program=301;counter={'n':0}
  def call():
   i=counter['n'];counter['n']+=1
   if i==fail:raise OSError('injected')
   return 0
  b.lib.clReleaseKernel=lambda _h:call();b.lib.clReleaseProgram=lambda _h:call();b.lib.clReleaseCommandQueue=lambda _h:call();b.lib.clReleaseContext=lambda _h:call();b.kernels=[(n,100+i) for i,n in enumerate(('gate_linear','up_linear','activation','down_linear'))];b.allocations=[(n,1000+i,z,lambda _c,_p:call()) for i,(n,z) in enumerate(back.BUFFER_TABLE)];b.close();rels=[x for x in b.ledger if x.get('op')=='release']
  if len(rels)!=21 or [x['attempt_index'] for x in rels]!=list(range(21)) or sum(x['exception'] is not None for x in rels)!=1 or b.ledger[-1]['release_attempts']!=21:return False
 return True
def codec_oracle_fixtures(common,verify):
 zeros=bytes(512*2048*2);field_word=sum(15<<(5*i) for i in range(8));codes=field_word.to_bytes(5,'little')*(512*2048//8);sc=(b'\x80\x3f')*(512*2048//128);crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=common.HEADER.pack(b'SQ5M',1,0,50,0,5,512,2048,128,len(codes),len(sc),crc,bytes(28));expected_record=head+codes+sc+bytes(4032);spec0=('fixture',0,(512,2048),(0,0),hashlib.sha256(zeros).hexdigest(),hashlib.sha256(codes).hexdigest(),hashlib.sha256(sc).hexdigest(),hashlib.sha256(expected_record).hexdigest())
 if common.record(spec0,zeros)!=expected_record:return False
 src=np.zeros((1,128),np.float32);src[0,:8]=np.asarray([-1,-.5,-.25,0,.25,.5,.75,1],np.float32);words=common.f2b(src).astype('<u2').tobytes()
 v=common.b2f(np.frombuffer(words,'<u2')).reshape(1,128);m=np.max(np.abs(v));scale=np.float32(m/15);q=np.clip(np.rint(np.asarray(v/scale,np.float32)),-15,15).astype(np.int16)
 if q[0,:8].tolist()!=[-15,-8,-4,0,4,8,11,15]:return False
 vectors=[(0x3f800000,0x3f800000,0,0x3f800000),(0x3f000000,0x40000000,0,0x3f800000),(0xbf800000,0x3f000000,0x3f800000,0x3f000000)]
 if any(verify.fma(a,b,c)!=e for a,b,c,e in vectors):return False
 w=np.zeros((1,64),np.uint16);x=np.zeros(64,np.uint16);w[0,:8]=np.asarray([0x3f80]*8,np.uint16);x[:8]=np.asarray([0x3f80]*8,np.uint16)
 return verify.linear(w,x).tolist()==[0x4100] and verify.mul(0x3f80,0x4000)==0x4000
def package_mutations(run):
 with tempfile.TemporaryDirectory() as td:
  d=Path(td);files={'intel_build.log':b'\n','intel_program.bin':b'x','intel_source.cl':b'y','result.json':b'{}\n'}
  for n,b in files.items():(d/n).write_bytes(b)
  rows=[{'name':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in sorted(files.items())];(d/'manifest.json').write_text(json.dumps({'kind':'het_next_l0_ph1_intel_compile_r2a_manifest','files':rows},sort_keys=True,separators=(',',':'))+'\n');(d/'commit.json').write_text(json.dumps({'kind':'het_next_l0_ph1_intel_compile_r2a_commit','manifest_sha256':fs(d/'manifest.json'),'result_sha256':fs(d/'result.json')},sort_keys=True,separators=(',',':'))+'\n')
  if not run.package_exact(d):return False
  good=(d/'commit.json').read_bytes();(d/'commit.json').write_text('{}\n')
  if run.package_exact(d):return False
  (d/'commit.json').write_bytes(good);m=json.loads((d/'manifest.json').read_text());m['kind']='wrong';(d/'manifest.json').write_text(json.dumps(m)+'\n')
  return not run.package_exact(d)
def verifier_mutations(verify):
 base=verify.mutation_fixture_result();checks=verify.verify_dict(base)
 if not all(checks.values()):return False
 paths=[('kind','wrong'),('identity',dict(base['identity'],pci='wrong')),('controls',base['controls'][:-1]),('outputs',dict(base['outputs'],gate='00')),('counters',dict(base['counters'],gate=0)),('ledger',dict(base['ledger'],order='wrong')),('ledger',dict(base['ledger'],cleanup_complete=False)),('ledger',dict(base['ledger'],forbidden_rows=1)),('provenance',dict(base['provenance'],authorized=False)),('resource',dict(base['resource'],peak=12*2**30+1))]
 for key,value in paths:
  mutant=dict(base);mutant[key]=value
  if all(verify.verify_dict(mutant).values()):return False
 return True
def transaction_sim(run):
 old=(run.R,run.OUT,run.FAILED,run.QUAR,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle)
 try:
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);run.R=root;run.OUT=root/'out';run.FAILED=root/'failed';run.QUAR=root/'quar';run.configure();run.OUT.mkdir();result={'kind':'ph1_intel_execution_r2','positive':True};run.base.write(run.OUT/'result.json',run.base.canon(result));row={'name':'result.json','bytes':(run.OUT/'result.json').stat().st_size,'sha256':run.fs(run.OUT/'result.json')};run.base.write(run.OUT/'manifest.json',run.base.canon({'kind':'ph1_intel_execution_r2_manifest','files':[row]}));run.base.write(run.OUT/'commit.json',run.base.canon({'kind':'ph1_intel_execution_r2_commit','manifest_sha256':run.fs(run.OUT/'manifest.json'),'result_sha256':run.fs(run.OUT/'result.json')}));
   if run.base.recover()['result']!=result:return False
   run.OUT.rename(root/'held');stale=root/(run.OUT.name+'.x.inprogress');stale.mkdir();
   try:run.base.recover();return False
   except RuntimeError:
    if stale.exists() or not any(run.QUAR.iterdir()):return False
   run.base.archive(run.FAILED,'attempt',{'kind':'ph1_intel_execution_r2_failure','device_opened':False})
   return any(run.FAILED.iterdir())
 finally:run.R,run.OUT,run.FAILED,run.QUAR,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle=old
def main():
 import sys;sys.path.insert(0,str(S));import het_next_l0_ph1_intel_execution_r2_common as common;import het_next_l0_ph1_intel_execution_r2_backend as back;import run_het_next_l0_ph1_intel_execution_r2 as run;import verify_het_next_l0_ph1_intel_execution_r2 as verify
 lock=json.loads(LOCK.read_text());observed={'runner_sha256':fs(RUN),'backend_sha256':fs(BACK),'common_sha256':fs(COMMON),'verifier_sha256':fs(VER),'preflight_sha256':fs(Path(__file__)),'prereg_sha256':fs(PR),'r0_backend_sha256':fs(R0B),'r0_runner_sha256':fs(R0R)};trees={p:ast.parse(p.read_text()) for p in (RUN,BACK,COMMON,VER,Path(__file__),R0B,R0R)};imports={n.module for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{a.name for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names};forbidden={'clCreateBuffer','clEnqueueReadBuffer','clEnqueueWriteBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects','clEnqueueMemAdviseINTEL'};calls={n.func.attr for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
 checks={'hash_bindings':all(lock.get(k)==v for k,v in observed.items()),'closed':lock.get('execution_open') is False and lock.get('audit_token')=='PENDING' and not OUT.exists() and not RESULT.exists(),'cardinalities':len(back.BUFFER_TABLE)==14 and sum(z for _,z in back.BUFFER_TABLE)==2185216 and sum(len(x) for _,x in back.ARGUMENT_MAPS)==18 and len(back.LAUNCHES)==4,'extension_abi':set(back.EXTENSION_ABI)=={'clHostMemAllocINTEL','clMemFreeINTEL','clSetKernelArgMemPointerINTEL','clGetMemAllocInfoINTEL'} and all(isinstance(a,tuple) and len(a)==2 and isinstance(a[1],tuple) for a in back.EXTENSION_ABI.values()),'no_device_static':not ({'pyopencl','cupy','torch','safetensors','transformers'}&imports) and all('WinDLL("OpenCL.dll")' not in p.read_text() for p in (Path(__file__),COMMON,VER)),'forbidden_call_surface':not (forbidden&calls),'synthetic_22_controls':synthetic_controls(common),'codec_fma_width8':codec_oracle_fixtures(common,verify),'package_commit_mutations':package_mutations(run),'cleanup_faults_21':cleanup_faults(back),'transaction_failure':transaction_sim(run),'actual_verifier_mutations':verifier_mutations(verify)};out={'kind':'ph1_intel_execution_r2_static_preflight','checks':checks,'pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'no_payload_compiler_device':True}
 if RESULT.exists():raise FileExistsError(RESULT)
 RESULT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
