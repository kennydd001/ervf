#!/usr/bin/env python3
"""No-device/no-payload static preflight for PH1 Intel execution R3."""
from __future__ import annotations
import ast,ctypes as C,hashlib,json,struct,tempfile,zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];S=ROOT/'scripts/streamq5_moe';R=ROOT/'reports/streamq5_moe'
RUN=S/'run_het_next_l0_ph1_intel_execution_r3.py';BACK=S/'het_next_l0_ph1_intel_execution_r3_backend.py';COMMON=S/'het_next_l0_ph1_intel_execution_r3_common.py';VER=S/'verify_het_next_l0_ph1_intel_execution_r3.py';R0B=S/'het_next_l0_ph1_intel_execution_r0_backend.py';R0R=S/'run_het_next_l0_ph1_intel_execution_r0.py';PR=R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R3_PREREGISTRATION_2026-08-14.md';LOCK=R/'het_next_l0_ph1_intel_execution_r3_lock.json';OUT=R/'het_next_l0_ph1_intel_execution_r3';RESULT=R/'het_next_l0_ph1_intel_execution_r3_static_preflight.json'
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
  cleanup=next(x for x in b.ledger if x.get('op')=='cleanup');post=[x for x in b.ledger if x.get('stage')=='post_cleanup']
  if len(rels)!=21 or [x['attempt_index'] for x in rels]!=list(range(21)) or sum(x['exception'] is not None for x in rels)!=1 or cleanup['release_attempts']!=21 or len(post)!=1:return False
 for fail in range(21):
  b=back.Backend();b.lib=Lib();b.context=303;b.queue=302;b.program=301;counter={'n':0}
  def code():i=counter['n'];counter['n']+=1;return -5 if i==fail else 0
  b.lib.clReleaseKernel=lambda _h:code();b.lib.clReleaseProgram=lambda _h:code();b.lib.clReleaseCommandQueue=lambda _h:code();b.lib.clReleaseContext=lambda _h:code();b.kernels=[(n,100+i) for i,n in enumerate(('gate_linear','up_linear','activation','down_linear'))];b.allocations=[(n,1000+i,z,lambda _c,_p:code()) for i,(n,z) in enumerate(back.BUFFER_TABLE)];b.close();rels=[x for x in b.ledger if x.get('op')=='release'];cleanup=next(x for x in b.ledger if x.get('op')=='cleanup')
  if len(rels)!=21 or sum(x['code']==-5 for x in rels)!=1 or cleanup['release_attempts']!=21:return False
 return True
def codec_oracle_fixtures(common,verify):
 zeros=bytes(512*2048*2);field_word=sum(15<<(5*i) for i in range(8));codes=field_word.to_bytes(5,'little')*(512*2048//8);sc=(b'\x80\x3f')*(512*2048//128);crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=common.HEADER.pack(b'SQ5M',1,0,50,0,5,512,2048,128,len(codes),len(sc),crc,bytes(28));expected_record=head+codes+sc+bytes(4032);spec0=('fixture',0,(512,2048),(0,0),hashlib.sha256(zeros).hexdigest(),hashlib.sha256(codes).hexdigest(),hashlib.sha256(sc).hexdigest(),hashlib.sha256(expected_record).hexdigest())
 if common.record(spec0,zeros)!=expected_record:return False
 src=np.zeros((512,2048),np.float32);src[0,:8]=np.asarray([-1,-.5,-.25,0,.25,.5,.75,1],np.float32);words=common.f2b(src).astype('<u2').tobytes();v=common.b2f(np.frombuffer(words,'<u2')).reshape(512,2048);bl=v.reshape(512,16,128);m=np.max(np.abs(bl),axis=-1,keepdims=True);scale=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(bl/scale,np.float32)),-15,15),0).astype(np.int16);fld=(q+15).astype(np.uint64).reshape(-1,8);packed=np.bitwise_or.reduce(fld<<(np.arange(8,dtype=np.uint64)*5),axis=1);nz_codes=np.stack([(packed>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes();nz_sc=common.f2b(scale.reshape(-1)).astype('<u2').tobytes();nz_crc=zlib.crc32(nz_sc,zlib.crc32(nz_codes))&0xffffffff;nz_head=common.HEADER.pack(b'SQ5M',1,0,50,0,5,512,2048,128,len(nz_codes),len(nz_sc),nz_crc,bytes(28));nz_rec=nz_head+nz_codes+nz_sc+bytes(4032);nz_spec=('fixture',0,(512,2048),(0,0),hashlib.sha256(words).hexdigest(),hashlib.sha256(nz_codes).hexdigest(),hashlib.sha256(nz_sc).hexdigest(),hashlib.sha256(nz_rec).hexdigest())
 if q[0,0,:8].tolist()!=[-15,-8,-4,0,4,8,11,15] or common.record(nz_spec,words)!=nz_rec:return False
 vr,decoded=verify.codec(words,nz_spec)
 if vr!=nz_rec or decoded.shape!=(512,2048) or decoded[0,:8].tolist()!=common.f2b(q.reshape(512,2048).astype(np.float32)*common.b2f(np.frombuffer(nz_sc,'<u2')).reshape(512,16).repeat(128,axis=1))[0,:8].tolist():return False
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
def verifier_mutations(verify,common,lock):
 import copy
 old=(verify.BUFF,verify.ARGS,verify.LAUNCH,verify.STAGE,verify.SPECS)
 try:
  mini_buff=(('gate_record',675840),('up_record',675840),('down_record',675840),('natural_input',128),('silu_lut',131072),('gate',128),('up',128),('silu',128),('activation',128),('down',2),('gate_counters',4),('up_counters',4),('activation_counters',4),('down_counters',4));verify.BUFF=mini_buff;verify.ARGS=(('gate_linear',('gate_record','natural_input','gate','gate_counters')),('up_linear',('up_record','natural_input','up','up_counters')),('activation',('gate','up','silu_lut','silu','activation','activation_counters')),('down_linear',('down_record','activation','down','down_counters')));verify.LAUNCH=[('gate_linear',4096,256),('up_linear',4096,256),('activation',512,256),('down_linear',16384,256)]
  inp=bytes(128);lut=bytes(131072);records={};specs=[]
  for ordinal,name in enumerate(('gate','up','down')):
   rows,cols=((512,2048) if name!='down' else (2048,512));word=(sum(15<<(5*i) for i in range(8))&~31)|16;codes=word.to_bytes(5,'little')*(rows*cols//8);sc=(b'\x80\x3f')*(rows*cols//128);crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=verify.HEADER.pack(b'SQ5M',1,0,50,ordinal,5,rows,cols,128,len(codes),len(sc),crc,bytes(28));rec=head+codes+sc+bytes(4032);spec=(name,ordinal,(rows,cols),(0,0),'unused',hashlib.sha256(codes).hexdigest(),hashlib.sha256(sc).hexdigest(),hashlib.sha256(rec).hexdigest());specs.append(spec);records[name]=rec
  specs=tuple(specs);verify.SPECS=specs;weights={'gate':np.zeros((64,64),np.uint16),'up':np.zeros((64,64),np.uint16),'down':np.zeros((1,64),np.uint16)};qg=verify.linear(weights['gate'],np.zeros(64,np.uint16));qu=verify.linear(weights['up'],np.zeros(64,np.uint16));qs=np.zeros(64,np.uint16);qa=np.zeros(64,np.uint16);qd=verify.linear(weights['down'],qa);out={'gate':qg.astype('<u2').tobytes(),'up':qu.astype('<u2').tobytes(),'silu':qs.astype('<u2').tobytes(),'activation':qa.astype('<u2').tobytes(),'down':qd.astype('<u2').tobytes(),'gate_counters':struct.pack('<I',1),'up_counters':struct.pack('<I',1),'activation_counters':struct.pack('<I',1),'down_counters':struct.pack('<I',1)};verify.STAGE={k:hashlib.sha256(out[k]).hexdigest() for k in ('gate','up','silu','activation','down')};pointers={n:4096*(i+1) for i,(n,_) in enumerate(mini_buff)};identity={'name':'Intel(R) Arc(TM) Pro 140T GPU (32GB)','vendor':'Intel(R) Corporation','driver':'32.0.101.8517','pci':'0000:00:02.0'};ledger=[{'op':'identity','identity':identity},{'op':'context_create'},{'op':'queue_create'},{'op':'program_create_binary'}]+[{'op':'kernel_create'} for _ in range(4)]+[{'op':'host_usm_allocate','name':n,'bytes':z,'pointer':pointers[n],'base':pointers[n],'queried_size':z,'type':0x4197} for n,z in mini_buff]
  for n,d in [('gate_record',records['gate']),('up_record',records['up']),('down_record',records['down']),('natural_input',inp),('silu_lut',lut)]:ledger.append({'op':'cpu_direct_write','name':n,'sha256':hashlib.sha256(d).hexdigest()})
  ledger +=[{'op':'initialize','name':n,'value':'ff'} for n in ('gate','up','silu','activation','down')]+[{'op':'initialize','name':n,'value':'00'} for n in ('gate_counters','up_counters','activation_counters','down_counters')]
  for kernel,names in verify.ARGS:
   for i,n in enumerate(names):ledger.append({'op':'set_pointer_arg','kernel':kernel,'index':i,'name':n,'pointer':pointers[n]})
  ledger +=[{'op':'resource_sample','stage':s,'available':2*2**30,'peak_wset':12*2**30} for s in ['backend_entry','pre_launch:0']]+[{'op':'enqueue','kernel':verify.LAUNCH[0][0],'global':4096,'local':256}]
  for i,(k,g,l) in enumerate(verify.LAUNCH[1:],1):ledger +=[{'op':'resource_sample','stage':'post_launch:'+str(i-1),'available':2*2**30,'peak_wset':12*2**30},{'op':'resource_sample','stage':'pre_launch:'+str(i),'available':2*2**30,'peak_wset':12*2**30},{'op':'enqueue','kernel':k,'global':g,'local':l}]
  ledger +=[{'op':'resource_sample','stage':'post_launch:3','available':2*2**30,'peak_wset':12*2**30},{'op':'resource_sample','stage':'pre_finish','available':2*2**30,'peak_wset':12*2**30},{'op':'finish'},{'op':'resource_sample','stage':'post_finish','available':2*2**30,'peak_wset':12*2**30}]
  for n in ('gate','up','silu','activation','down','gate_counters','up_counters','activation_counters','down_counters'):ledger.append({'op':'cpu_direct_read','name':n,'bytes':dict(mini_buff)[n],'sha256':hashlib.sha256(out[n]).hexdigest(),'after_finish':True})
  rel=['usm:'+x[0] for x in reversed(mini_buff)]+['kernel:down_linear','kernel:activation','kernel:up_linear','kernel:gate_linear','program','queue','context'];ledger +=[{'op':'release','name':n,'attempt_index':i,'attempted':True,'code':0,'exception':None,'owned_after':False} for i,n in enumerate(rel)]+[{'op':'cleanup','cleanup_complete':True,'release_attempts':21,'live_owned_resources':0,'live_resource_names':[]},{'op':'resource_sample','stage':'post_cleanup','available':2*2**30,'peak_wset':12*2**30}]
  observed={k:fs(v) for k,v in verify.PROVENANCE.items()};prepared=(specs,inp,lut,records,weights,lock,observed);oldspec=common.SPECS
  try:common.SPECS=specs;controls=common.controls(records,inp,lut)
  finally:common.SPECS=oldspec
  ev={'outputs':{k:v.hex() for k,v in out.items()},'ledger':ledger,'identity':identity,'extension_counts':{'clHostMemAllocINTEL':14,'clMemFreeINTEL':14,'clSetKernelArgMemPointerINTEL':18,'clGetMemAllocInfoINTEL':42},'forbidden_calls':{k:0 for k in ('clCreateBuffer','clEnqueueReadBuffer','clEnqueueWriteBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects','clEnqueueMemAdviseINTEL')}};gates={'fixture':True};base={'positive':True,'status':'intel_execution_positive','claim':'one real expert/input Intel correctness component only','evidence':ev,'controls':controls,'authorization':{'lock_sha256':fs(verify.LOCK),'observed':observed},'resource':{'sampling_claim':'retained phase samples, not continuous peak monitor','start_available':16*2**30,'post_payload_available':2*2**30,'final_available':2*2**30,'post_serialize_available':2*2**30,'peak_retained_wset':12*2**30},'gates':gates,'stage_hashes':verify.STAGE}
  if not all(verify.verify_dict(base,prepared).values()):return False
  mutations=[lambda x:x['evidence']['identity'].__setitem__('pci','wrong'),lambda x:x['controls'].pop(),lambda x:x['evidence']['outputs'].__setitem__('gate','00'),lambda x:x['evidence']['ledger'][8].__setitem__('pointer',0),lambda x:x['evidence']['ledger'][-2].__setitem__('cleanup_complete',False),lambda x:x['authorization']['observed'].__setitem__('runner_sha256','0'*64),lambda x:x['resource'].__setitem__('peak_retained_wset',12*2**30+1),lambda x:x['evidence']['forbidden_calls'].__setitem__('clCreateBuffer',1),lambda x:x['stage_hashes'].__setitem__('gate','0'*64)]
  for mutate in mutations:
   m=copy.deepcopy(base);mutate(m)
   if all(verify.verify_dict(m,prepared).values()):return False
  return True
 finally:verify.BUFF,verify.ARGS,verify.LAUNCH,verify.STAGE,verify.SPECS=old
def transaction_sim(run):
 old=(run.R,run.OUT,run.FAILED,run.QUAR,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle)
 try:
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);run.R=root;run.OUT=root/'out';run.FAILED=root/'failed';run.QUAR=root/'quar';run.configure();run.OUT.mkdir();result={'kind':'ph1_intel_execution_r3','positive':True};run.base.write(run.OUT/'result.json',run.base.canon(result));row={'name':'result.json','bytes':(run.OUT/'result.json').stat().st_size,'sha256':run.fs(run.OUT/'result.json')};run.base.write(run.OUT/'manifest.json',run.base.canon({'kind':'ph1_intel_execution_r3_manifest','files':[row]}));run.base.write(run.OUT/'commit.json',run.base.canon({'kind':'ph1_intel_execution_r3_commit','manifest_sha256':run.fs(run.OUT/'manifest.json'),'result_sha256':run.fs(run.OUT/'result.json')}));
   if run.base.recover()['result']!=result:return False
   run.OUT.rename(root/'held');stale=root/(run.OUT.name+'.x.inprogress');stale.mkdir();
   try:run.base.recover();return False
   except RuntimeError:
    if stale.exists() or not any(run.QUAR.iterdir()):return False
   run.base.archive(run.FAILED,'attempt',{'kind':'ph1_intel_execution_r3_failure','device_opened':False})
   return any(run.FAILED.iterdir())
 finally:run.R,run.OUT,run.FAILED,run.QUAR,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle=old
def lifecycle_sim(run):
 old=(run.R,run.OUT,run.FAILED,run.QUAR,run.authorize,run.sys.argv,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle)
 try:
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);run.R=root;run.OUT=root/'out';run.FAILED=root/'failed';run.QUAR=root/'quar';run.configure();run.sys.argv=['runner','--ack',run.ACK];run.authorize=lambda:(_ for _ in ()).throw(RuntimeError('auth_injected'))
   if run.main()!=3:return False
   f=list(run.FAILED.glob('*/failure.json'))
   if len(f)!=1 or json.loads(f[0].read_text()).get('stage')!='authorization_start_or_payload' or sum(x.stat().st_size for x in run.FAILED.rglob('*') if x.is_file())>16*2**20:return False
   for d in list(run.FAILED.iterdir()):
    for x in d.iterdir():x.unlink()
    d.rmdir()
   run.FAILED.rmdir();run.authorize=lambda:{'lock_sha256':'x','observed':{}};run.OUT.mkdir();result={'kind':'ph1_intel_execution_r3','positive':False};run.base.write(run.OUT/'result.json',run.base.canon(result));row={'name':'result.json','bytes':(run.OUT/'result.json').stat().st_size,'sha256':run.fs(run.OUT/'result.json')};run.base.write(run.OUT/'manifest.json',run.base.canon({'kind':'ph1_intel_execution_r3_manifest','files':[row]}));run.base.write(run.OUT/'commit.json',run.base.canon({'kind':'ph1_intel_execution_r3_commit','manifest_sha256':run.fs(run.OUT/'manifest.json'),'result_sha256':run.fs(run.OUT/'result.json')}))
   return run.main()==3 and not run.FAILED.exists()
 finally:run.R,run.OUT,run.FAILED,run.QUAR,run.authorize,run.sys.argv,run.base.REPORTS,run.base.OUT,run.base.FAILED,run.base.QUAR,run.base.verify_bundle=old
def main():
 import sys;sys.path.insert(0,str(S));import het_next_l0_ph1_intel_execution_r3_common as common;import het_next_l0_ph1_intel_execution_r3_backend as back;import run_het_next_l0_ph1_intel_execution_r3 as run;import verify_het_next_l0_ph1_intel_execution_r3 as verify
 lock=json.loads(LOCK.read_text());observed={k:fs(v) for k,v in verify.PROVENANCE.items()};trees={p:ast.parse(p.read_text()) for p in (RUN,BACK,COMMON,VER,Path(__file__),R0B,R0R)};imports={n.module for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{a.name for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names};forbidden={'clCreateBuffer','clEnqueueReadBuffer','clEnqueueWriteBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects','clEnqueueMemAdviseINTEL'};calls={n.func.attr for t in trees.values() for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
 checks={'hash_bindings':all(lock.get(k)==v for k,v in observed.items()),'closed':lock.get('execution_open') is False and lock.get('audit_token')=='PENDING' and not OUT.exists() and not RESULT.exists(),'cardinalities':len(back.BUFFER_TABLE)==14 and sum(z for _,z in back.BUFFER_TABLE)==2185216 and sum(len(x) for _,x in back.ARGUMENT_MAPS)==18 and len(back.LAUNCHES)==4,'extension_abi':set(back.EXTENSION_ABI)=={'clHostMemAllocINTEL','clMemFreeINTEL','clSetKernelArgMemPointerINTEL','clGetMemAllocInfoINTEL'} and all(isinstance(a,tuple) and len(a)==2 and isinstance(a[1],tuple) for a in back.EXTENSION_ABI.values()),'no_device_static':not ({'pyopencl','cupy','torch','safetensors','transformers'}&imports) and all('WinDLL("OpenCL.dll")' not in p.read_text() for p in (Path(__file__),COMMON,VER)),'forbidden_call_surface':not (forbidden&calls),'synthetic_22_controls':synthetic_controls(common),'codec_fma_width8':codec_oracle_fixtures(common,verify),'package_commit_mutations':package_mutations(run),'cleanup_faults_21':cleanup_faults(back),'transaction_failure':transaction_sim(run),'production_lifecycle':lifecycle_sim(run),'actual_verifier_mutations':verifier_mutations(verify,common,lock)};out={'kind':'ph1_intel_execution_r3_static_preflight','checks':checks,'pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'no_payload_compiler_device':True}
 if RESULT.exists():raise FileExistsError(RESULT)
 RESULT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
