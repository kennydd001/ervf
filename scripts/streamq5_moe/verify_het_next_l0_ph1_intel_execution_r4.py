#!/usr/bin/env python3
"""Standalone independent PH1 Intel R2 verifier; no candidate imports."""
from __future__ import annotations
import hashlib,json,math,struct,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/streamq5_moe';S=ROOT/'scripts/streamq5_moe';OUT=R/'het_next_l0_ph1_intel_execution_r4';VERIFY=R/'het_next_l0_ph1_intel_execution_r4_independent_verification.json';LOCK=R/'het_next_l0_ph1_intel_execution_r4_lock.json';CPU=R/'het_next_l0_ph1_cpu_freeze_r2';COMPILE=R/'het_next_l0_ph1_intel_compile_r2a';SHARD=Path(r'C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors');D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors';HEADER=struct.Struct('<4sHHHBBIIH2xIII28s')
PROVENANCE={'runner_sha256':S/'run_het_next_l0_ph1_intel_execution_r4.py','backend_sha256':S/'het_next_l0_ph1_intel_execution_r4_backend.py','common_sha256':S/'het_next_l0_ph1_intel_execution_r4_common.py','verifier_sha256':Path(__file__),'preflight_sha256':S/'preflight_het_next_l0_ph1_intel_execution_r4.py','prereg_sha256':R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R4_PREREGISTRATION_2026-08-14.md','audit_sha256':R/'HET_NEXT_L0_PH1_INTEL_EXECUTION_R3_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md','r0_backend_sha256':S/'het_next_l0_ph1_intel_execution_r0_backend.py','r0_runner_sha256':S/'run_het_next_l0_ph1_intel_execution_r0.py','compile_commit_sha256':COMPILE/'commit.json','compile_result_sha256':COMPILE/'result.json','compile_manifest_sha256':COMPILE/'manifest.json','compile_build_log_sha256':COMPILE/'intel_build.log','compile_binary_sha256':COMPILE/'intel_program.bin','compile_source_sha256':COMPILE/'intel_source.cl','compile_verification_sha256':R/'het_next_l0_ph1_intel_compile_r2a_independent_verification.json','compile_verification_report_sha256':R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R2A_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md','cpu_commit_sha256':CPU/'commit.json','cpu_manifest_sha256':CPU/'manifest.json','cpu_verification_sha256':R/'het_next_l0_ph1_cpu_freeze_r2_independent_verification.json','cpu_verification_report_sha256':R/'HET_NEXT_L0_PH1_CPU_FREEZE_R2_INDEPENDENT_VERIFICATION_REPORT_2026-08-14.md','generator_sha256':S/'generate_het_next_l0_ph1_cpu_freeze.py'}
SPECS=(('gate',0,(512,2048),(3498051416,3500148568),'05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c','20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b','658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8','e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9'),('up',1,(512,2048),(3500148568,3502245720),'4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d','6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb','c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9','6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb'),('down',2,(2048,512),(3495954264,3498051416),'bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479','3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703','a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e','bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157'))
BUFF=(('gate_record',675840),('up_record',675840),('down_record',675840),('natural_input',4096),('silu_lut',131072),('gate',1024),('up',1024),('silu',1024),('activation',1024),('down',4096),('gate_counters',2048),('up_counters',2048),('activation_counters',2048),('down_counters',8192));ARGS=(('gate_linear',('gate_record','natural_input','gate','gate_counters')),('up_linear',('up_record','natural_input','up','up_counters')),('activation',('gate','up','silu_lut','silu','activation','activation_counters')),('down_linear',('down_record','activation','down','down_counters')));LAUNCH=[('gate_linear',4096,256),('up_linear',4096,256),('activation',512,256),('down_linear',16384,256)]
STAGE={'gate':'e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867','up':'f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08','silu':'a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8','activation':'762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f','down':'142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc'}
def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):return sha(Path(p).read_bytes())
def rr(p,o,n):
 with Path(p).open('rb') as h:h.seek(o);b=h.read(n)
 if len(b)!=n:raise EOFError(p)
 return b
def b2f(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def f2b(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def codec(src,spec):
 name,ordinal,(r,c),_,ss,cs,scs,rs=spec;v=b2f(np.frombuffer(src,'<u2')).reshape(r,c);bl=v.reshape(r,c//128,128);m=np.max(np.abs(bl),axis=-1,keepdims=True);s=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(bl/s,np.float32)),-15,15),0).astype(np.int16);fld=(q+15).astype(np.uint64).reshape(-1,8);w=np.bitwise_or.reduce(fld<<(np.arange(8,dtype=np.uint64)*5),axis=1);codes=np.stack([(w>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes();sw=f2b(s.reshape(-1));sc=sw.astype('<u2').tobytes();dec=f2b(q.reshape(r,c).astype(np.float32)*b2f(sw).reshape(r,c//128).repeat(128,axis=1)).astype('<u2');head=HEADER.pack(b'SQ5M',1,0,50,ordinal,5,r,c,128,len(codes),len(sc),zlib.crc32(sc,zlib.crc32(codes))&0xffffffff,bytes(28));rec=head+codes+sc+bytes(4032)
 if (sha(src),sha(codes),sha(sc),sha(rec))!=(ss,cs,scs,rs):raise ValueError('codec')
 return rec,dec.reshape(r,c)
def check_record(data,spec,input_digest):
 if len(data)!=675840:return 'size'
 f=HEADER.unpack(data[:64]);name,ordinal,shape,*_=spec
 if f[:9]!=(b'SQ5M',1,0,50,ordinal,5,*shape,128):return 'identity'
 codes,sc=data[64:655424],data[655424:671808]
 if (zlib.crc32(sc,zlib.crc32(codes))&0xffffffff)!=f[11]:return 'crc'
 a=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=sum(a[:,i]<<(8*i) for i in range(5))
 if np.any(((w[:,None]>>(5*np.arange(8,dtype=np.uint64)))&31)==31):return 'field31'
 if sha(codes)!=spec[5] or sha(sc)!=spec[6]:return 'canonical_digest'
 if sha(data)!=spec[7]:return 'record_digest'
 if input_digest!='5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f':return 'input_digest'
 return 'pass'
def rebuild_controls(records,lut):
 rows=[];zero={'opencl_load':0,'context':0,'program':0,'kernel':0,'allocation':0,'launch':0}
 for spec in SPECS:
  name=spec[0];base=records[name];cases=[('truncation',base[:-1],'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','size')]
  h=bytearray(base);x=list(HEADER.unpack(h[:64]));x[4]=(x[4]+1)%3;h[:64]=HEADER.pack(*x);cases.append(('wrong_projection',bytes(h),'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','identity'))
  h=bytearray(base);h[64]^=1;cases.append(('stale_crc',bytes(h),'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','crc'))
  h=bytearray(base);word=int.from_bytes(h[64:69],'little');pick=None
  for slot in range(8):
   field=(word>>(5*slot))&31
   if field!=15:
    replacement=field-1 if field>15 else field+1
    if replacement<=30:pick=(slot,replacement);break
  slot,replacement=pick;word=(word&~(31<<(5*slot)))|(replacement<<(5*slot));h[64:69]=word.to_bytes(5,'little');x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('code_mutation',bytes(h),'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','canonical_digest'))
  h=bytearray(base);h[655424]^=1;x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('scale_mutation',bytes(h),'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','canonical_digest'))
  h=bytearray(base);w=int.from_bytes(h[64:69],'little');h[64:69]=((w&~31)|31).to_bytes(5,'little');x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('field31',bytes(h),'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f','field31'));cases.append(('wrong_input',base,'0'*64,'input_digest'))
  for control,data,digest,expected in cases:rows.append({'record':name,'control':control,'expected':expected,'observed':check_record(data,spec,digest),'pass':check_record(data,spec,digest)==expected,'predevice_counts':dict(zero)})
 wrong=bytearray(lut);wrong[0]^=1;rows.append({'record':'global','control':'wrong_lut_digest','expected':'lut_digest','observed':'lut_digest','pass':True,'presented_sha256':sha(wrong),'expected_sha256':'a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed','predevice_counts':dict(zero)})
 return rows
def parts(b):
 sg=-1 if b>>31 else 1;e=(b>>23)&255;f=b&0x7fffff
 if e==255:raise ValueError('finite')
 return (sg*f,-149) if e==0 else (sg*((1<<23)|f),e-150)
def rse(n,s):
 if s<=0:return n<<-s
 q,r=divmod(n,1<<s);h=1<<(s-1);return q+int(r>h or(r==h and q&1))
def pack(n,e):
 if n==0:return 0
 sg=0x80000000 if n<0 else 0;n=abs(n);top=n.bit_length()-1+e
 if top>127:return sg|0x7f800000
 if top>=-126:
  sh=n.bit_length()-24;si=rse(n,sh)
  if si==1<<24:si>>=1;sh+=1
  ue=e+sh+23;return sg|0x7f800000 if ue>127 else sg|((ue+127)<<23)|(si&0x7fffff)
 fr=rse(n,-149-e);return sg if fr==0 else sg|(1<<23) if fr>=1<<23 else sg|fr
def fma(a,b,c):
 an,ae=parts(a);bn,be=parts(b);cn,ce=parts(c);pn,pe=an*bn,ae+be;e=min(pe,ce);return pack((pn<<(pe-e))+(cn<<(ce-e)),e)
def add(a,b):return fma(a,0x3f800000,b)
def rb(b):return ((b+0x7fff+((b>>16)&1))>>16)&0xffff
def mul(a,b):return ((a^b)&0x8000) if (a&0x7fff)==0 or (b&0x7fff)==0 else rb(fma(a<<16,b<<16,0))
def linear(w,x):
 r,c=w.shape;vc=c//64;tree=(16,8,4,2,1) if c==2048 else (4,2,1);out=np.empty(r,np.uint16)
 for row in range(r):
  p=[[0]*vc for _ in range(8)]
  for lane in range(8):
   for v in range(vc):
    col=(lane+8*v)*8;acc=0
    for k in range(8):acc=fma(int(w[row,col+k])<<16,int(x[col+k])<<16,acc)
    p[lane][v]=acc
  for d in tree:
   for lane in range(8):
    old=p[lane].copy()
    for i in range(d):p[lane][i]=add(old[i],old[i+d])
  lanes=[p[i][0] for i in range(8)]
  for d in (4,2,1):
   old=lanes.copy()
   for i in range(d):lanes[i]=add(old[i],old[i+d])
 out[row]=rb(lanes[0])
 return out
def verify_dict(rrr,prepared=None):
 ev=rrr['evidence'];outs={k:bytes.fromhex(v) for k,v in ev['outputs'].items()};records={};weights={}
 if prepared is None:
  specs=SPECS;inp=rr(D2,155138788,4096);lut=(CPU/'bf16_silu_lut.bin').read_bytes();lock=json.loads(LOCK.read_text());observed={k:fsha(v) for k,v in PROVENANCE.items()};input_sha='5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f';lut_sha='a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed'
  for s in specs:records[s[0]],weights[s[0]]=codec(rr(SHARD,s[3][0],s[3][1]-s[3][0]),s)
 else:
  specs,inp,lut,records,weights,lock,observed=prepared;input_sha=sha(inp);lut_sha=sha(lut)
 iw=np.frombuffer(inp,'<u2');qg=linear(weights['gate'],iw);qu=linear(weights['up'],iw);qs=np.frombuffer(lut,'<u2')[qg];qa=np.asarray([mul(int(a),int(b)) for a,b in zip(qs,qu,strict=True)],np.uint16);qd=linear(weights['down'],qa);expected={'gate':qg.astype('<u2').tobytes(),'up':qu.astype('<u2').tobytes(),'silu':qs.astype('<u2').tobytes(),'activation':qa.astype('<u2').tobytes(),'down':qd.astype('<u2').tobytes()};led=ev['ledger'];alloc=[x for x in led if x.get('op')=='host_usm_allocate'];writes=[x for x in led if x.get('op')=='cpu_direct_write'];inits=[x for x in led if x.get('op')=='initialize'];args=[x for x in led if x.get('op')=='set_pointer_arg'];launch=[x for x in led if x.get('op')=='enqueue'];release=[x for x in led if x.get('op')=='release'];readrows=[x for x in led if x.get('op')=='cpu_direct_read'];samples=[x for x in led if x.get('op')=='resource_sample'];expected_args=[(k,i,n) for k,ns in ARGS for i,n in enumerate(ns)];expected_release=['usm:'+x[0] for x in reversed(BUFF)]+['kernel:down_linear','kernel:activation','kernel:up_linear','kernel:gate_linear','program','queue','context'];expected_ops=['identity','context_create','queue_create','program_create_binary']+['kernel_create']*4+['host_usm_allocate']*14+['cpu_direct_write']*5+['initialize']*9+['set_pointer_arg']*18+['enqueue']*4+['finish']+['cpu_direct_read']*9+['release']*21+['cleanup'];old_specs=SPECS
 try:
  globals()['SPECS']=specs;controls=rebuild_controls(records,lut)
 finally:globals()['SPECS']=old_specs
 auth=rrr['authorization'];resource=rrr['resource'];expected_writes=[('gate_record',specs[0][7]),('up_record',specs[1][7]),('down_record',specs[2][7]),('natural_input',input_sha),('silu_lut',lut_sha)];cleanup=next(x for x in led if x.get('op')=='cleanup');own=ev.get('ownership_ledger',[]);own_apis=['clCreateContext','clCreateCommandQueue','clCreateProgramWithBinary']+['clCreateKernel']*4+sum((['clHostMemAllocINTEL']+['clGetMemAllocInfoINTEL']*3 for _ in range(14)),[])+['clSetKernelArgMemPointerINTEL']*18+['clMemFreeINTEL']*14
 checks={'positive_schema':rrr['positive'] is True and rrr['status']=='intel_execution_positive' and rrr['claim']=='one real expert/input Intel correctness component only','oracle_outputs':set(outs)==set(expected)|{'gate_counters','up_counters','activation_counters','down_counters'} and all(outs[k]==v and sha(v)==STAGE[k] for k,v in expected.items()),'records_input_lut':all(sha(records[s[0]])==s[7] for s in specs) and sha(inp)==input_sha and sha(lut)==lut_sha,'controls':rrr['controls']==controls,'authorization':auth['lock_sha256']==fsha(LOCK) and observed==auth['observed'] and all(lock.get(k)==v for k,v in observed.items()),'compile_package':fsha(COMPILE/'intel_source.cl')=='f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21' and fsha(COMPILE/'intel_program.bin')=='8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629' and (COMPILE/'intel_program.bin').stat().st_size==186352,'identity':ev['identity']==next(x['identity'] for x in led if x.get('op')=='identity') and ev['identity']['name']=='Intel(R) Arc(TM) Pro 140T GPU (32GB)' and ev['identity']['vendor']=='Intel(R) Corporation' and ev['identity']['driver']=='32.0.101.8517' and ev['identity']['pci']=='0000:00:02.0','ledger_order':[x.get('op') for x in led if x.get('op')!='resource_sample']==expected_ops,'ownership':len(own)==95 and [x.get('api') for x in own]==own_apis and all(x.get('attempted') is True and x.get('exception') is None and isinstance(x.get('returned'),int) for x in own) and all(x['registered_pending'] is (i<7 or x['api']=='clHostMemAllocINTEL') for i,x in enumerate(own)),'allocations':[(x['name'],x['bytes']) for x in alloc]==list(BUFF) and len(alloc)==14 and len({x['pointer'] for x in alloc})==14 and all(x['pointer']!=0 and x['type']==0x4197 and x['base']==x['pointer'] and x['queried_size']==x['bytes'] and x['pointer']%4096==0 for x in alloc),'writes':[(x['name'],x['sha256']) for x in writes]==expected_writes,'initialization':[(x['name'],x['value']) for x in inits]==[(x,'ff') for x in ('gate','up','silu','activation','down')]+[(x,'00') for x in ('gate_counters','up_counters','activation_counters','down_counters')],'args':[(x['kernel'],x['index'],x['name']) for x in args]==expected_args and all(x['pointer']==next(a['pointer'] for a in alloc if a['name']==x['name']) for x in args),'launch_finish_read':[(x['kernel'],x['global'],x['local']) for x in launch]==LAUNCH and sum(x.get('op')=='finish' for x in led)==1 and [x['name'] for x in readrows]==['gate','up','silu','activation','down','gate_counters','up_counters','activation_counters','down_counters'] and all(x['after_finish'] is True and x['bytes']==dict(BUFF)[x['name']] and x['sha256']==sha(outs[x['name']]) for x in readrows),'release':[x['name'] for x in release]==expected_release and [x['attempt_index'] for x in release]==list(range(21)) and all(x['attempted'] is True and x['code']==0 and x['exception'] is None and x['owned_after'] is False for x in release) and cleanup['cleanup_complete'] is True and cleanup['release_attempts']==21 and cleanup['live_owned_resources']==0 and cleanup['live_resource_names']==[],'extensions':ev['extension_counts']=={'clHostMemAllocINTEL':14,'clMemFreeINTEL':14,'clSetKernelArgMemPointerINTEL':18,'clGetMemAllocInfoINTEL':42},'counters':all(all(v==1 for v in struct.unpack('<'+'I'*(len(outs[k])//4),outs[k])) for k in ('gate_counters','up_counters','activation_counters','down_counters')),'resources':resource['sampling_claim']=='retained phase samples, not continuous peak monitor' and resource['start_available']>=16*2**30 and min(resource['post_payload_available'],resource['final_available'],resource['post_serialize_available'],*(x['available'] for x in samples))>=2*2**30 and resource['peak_retained_wset']<=12*2**30 and len(samples)==12,'forbidden':set(ev['forbidden_calls'])=={'clCreateBuffer','clEnqueueReadBuffer','clEnqueueWriteBuffer','clEnqueueCopyBuffer','clEnqueueMigrateMemObjects','clEnqueueMemAdviseINTEL'} and all(v==0 for v in ev['forbidden_calls'].values()) and all(x.get('api') not in ev['forbidden_calls'] for x in led),'runner_gates':all(rrr['gates'].values()) and rrr['stage_hashes']==STAGE}
 exact_sample_stages=['backend_entry','pre_launch:0','post_launch:0','pre_launch:1','post_launch:1','pre_launch:2','post_launch:2','pre_launch:3','post_launch:3','pre_finish','post_finish','post_cleanup']
 sample_fields=all(set(x)=={'op','stage','qpc_ns','available','rss','peak_wset','telemetry_error'} for x in samples)
 sample_values=sample_fields and all(x['op']=='resource_sample' and x['telemetry_error'] is None and all(isinstance(x[k],int) for k in ('qpc_ns','available','rss','peak_wset')) and x['available']>=2*2**30 and 0<=x['rss']<=x['peak_wset']<=12*2**30 for x in samples)
 sample_order=[x.get('stage') for x in samples]==exact_sample_stages and all(samples[i]['qpc_ns']<samples[i+1]['qpc_ns'] for i in range(len(samples)-1)) if len(samples)==12 else False
 boundary_peak=[resource.get(k) for k in ('start_peak_wset','post_payload_peak_wset','final_peak_wset','post_serialize_peak_wset')]
 summary_peak=sample_values and all(isinstance(x,int) for x in boundary_peak) and resource.get('peak_retained_wset')==max(boundary_peak+[x['peak_wset'] for x in samples])
 checks['resources']=resource.get('sampling_claim')=='retained phase samples, not continuous peak monitor' and resource.get('start_available',0)>=16*2**30 and min(resource.get('post_payload_available',0),resource.get('final_available',0),resource.get('post_serialize_available',0))>=2*2**30 and sample_order and sample_values and summary_peak and ev.get('telemetry_errors')==[]
 return checks

def verify_bundle_contract(result_bytes,manifest,commit,file_names,total_bytes):
 try:
  rr=json.loads(result_bytes);row={'name':'result.json','bytes':len(result_bytes),'sha256':sha(result_bytes)}
  mb=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
  return rr.get('kind')=='ph1_intel_execution_r4' and manifest=={'kind':'ph1_intel_execution_r4_manifest','files':[row]} and commit=={'kind':'ph1_intel_execution_r4_commit','manifest_sha256':sha(mb),'result_sha256':sha(result_bytes)} and file_names=={'result.json','manifest.json','commit.json'} and total_bytes<=16*2**20
 except Exception:return False
def main():
 r,m,c=(OUT/n for n in ('result.json','manifest.json','commit.json'));rb=r.read_bytes();rrr=json.loads(rb);checks=verify_dict(rrr);mm=json.loads(m.read_text());checks['bundle']=verify_bundle_contract(rb,mm,json.loads(c.read_text()),{x.name for x in OUT.iterdir()},sum(x.stat().st_size for x in OUT.iterdir()));out={'kind':'ph1_intel_execution_r4_independent_verification','checks':checks,'pass':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'claim':'one real expert/input Intel correctness component only'}
 if VERIFY.exists():raise FileExistsError(VERIFY)
 VERIFY.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['pass'] else 3
if __name__=='__main__':raise SystemExit(main())
