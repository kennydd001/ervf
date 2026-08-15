#!/usr/bin/env python3
"""Independent PH0-R3 verifier. Does not import runner/common/backend/codec helpers."""
from __future__ import annotations
import argparse,hashlib,json,struct,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/runs/streamq5_moe/het_next_l0_ph0r3_single_projection';REPORTS=ROOT/'reports/streamq5_moe'
FILES={p:R/f'{p}.json' for p in ('cpu','intel','nvidia')};COMMIT=R/'commit.json';HEADER=struct.Struct('<4sHHHBBIIH2xIII28s')
SOURCE=Path(r'C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors');D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def bf(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def rb(values):
 b=np.asarray(values,np.float32).view(np.uint32);return ((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)
def rebuild():
 with SOURCE.open('rb') as f:f.seek(3_498_051_416);src=f.read(2_097_152)
 with D2.open('rb') as f:f.seek(155_138_788);inp=f.read(4_096)
 if sha(src)!='05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c' or sha(inp)!='5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f':raise ValueError('range')
 v=bf(np.frombuffer(src,'<u2')).reshape(512,2048);w=v.reshape(512,16,128);m=np.max(np.abs(w),-1,keepdims=True);s=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(w/s,np.float32)),-15,15),0).astype(np.int16);field=(q+15).astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(field<<(np.arange(8,dtype=np.uint64)*5),1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],1).astype(np.uint8).tobytes();sc=rb(s.reshape(-1)).astype('<u2').tobytes();crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=HEADER.pack(b'SQ5M',1,0,50,0,5,512,2048,128,len(codes),len(sc),crc,bytes(28));rec=head+codes+sc+bytes(4032)
 if (sha(codes),sha(sc),sha(rec),crc)!=('20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b','658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8','e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9',1976639022):raise ValueError('codec')
 return rec,inp
def check_backend(name,row,cpu):
 b=row['backend'];out=bytes.fromhex(b['output_hex']);cnt=np.frombuffer(bytes.fromhex(b['counters_hex']),'<u4');checks=[len(out)==1024,np.array_equal(np.frombuffer(out,'<u2'),cpu),len(cnt)==512 and np.all(cnt==1),np.all(np.frombuffer(out,'<u2')!=0xffff),b['ledger'][-1]['cleanup_complete']]
 if name=='intel':checks += [b['forbidden_copy_calls']==0,b['enqueue_calls']==1,b['identity']['name']=='Intel(R) Arc(TM) Pro 140T GPU (32GB)',b['identity']['pci']=='0000:00:02.0']
 else:
  ops=[x.get('op') for x in b['ledger'] if 'op' in x];checks += [b['memset_calls']==2,b['h2d_calls']==2,b['kernel_calls']==1,b['d2h_calls']==2,b['sync_calls']==1,ops==['memset','memset','H2D','H2D','kernel','D2H','D2H','synchronize'],b['identity']['pci']=='0000:01:00.0','tiled_partition<8>' in b['source']]
 return checks
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--precommit',action='store_true');a=ap.parse_args();checks=[]
 rec,inp=rebuild();cpurow=json.loads(FILES['cpu'].read_text());cpu=np.frombuffer(bytes.fromhex(cpurow['output_hex']),'<u2');checks += [cpurow['status']=='cpu_committed',len(cpu)==512,len(cpurow['controls'])==8,all(x['pass'] for x in cpurow['controls']),cpurow['record_evidence']['record_sha256']==sha(rec)]
 intel=json.loads(FILES['intel'].read_text());nvidia=json.loads(FILES['nvidia'].read_text());checks += check_backend('intel',intel,cpu)+check_backend('nvidia',nvidia,cpu)
 checks += [intel['status']=='intel_committed',nvidia['status']=='nvidia_committed',intel['backend']['identity']['pci']!=nvidia['backend']['identity']['pci'],intel['resources'][0]['available']>=2<<30,nvidia['resources'][0]['available']>=2<<30,max(x['peak'] for x in intel['resources']+nvidia['resources'])<=2<<30]
 if not a.precommit:
  c=json.loads(COMMIT.read_text());checks += [c['status']=='positive_single_real_projection_component',all(fsha(R/name)==v['sha256'] for name,v in c['files'].items())]
 result={'kind':'het_next_l0_ph0r3_independent_verification','checks':checks,'passed':sum(checks),'total':len(checks),'all_pass':all(checks),'claim_boundary':'validation-only single real projection'}
 print(json.dumps(result,sort_keys=True));return 0 if all(checks) else 2
if __name__=='__main__':raise SystemExit(main())
