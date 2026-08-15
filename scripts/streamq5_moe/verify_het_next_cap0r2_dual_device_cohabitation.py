#!/usr/bin/env python3
"""Independent CAP0 evidence verifier; imports no CAP0 runner/protocol/kernel."""
from __future__ import annotations
import hashlib, json, math, os, struct, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/streamq5_moe'; D=ROOT/'reports/runs/streamq5_moe/het_next_cap0r2_dual_device_cohabitation'
RESULT=D/'cap0r2_result.json'; COMMIT=D/'cap0r2_commit.json'; FAILURE=D/'cap0r2_failure.json'; VLOCK=R/'het_next_cap0r2_verifier_lock.json'
SEED=0x4845544E45585430; COUNT=1024
FIX={'input':'a9d32afd712f6ac80ef7739b11c2baa59e4f84c2067e20307f175de4e8a1acca','intel':'c83e434be87333bc6bf15d3f0ee492c3e3f9d65b847902bea55310165a42923f','nvidia':'f07c3d87d952d1dc82c65d90f467af87426c1658267b7d94f359122e73eafd5f'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def packed(a):return hashlib.sha256(struct.pack('<1024I',*a)).hexdigest()
def arrays():
 x=[SEED&0xffffffff]
 for _ in range(1,COUNT):x.append((1664525*x[-1]+1013904223)&0xffffffff)
 yi=[(((((v^0xa5a5a5a5)<<7)|((v^0xa5a5a5a5)>>25))+0x3c6ef372)&0xffffffff) for v in x]
 yn=[((((((v+0x9e3779b9)&0xffffffff)>>11)|(((v+0x9e3779b9)&0xffffffff)<<21))&0xffffffff)^0xc3c3c3c3 for v in x]
 return x,yi,yn
def verify():
 lock=json.loads(VLOCK.read_text()); checks={}; bindings={k:sha(ROOT/v['path'])==v['sha256'] and (ROOT/v['path']).stat().st_size==v['bytes'] for k,v in lock['files'].items()};checks['bindings']=all(bindings.values()) and lock['execution_open'] is False and lock['audit_token']=='PENDING_INDEPENDENT_SOURCE_AUDIT'
 if FAILURE.exists() and not (RESULT.exists() or COMMIT.exists()):
  f=json.loads(FAILURE.read_text());cl=f.get('cleanup',{});checks['valid_failure']=f.get('kind')=='het_next_cap0_failure' and f.get('status')=='valid_failure' and f.get('scope_provenance')=={'opened_paths':[],'forbidden_payload_count':0} and bool(f.get('error')) and bool(f.get('traceback')) and isinstance(f.get('dispositions'),list) and isinstance(cl,dict) and (cl.get('cleanup_not_initialized') is True or (not cl.get('hung_threads') and cl.get('closed') is True and all(z.get('release_attempts')==1 for z in cl.get('ledger',[]))));return {'kind':'het_next_cap0_independent_verification','pass':False,'valid_failure':all(checks.values()),'checks':checks,'bindings':bindings}
 checks['three_files']=RESULT.exists() and COMMIT.exists() and not FAILURE.exists()
 if not checks['three_files']:return {'kind':'het_next_cap0_independent_verification','pass':False,'checks':checks,'bindings':bindings}
 r=json.loads(RESULT.read_text());c=json.loads(COMMIT.read_text());checks['commit']=c=={'kind':'het_next_cap0_commit','result':{'bytes':RESULT.stat().st_size,'sha256':sha(RESULT)},'status':r.get('status')}
 x,yi,yn=arrays();checks['fixtures']=(packed(x),packed(yi),packed(yn))==(FIX['input'],FIX['intel'],FIX['nvidia']) and r.get('fixtures')=={'seed':SEED,'word_count':COUNT,'input_sha256':FIX['input'],'intel_sha256':FIX['intel'],'nvidia_sha256':FIX['nvidia']}
 checks['kind_status']=r.get('kind')=='het_next_cap0r2_dual_device_cohabitation' and r.get('status')=='dual_device_cohabitation_positive' and r.get('claim_boundary')=='4KiB dual-device lifecycle/correctness only; no performance or model claim'
 tr=r.get('thread_rows',{});roles=('coordinator','intel','nvidia','monitor');checks['topology']=set(tr)==set(roles) and [tr[k].get('logical_processor') for k in roles]==[0,2,4,6] and all(tr[k].get('processor_group')==0 for k in roles) and len({tr[k].get('physical_core_record_offset') for k in roles})==4 and len({tr[k].get('thread_id') for k in roles})==4 and all(tr[k].get('end_qpc_ns',0)>tr[k].get('start_qpc_ns',0) for k in roles)
 init=r.get('initialization',{});ii=init.get('intel',{});ni=init.get('nvidia',{});checks['identities']=set(init)=={'intel','nvidia'} and 'Intel' in ii.get('vendor','') and 'Arc' in ii.get('name','') and ii.get('allocation')=={'type':0x4197,'base_matches':True,'bytes':4096,'api':'clHostMemAllocINTEL','cl_mem':False,'write_read_buffer':False,'migrate_prefetch':False} and 'cl_intel_unified_shared_memory' in ii.get('extensions',[]) and ii.get('usm_capabilities',{}).get('host',0)!=0 and ni.get('allocation')=={'pinned_host_bytes':4096,'device_bytes':4096,'managed':False} and ni.get('free_memory_start',0)>=64<<20 and ni.get('binary_bytes',0)>0 and ii.get('binary_bytes',0)>0 and len(ii.get('binary_sha256',''))==64 and len(ni.get('binary_sha256',''))==64 and ii.get('build_options')=='-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt' and ni.get('compile_options')==['--std=c++14','--fmad=false'] and ni.get('binary_kind')=='PTX' and isinstance(ii.get('build_log'),str) and isinstance(ni.get('compile_log'),str)
 reps=r.get('repetitions',{});checks['cardinality']=set(reps)=={'intel','nvidia'} and all(len(reps[k])==3 for k in reps) and [z.get('epoch') for z in reps['intel']]==[1,2,3] and [z.get('epoch') for z in reps['nvidia']]==[1,2,3]
 expected={'intel':yi,'nvidia':yn};checks['exact_outputs']=checks['cardinality'] and all(len(z.get('output_words',[]))==1024 and z['output_words']==expected[k] and sum(a!=b for a,b in zip(z['output_words'],expected[k]))==z.get('different_words')==0 and packed(z['output_words'])==z.get('output_sha256')==FIX[k] and z.get('expected_sha256')==FIX[k] and z.get('done_qpc_ns',0)>z.get('submit_qpc_ns',0) for k in ('intel','nvidia') for z in reps[k]) and len({z['output_sha256'] for z in reps['intel']})==1 and len({z['output_sha256'] for z in reps['nvidia']})==1
 life=r.get('lifecycle',[]);checks['concurrency']=len(life)==3 and [z.get('epoch') for z in life]==[1,2,3] and all(z.get('strict_work_interval_overlap') is True and z.get('coordinator_t1',0)>z.get('coordinator_t0',0) for z in life)
 pl=r.get('protocol_log',[]);checks['protocol']=sum(z.get('op')=='publish' for z in pl)==3 and sum(z.get('op')=='release' for z in pl)==3 and sum(z.get('op')=='ack_done' for z in pl)==6 and sum(z.get('op')=='collect_reset' for z in pl)==3 and sum(z.get('op')=='initialized' for z in pl)==2 and len(r.get('primitive_calls',[]))>=80
 mon=r.get('monitor',{});samples=mon.get('samples',[]);intervals=mon.get('interval_ms',[]);paths=[r'\Memory\Page Reads/sec',r'\Memory\Pages Input/sec',r'\Paging File(_Total)\% Usage'];checks['monitor']=mon.get('paths')==paths and mon.get('valid_protocol') is True and len(samples)>=11 and len(intervals)==len(samples)-1 and all(80<=v<=120 for v in intervals) and all(z.get('lateness_ms',999)<=20 and set(z.get('values',{}))==set(paths) and all(math.isfinite(v) for v in z['values'].values()) and all(s.get('return')==0 and s.get('cstatus')==0 for s in z.get('statuses',{}).values()) for z in samples)
 ledger=r.get('ledger',[]);expected={'win32_event':18,'win32_srwlock':1,'intel_context':1,'intel_queue':1,'intel_program':1,'intel_kernel':1,'intel_host_usm':1,'intel_event':3,'nvidia_context':1,'nvidia_nvrtc_program':1,'nvidia_module':1,'nvidia_stream':1,'nvidia_pinned_host':1,'nvidia_device_memory':1,'nvidia_start_event':3,'nvidia_end_event':3,'pdh_query':1,'pdh_counter':3,'pdh_waitable_timer':1};actual={k:sum(z.get('kind')==k for z in ledger) for k in expected};checks['cleanup']=actual==expected and len({z.get('id') for z in ledger})==len(ledger)==sum(expected.values()) and not r.get('cleanup_errors') and not r.get('errors') and all(z.get('release_attempts')==1 and z.get('release_code')==0 and z.get('final_state')=='released' and z.get('creator_thread_id',0)>0 for z in ledger)
 checks['process']=r.get('process',{}).get('pid',0)>0 and r.get('process',{}).get('create_filetime',0)>0 and '--phase' in r.get('process',{}).get('argv',[]) and '--ack' in r.get('process',{}).get('argv',[])
 checks['resources_pci']=r.get('resources',{}).get('available_ram_start',0)>=r.get('resources',{}).get('minimum_ram')==2<<30 and r.get('pci_distinct') is True and ii.get('eligible_device_count')==ni.get('eligible_device_count')==1 and isinstance(ii.get('pci'),dict) and bool(ni.get('pci_bus_id')) and f"{ii['pci']['domain']:04x}:{ii['pci']['bus']:02x}:{ii['pci']['device']:02x}.{ii['pci']['function']}".lower()!=ni['pci_bus_id'].lower()
 return {'kind':'het_next_cap0_independent_verification','pass':all(checks.values()),'checks':checks,'bindings':bindings,'check_count':len(checks)}
if __name__=='__main__':
 out=verify();print(json.dumps(out,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
