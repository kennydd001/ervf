#!/usr/bin/env python3
"""Independent CAP0P evidence verifier."""
import hashlib,json,os,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'reports/runs/streamq5_moe/het_next_cap0p_process_isolated';RES=D/'cap0p_result.json';COM=D/'cap0p_commit.json';LOCK=ROOT/'reports/streamq5_moe/het_next_cap0p_verifier_lock.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays():
 x=[0x45585430]
 for _ in range(1,1024):x.append((1664525*x[-1]+1013904223)&0xffffffff)
 i=[(((((v^0xa5a5a5a5)<<7)|((v^0xa5a5a5a5)>>25))+0x3c6ef372)&0xffffffff) for v in x];n=[((((((v+0x9e3779b9)&0xffffffff)>>11)|(((v+0x9e3779b9)&0xffffffff)<<21))&0xffffffff)^0xc3c3c3c3 for v in x];return i,n
def verify():
 l=json.loads(LOCK.read_text());c={};c['bindings']=all(sha(ROOT/v['path'])==v['sha256'] for v in l['files'].values()) and not l['execution_open'];c['files']=RES.exists() and COM.exists()
 if not c['files']:return {'pass':False,'checks':c}
 r=json.loads(RES.read_text());m=json.loads(COM.read_text());c['commit']=m=={'result':{'bytes':RES.stat().st_size,'sha256':sha(RES)}};want=dict(zip(('intel','nvidia'),arrays()));reps=r.get('repetitions',[]);c['cardinality']=len(reps)==3 and [z['epoch'] for z in reps]==[1,2,3]
 c['raw_exact']=c['cardinality'] and all(len(z[d].get('output_words',[]))==1024 and z[d]['output_words']==want[d] and z[d]['different_words']==0 and hashlib.sha256(struct.pack('<1024I',*z[d]['output_words'])).hexdigest()==z[d]['sha256'] for z in reps for d in ('intel','nvidia'))
 c['overlap']=all(z['strict_overlap'] is True and max(z[d]['submit_ns'] for d in ('intel','nvidia'))<min(z[d]['done_ns'] for d in ('intel','nvidia')) for z in reps);ch=r.get('children',{});c['processes']=set(ch)=={'intel','nvidia'} and len({x['pid'] for x in ch.values()})==2 and all(x['ready']['type']=='ready' and x['ready']['pid']==x['pid'] for x in ch.values());c['cleanup']=set(r.get('cleanup',{}))=={'intel','nvidia'} and all(r['cleanup'][d]['type']=='cleanup' and r['exits'][d]==0 for d in ('intel','nvidia'));c['job']=r.get('job')=={'kill_on_close':True,'assigned_before_resume':True,'closed':True};c['kind']=r.get('kind')=='het_next_cap0p_process_isolated' and r.get('status')=='process_isolated_cohabitation_positive';return {'kind':'cap0p_independent_verification','pass':all(c.values()),'checks':c,'count':len(c)}
if __name__=='__main__':o=verify();print(json.dumps(o,sort_keys=True));raise SystemExit(0 if o['pass'] else 1)
