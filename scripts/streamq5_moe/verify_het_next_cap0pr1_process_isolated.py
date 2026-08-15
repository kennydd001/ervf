#!/usr/bin/env python3
import hashlib,json,struct
from pathlib import Path
from het_next_cap0pr1_protocol import Machine
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'reports/runs/streamq5_moe/het_next_cap0pr1_process_isolated';RES=D/'cap0pr1_result.json';COM=D/'cap0pr1_commit.json';LOCK=ROOT/'reports/streamq5_moe/het_next_cap0pr1_verifier_lock.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expected():
 x=[0x45585430]
 for _ in range(1,1024):x.append((1664525*x[-1]+1013904223)&0xffffffff)
 return {'intel':[(((((v^0xa5a5a5a5)<<7)|((v^0xa5a5a5a5)>>25))+0x3c6ef372)&0xffffffff) for v in x],'nvidia':[((((((v+0x9e3779b9)&0xffffffff)>>11)|(((v+0x9e3779b9)&0xffffffff)<<21))&0xffffffff)^0xc3c3c3c3 for v in x]}
def verify():
 l=json.loads(LOCK.read_text());c={'bindings':all(sha(ROOT/v['path'])==v['sha256'] for v in l['files'].values()) and not l['execution_open'],'files':RES.exists() and COM.exists()}
 if not c['files']:return {'pass':False,'checks':c}
 r=json.loads(RES.read_text());c['commit']=json.loads(COM.read_text())=={'result':{'bytes':RES.stat().st_size,'sha256':sha(RES)}};want=expected();reps=r.get('repetitions',[]);c['cardinality']=len(reps)==3 and [z['epoch'] for z in reps]==[1,2,3];c['raw']=c['cardinality'] and all(z[d]['output_words']==want[d] and z[d]['different_words']==0 and sha256_words(z[d]['output_words'])==z[d]['sha256'] for z in reps for d in want);c['overlap']=all(max(z[d]['submit_ns'] for d in want)<min(z[d]['done_ns'] for d in want) for z in reps);c['process']=all(x['exit_code']==0 and x['exit_filetime']>x['create_filetime'] and len(x['transcript_hex'])==5 for x in r['process'].values()) and len({x['pid'] for x in r['process'].values()})==2;c['job']=r['job']=={'assigned_before_resume':True,'kill_on_close':True,'closed':True};c['cleanup']=all(r['cleanup'][d]['device']==d and all(x.get('code',x.get('used_bytes',0))==0 for x in r['cleanup'][d]['rows']) for d in want);c['transcripts']=transcripts(r);return {'kind':'cap0pr1_verification','pass':all(c.values()),'checks':c}
def sha256_words(a):return hashlib.sha256(struct.pack('<1024I',*a)).hexdigest()
def transcripts(r):
 try:
  m=Machine(r['nonce'])
  for role in ('intel','nvidia'):
   raw=[bytes.fromhex(x) for x in r['process'][role]['transcript_hex']];m.accept(role,'ready',raw[0]);[m.accept(role,'result',raw[e]) for e in range(1,4)];m.accept(role,'cleanup',raw[4])
  return True
 except Exception:return False
if __name__=='__main__':o=verify();print(json.dumps(o,sort_keys=True));raise SystemExit(0 if o['pass'] else 1)
