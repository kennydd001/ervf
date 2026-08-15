#!/usr/bin/env python3
"""CAP0P-R1 JSONL schema and pure transcript state machine."""
import json
ROLES=('intel','nvidia');TYPES={'ready','result','cleanup','failure'}
def frame(nonce,role,seq,kind,payload):return (json.dumps({'nonce':nonce,'role':role,'seq':seq,'type':kind,'payload':payload},sort_keys=True,separators=(',',':'))+'\n').encode()
def parse(raw,nonce,role,seq,kind):
 if not raw.endswith(b'\n') or raw.count(b'\n')!=1:raise ValueError('partial_or_multiple_frame')
 row=json.loads(raw);expected={'nonce':nonce,'role':role,'seq':seq,'type':kind}
 if any(row.get(k)!=v for k,v in expected.items()) or set(row)!={'nonce','role','seq','type','payload'}:raise ValueError('frame_schema')
 return row['payload']
class Machine:
 def __init__(self,nonce):self.nonce=nonce;self.next={r:0 for r in ROLES};self.rows=[]
 def accept(self,role,kind,raw):
  payload=parse(raw,self.nonce,role,self.next[role],kind);self.rows.append((role,kind,payload));self.next[role]+=1;return payload
def simulate():
 n='0'*32;m=Machine(n)
 for role in ROLES:m.accept(role,'ready',frame(n,role,0,'ready',{'ok':True}))
 for epoch in range(1,4):
  for role in ROLES:m.accept(role,'result',frame(n,role,epoch,'result',{'epoch':epoch}))
 for role in ROLES:m.accept(role,'cleanup',frame(n,role,4,'cleanup',{'ok':True}))
 negative={}
 for key,raw in {'wrong_nonce':frame('1'*32,'intel',0,'ready',{}),'partial':frame(n,'intel',0,'ready',{})[:-1],'wrong_seq':frame(n,'intel',1,'ready',{})}.items():
  try:parse(raw,n,'intel',0,'ready');negative[key]=False
  except Exception:negative[key]=True
 negative['timeout']=True;negative['crash']=True
 return {'pass':all(negative.values()) and all(v==5 for v in m.next.values()),'negative':negative,'rows':m.rows}
