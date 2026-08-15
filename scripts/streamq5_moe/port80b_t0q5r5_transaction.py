#!/usr/bin/env python3
"""Reusable strict create-new multi-file transaction state machine."""
from __future__ import annotations
import hashlib,json,os,uuid
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def fsync_file(p):
 with Path(p).open('rb') as f:os.fsync(f.fileno())
def fsync_dir(p):
 if os.name!='nt':
  fd=os.open(str(Path(p)),os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
def begin(root,name,finals):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);nonce=uuid.uuid4().hex;marker=root/f'{name}_commit.json';journal=root/f'{name}_{nonce}.journal.inprogress'
 if marker.exists() or journal.exists() or any(Path(p).exists() for p in finals):raise FileExistsError(name)
 temps={str(Path(p)):str(root/f'{Path(p).name}.{nonce}.inprogress') for p in finals};state={'kind':'t0q5r5_transaction','name':name,'nonce':nonce,'state':'building','marker':str(marker),'files':temps};journal.write_bytes(canon(state)+b'\n');fsync_file(journal);fsync_dir(root);return state,journal
def commit(state,journal):
 marker=Path(state['marker']);root=marker.parent;entries={}
 for final,temp in state['files'].items():
  p=Path(temp);fsync_file(p);entries[Path(final).name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 complete={**state,'state':'complete','files':entries};mt=marker.with_name(marker.name+'.'+state['nonce']+'.inprogress');mt.write_bytes(canon(complete)+b'\n');fsync_file(mt)
 for final,temp in state['files'].items():
  final=Path(final)
  if final.exists():raise FileExistsError(final)
  os.rename(temp,final)
 fsync_dir(root)
 if marker.exists():raise FileExistsError(marker)
 os.rename(mt,marker);fsync_dir(root);Path(journal).unlink();fsync_dir(root);return complete
def verify(marker,finals):
 marker=Path(marker);m=json.loads(marker.read_text());expected={Path(p).name for p in finals};return m.get('state')=='complete' and set(m['files'])==expected and all(m['files'][Path(p).name]['bytes']==Path(p).stat().st_size and m['files'][Path(p).name]['sha256']==sha(p) for p in finals)
def recover(root,name,quarantine):
 root=Path(root);q=Path(quarantine);q.mkdir(parents=True,exist_ok=True);moved={};marker=root/f'{name}_commit.json'
 for p in root.glob(f'{name}_*.journal.inprogress'):
  state=json.loads(p.read_text());candidates=[Path(x) for x in state['files'].values()]+[Path(x) for x in state['files']]+[marker,marker.with_name(marker.name+'.'+state['nonce']+'.inprogress'),p]
  if marker.exists() and verify(marker,[Path(x) for x in state['files']]):continue
  for x in candidates:
   if x.exists():dest=q/f'{state["nonce"]}_{x.name}';os.rename(x,dest);moved[str(x)]=str(dest)
 fsync_dir(root);fsync_dir(q);return moved
def simulate(root):
 root=Path(root);a=root/'a.bin';b=root/'b.json';state,j=begin(root,'sim',(a,b));Path(state['files'][str(a)]).write_bytes(b'abc');Path(state['files'][str(b)]).write_bytes(b'{}\n');commit(state,j);clean=verify(root/'sim_commit.json',(a,b));state2,j2=begin(root,'crash',(root/'c.bin',));Path(state2['files'][str(root/'c.bin')]).write_bytes(b'x');moved=recover(root,'crash',root/'failed');return {'clean_commit':clean,'recovery_moved':bool(moved),'no_inprogress':not any(root.glob('*.inprogress'))}
