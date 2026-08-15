#!/usr/bin/env python3
"""Static, no-device PH1 Intel compile preflight."""
from __future__ import annotations
import ast, hashlib, json, struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; S=ROOT/'scripts/streamq5_moe'; R=ROOT/'reports/streamq5_moe'
BACKEND=S/'het_next_l0_ph1_intel_backend.py'; RUNNER=S/'run_het_next_l0_ph1_intel_compile_r0.py'; PR=R/'HET_NEXT_L0_PH1_INTEL_COMPILE_R0_PREREGISTRATION_2026-08-14.md'; LOCK=R/'het_next_l0_ph1_intel_compile_lock.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_literal():
 tree=ast.parse(BACKEND.read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SRC' for t in node.targets):return ast.literal_eval(node.value)
 raise RuntimeError('SRC')
def mul(a,b):
 # Independent exact rational product -> direct BF16 enumeration oracle for fixed vectors.
 import fractions
 def val(w):
  sign=-1 if w>>15 else 1;e=(w>>7)&255;f=w&127
  if e==255:raise ValueError('nonfinite')
  return fractions.Fraction(sign*f,2**133) if e==0 else fractions.Fraction(sign*(128+f),128)*fractions.Fraction(2)**(e-127)
 x=val(a)*val(b); candidates=[]
 for w in range(0x10000):
  if ((w>>7)&255)==255:continue
  y=val(w); candidates.append((abs(y-x),w&1,w))
 return min(candidates)[2]
def main():
 src=source_literal(); lock=json.loads(LOCK.read_text()); tests={}
 tests['hashes']=lock['backend_sha256']==sha(BACKEND) and lock['runner_sha256']==sha(RUNNER) and lock['prereg_sha256']==sha(PR) and lock['source_sha256']==hashlib.sha256(src.encode()).hexdigest()
 tests['closed']=lock['execution_open'] is False and lock['audit_token']=='PENDING'
 tests['entrypoints']=all(src.count('void '+x+'(')==1 for x in ('gate_linear','up_linear','activation','down_linear'))
 tests['geometry_contract']=all(x in src for x in ('float partial[32]','float partial[8]','for (int distance = 16','for (int distance = 4','intel_reqd_sub_group_size(8)'))
 tests['integer_activation']=all(x in src for x in ('multiply_bf16_exact','lut[(uint)gate_word]','activated[row] = activation_word'))
 tests['no_bad_options']=not any(x in src for x in ('fast-relaxed-math','finite-math-only','unsafe-math','mad-enable','ftz'))
 vectors={(0x0000,0x3f80):0x0000,(0x8000,0x3f80):0x8000,(0x0000,0xbf80):0x8000,(0x8000,0xbf80):0x0000,(0x3f80,0x3f80):0x3f80,(0xbf80,0x3f80):0xbf80,(0x3f00,0x4000):0x3f80,(0x0001,0x3f80):0x0001,(0x0001,0x3f00):0x0000,(0x0003,0x3f00):0x0002,(0x7f7f,0x0001):0x3cff}
 tests['vectors']=all(mul(a,b)==z for (a,b),z in vectors.items())
 tree=ast.parse(Path(__file__).read_text()); imported={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imported.update(alias.name.split('.')[0] for alias in node.names)
  elif isinstance(node,ast.ImportFrom) and node.module:imported.add(node.module.split('.')[0])
 tests['no_payload_import']=not bool(imported & {'torch','safetensors','cupy','ctypes','transformers'})
 tests['output_absent']=not (R/'het_next_l0_ph1_intel_compile_r0').exists()
 print(json.dumps({'kind':'ph1_intel_compile_r0_static_preflight','tests':tests,'pass':all(tests.values()),'passed':sum(tests.values()),'total':len(tests)},sort_keys=True,indent=2));return 0 if all(tests.values()) else 3
if __name__=='__main__':raise SystemExit(main())
