#!/usr/bin/env python3
"""Exact SQ5M R3 wire, decoded digest and canonical-manifest contract; no execution."""
from __future__ import annotations
import hashlib,json,struct,zlib
import numpy as np
import torch
REVISION="a19358a7659bd1f564300250ee189120c49a562f";HEADER_FORMAT="<4sHHHBBIIH2xIII28s";HEADER_BYTES=struct.calcsize(HEADER_FORMAT)
GROUP=128;CODE_BYTES=655_360;SCALE_BYTES=16_384;PADDING_BYTES=4_032;MATRIX_BYTES=675_840;EXPERT_BYTES=2_027_520;RECORDS=513;MATRICES=1539;BANK_BYTES=1_040_117_760;SHAPES=((512,2048),(512,2048),(2048,512));NAMES=('gate','up','down')
def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def sha(x):return hashlib.sha256(x).hexdigest()
def quantize(v):
 if v.dtype!=torch.bfloat16 or tuple(v.shape) not in set(SHAPES):raise ValueError('source schema')
 r,c=v.shape;w=v.float().reshape(r,c//GROUP,GROUP);mx=w.abs().amax(-1,keepdim=True);s=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/s).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);f=(q.to(torch.int16)+15).cpu().numpy()
 if f.min()<0 or f.max()>30:raise ValueError('field31')
 z=f.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(z<<(np.arange(8,dtype=np.uint64)*5),-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=s.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes();return codes,scales
def decode(codes,scales,rows,cols):
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;f=np.stack([(w>>(5*i))&31 for i in range(8)],-1)
 if bool((f==31).any()) or f.max()>30:raise ValueError('field31')
 q=torch.from_numpy((f.astype(np.int16)-15).reshape(rows,cols//GROUP,GROUP));bits=torch.from_numpy(np.frombuffer(scales,'<u2').copy()).to(torch.uint16);s=bits.view(torch.bfloat16).float().reshape(rows,cols//GROUP,1);return (q.float()*s).reshape(rows,cols).to(torch.bfloat16)
def make_record(v,expert,projection):
 rows,cols=v.shape;codes,scales=quantize(v);crc=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff;header=struct.pack(HEADER_FORMAT,b'SQ5M',1,0,expert,projection,5,rows,cols,GROUP,len(codes),len(scales),crc,bytes(28));pad=bytes(PADDING_BYTES);record=header+codes+scales+pad
 if HEADER_BYTES!=64 or len(record)!=MATRIX_BYTES:raise ValueError('wire')
 d=decode(codes,scales,rows,cols);return record,{'revision':REVISION,'layer':0,'expert':expert,'shared':expert==512,'projection':projection,'projection_name':NAMES[projection],'rows':rows,'columns':cols,'source_dtype':'torch.bfloat16','source_sha256':sha(raw(v)),'header_sha256':sha(header),'codes_sha256':sha(codes),'scales_sha256':sha(scales),'padding_sha256':sha(pad),'decoded_weight_sha256':sha(raw(d)),'record_sha256':sha(record),'crc32':crc,'header_bytes':64,'code_bytes':CODE_BYTES,'scale_bytes':SCALE_BYTES,'padding_bytes':PADDING_BYTES,'record_bytes':MATRIX_BYTES}
def canonical_core(core):return json.dumps(core,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def manifest_file(core):
 digest=sha(canonical_core(core));envelope={'kind':'port80b_t0q5r3_manifest','manifest':core,'manifest_sha256':digest};return canonical_core(envelope)+b'\n'
