#!/usr/bin/env python3
"""Frozen T0Q5-R1 wire contract. Source-auditable; not an experiment runner."""
from __future__ import annotations
import numpy as np
import torch

REVISION="a19358a7659bd1f564300250ee189120c49a562f"
SHARD_BYTES=3_999_619_288
SHARD_SHA256="8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
GROUP=128;QMIN=-15;QMAX=15;BIAS=15;FORBIDDEN_FIELD=31
HEADER_BYTES=64;CODE_BYTES=655_360;SCALE_BYTES=16_384;PADDING_BYTES=4_032
MATRIX_BYTES=675_840;EXPERT_BYTES=2_027_520;EXPERT_RECORDS=513;BANK_BYTES=1_040_117_760
PROJECTIONS=(("gate",512,2048),("up",512,2048),("down",2048,512))

def quantize_matrix_bf16_source(source_bf16:torch.Tensor)->tuple[bytes,bytes]:
 if tuple(source_bf16.shape) not in ((512,2048),(2048,512)) or source_bf16.dtype!=torch.bfloat16:raise ValueError("matrix schema")
 rows,columns=source_bf16.shape;groups=source_bf16.float().reshape(rows,columns//GROUP,GROUP);amax=groups.abs().amax(-1,keepdim=True);scale=torch.where(amax>0,amax/15,torch.ones_like(amax));q=torch.round(groups/scale).clamp(QMIN,QMAX).to(torch.int8);q=torch.where(amax>0,q,torch.zeros_like(q));fields=(q.to(torch.int16)+BIAS).to(torch.uint8).cpu().numpy()
 if int(fields.max())>30:raise ValueError("forbidden field")
 flat=fields.reshape(-1,8).astype(np.uint64);word=sum(flat[:,i]<<(5*i) for i in range(8));packed=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=scale.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes()
 if len(packed)!=CODE_BYTES or len(scales)!=SCALE_BYTES:raise ValueError("wire length")
 return packed,scales

def unpack_fields(packed:bytes)->np.ndarray:
 if len(packed)!=CODE_BYTES:raise ValueError("codes length")
 p=np.frombuffer(packed,np.uint8).reshape(-1,5).astype(np.uint64);word=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;fields=np.stack([(word>>(5*i))&31 for i in range(8)],-1).astype(np.uint8)
 if bool((fields==FORBIDDEN_FIELD).any()):raise ValueError("field31")
 return fields.reshape(-1)

def dequantize_matrix(packed:bytes,scales:bytes,rows:int,columns:int)->torch.Tensor:
 if (rows,columns) not in ((512,2048),(2048,512)) or len(scales)!=SCALE_BYTES:raise ValueError("schema")
 q=torch.from_numpy(unpack_fields(packed).astype(np.int16)-BIAS).reshape(rows,columns//GROUP,GROUP).float();bits=torch.from_numpy(np.frombuffer(scales,dtype='<u2').copy()).to(torch.uint16);s=bits.view(torch.bfloat16).float().reshape(rows,columns//GROUP,1);return (q*s).reshape(rows,columns).to(torch.bfloat16)

def source_key(expert:int,projection:str)->str:
 if not 0<=expert<=512 or projection not in {x[0] for x in PROJECTIONS}:raise ValueError("identity")
 return (f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight" if expert<512 else f"model.layers.0.mlp.shared_expert.{projection}_proj.weight")
