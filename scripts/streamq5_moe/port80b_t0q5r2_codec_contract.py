#!/usr/bin/env python3
"""Exact R2 SQ5M wire/codec contract; no checkpoint or experiment execution."""
from __future__ import annotations
import hashlib,struct,zlib
import numpy as np
import torch
REVISION="a19358a7659bd1f564300250ee189120c49a562f"
HEADER_FORMAT="<4sHHHBBIIH2xIII28s";HEADER_BYTES=struct.calcsize(HEADER_FORMAT)
GROUP=128;CODE_BYTES=655_360;SCALE_BYTES=16_384;PADDING_BYTES=4_032;MATRIX_BYTES=675_840;EXPERT_BYTES=2_027_520;EXPERTS=512;RECORDS=513;BANK_BYTES=1_040_117_760
SHAPES=((512,2048),(512,2048),(2048,512));NAMES=("gate","up","down")
def tensor_bytes(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def quantize(value):
 if value.dtype!=torch.bfloat16 or tuple(value.shape) not in set(SHAPES):raise ValueError('source schema')
 rows,cols=value.shape;w=value.float().reshape(rows,cols//GROUP,GROUP);mx=w.abs().amax(-1,keepdim=True);scale=torch.where(mx>0,mx/15,torch.ones_like(mx));q=torch.where(mx>0,torch.round(w/scale).clamp(-15,15),torch.zeros_like(w)).to(torch.int8);fields=(q.to(torch.int16)+15).cpu().numpy()
 if fields.min()<0 or fields.max()>30:raise ValueError('field31')
 f=fields.astype(np.uint64).reshape(-1,8);word=np.bitwise_or.reduce(f<<(np.arange(8,dtype=np.uint64)*5),axis=-1);codes=np.stack([(word>>(8*i))&255 for i in range(5)],-1).astype(np.uint8).tobytes();scales=scale.squeeze(-1).to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes()
 if len(codes)!=CODE_BYTES or len(scales)!=SCALE_BYTES:raise ValueError('payload lengths')
 return codes,scales
def make_record(value,expert,projection):
 if not 0<=expert<=512 or not 0<=projection<=2:raise ValueError('identity')
 rows,cols=value.shape;codes,scales=quantize(value);crc=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff;header=struct.pack(HEADER_FORMAT,b'SQ5M',1,0,expert,projection,5,rows,cols,GROUP,len(codes),len(scales),crc,bytes(28));padding=bytes(PADDING_BYTES);record=header+codes+scales+padding
 if HEADER_BYTES!=64 or len(record)!=MATRIX_BYTES:raise ValueError('record size')
 meta={'revision':REVISION,'layer':0,'expert':expert,'shared':expert==512,'projection':projection,'projection_name':NAMES[projection],'rows':rows,'columns':cols,'source_dtype':'torch.bfloat16','source_sha256':hashlib.sha256(tensor_bytes(value)).hexdigest(),'header_sha256':hashlib.sha256(header).hexdigest(),'codes_sha256':hashlib.sha256(codes).hexdigest(),'scales_sha256':hashlib.sha256(scales).hexdigest(),'padding_sha256':hashlib.sha256(padding).hexdigest(),'record_sha256':hashlib.sha256(record).hexdigest(),'crc32':crc,'header_bytes':64,'code_bytes':CODE_BYTES,'scale_bytes':SCALE_BYTES,'padding_bytes':PADDING_BYTES,'record_bytes':MATRIX_BYTES}
 return record,meta
def parse_record(record,expected_expert,expected_projection):
 if len(record)!=MATRIX_BYTES:raise ValueError('record size')
 fields=struct.unpack(HEADER_FORMAT,record[:64]);magic,version,layer,expert,projection,bits,rows,cols,group,cb,sb,crc,reserved=fields
 codes=record[64:64+cb];scales=record[64+cb:64+cb+sb];padding=record[64+cb+sb:]
 if (magic,version,layer,expert,projection,bits,group,cb,sb)!=(b'SQ5M',1,0,expected_expert,expected_projection,5,128,CODE_BYTES,SCALE_BYTES):raise ValueError('header identity')
 if reserved!=bytes(28) or padding!=bytes(PADDING_BYTES) or (rows,cols)!=SHAPES[projection]:raise ValueError('schema/padding')
 if zlib.crc32(scales,zlib.crc32(codes))&0xffffffff!=crc:raise ValueError('crc')
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);word=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32;decoded=np.stack([(word>>(5*i))&31 for i in range(8)],-1)
 if bool((decoded==31).any()) or int(decoded.max())>30:raise ValueError('reserved code')
 return {'expert':expert,'projection':projection,'rows':rows,'columns':cols,'crc32':crc,'record_sha256':hashlib.sha256(record).hexdigest()}
