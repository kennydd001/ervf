#!/usr/bin/env python3
"""CPU-only R3 package reader, codec, safe controls and exact contracts."""
from __future__ import annotations
import hashlib,json,struct,zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];REPORTS=ROOT/'reports/streamq5_moe';CPU=REPORTS/'het_next_l0_ph1_cpu_freeze_r2';COMPILE=REPORTS/'het_next_l0_ph1_intel_compile_r2a'
SHARD=Path(r'C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors');D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
INPUT=(155138788,4096,'5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f');LUT_SHA='a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed';HEADER=struct.Struct('<4sHHHBBIIH2xIII28s')
SPECS=(('gate',0,(512,2048),(3498051416,3500148568),'05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c','20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b','658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8','e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9'),('up',1,(512,2048),(3500148568,3502245720),'4b36f661a351aaf907be1e041743833bc7a0564e07a6c140917ef1c8d69e4c0d','6b2a3f124c3bc42d584b2816b063801d63244bd2a9e59cb00a32e339591e25cb','c275fd13db6ea41ab8af1563a32a8de188e5fa488f91a6c7c939c4d3ca80a9f9','6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb'),('down',2,(2048,512),(3495954264,3498051416),'bdf53c222b88c66b5845fd548ae984c20959231150b2fd34ddccf10d1777e479','3d8782d588d507fea2a2c51ef8a3ea18ce6795d72b4be047b0c123652d77a703','a3cd1a7c827dd9cb64925ad15299adbc18d74e592a1414504c3015e29854977e','bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157'))
def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):return sha(Path(p).read_bytes())
def rr(p,o,n):
 with Path(p).open('rb') as h:h.seek(o);b=h.read(n)
 if len(b)!=n:raise EOFError(p)
 return b
def b2f(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def f2b(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def record(spec,src):
 name,ordinal,(r,c),_,ss,cs,scs,rs=spec
 if sha(src)!=ss:raise ValueError('source_digest')
 v=b2f(np.frombuffer(src,'<u2')).reshape(r,c);bl=v.reshape(r,c//128,128);m=np.max(np.abs(bl),axis=-1,keepdims=True);scale=np.where(m>0,np.asarray(m/np.float32(15),np.float32),np.float32(1));q=np.where(m>0,np.clip(np.rint(np.asarray(bl/scale,np.float32)),-15,15),0).astype(np.int16);fld=(q+15).astype(np.uint64).reshape(-1,8);words=np.bitwise_or.reduce(fld<<(np.arange(8,dtype=np.uint64)*5),axis=1);codes=np.stack([(words>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes();sc=f2b(scale.reshape(-1)).astype('<u2').tobytes();crc=zlib.crc32(sc,zlib.crc32(codes))&0xffffffff;head=HEADER.pack(b'SQ5M',1,0,50,ordinal,5,r,c,128,len(codes),len(sc),crc,bytes(28));rec=head+codes+sc+bytes(4032)
 if sha(codes)!=cs or sha(sc)!=scs or sha(rec)!=rs or len(rec)!=675840:raise ValueError('record_digest')
 return rec
def check_record(data,spec,input_digest):
 if len(data)!=675840:return 'size'
 fields=HEADER.unpack(data[:64]);name,ordinal,shape,*_=spec
 if fields[:9]!=(b'SQ5M',1,0,50,ordinal,5,*shape,128):return 'identity'
 codes,sc=data[64:655424],data[655424:671808]
 if (zlib.crc32(sc,zlib.crc32(codes))&0xffffffff)!=fields[11]:return 'crc'
 vals=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64);packed=sum(vals[:,i]<<(8*i) for i in range(5))
 if np.any(((packed[:,None]>>(5*np.arange(8,dtype=np.uint64)))&31)==31):return 'field31'
 if sha(codes)!=spec[5] or sha(sc)!=spec[6]:return 'canonical_digest'
 if sha(data)!=spec[7]:return 'record_digest'
 if input_digest!=INPUT[2]:return 'input_digest'
 return 'pass'
def controls(records,input_bytes,lut):
 rows=[]
 for spec in SPECS:
  name=spec[0];base=records[name]
  cases=[];cases.append(('truncation',base[:-1],INPUT[2],'size'));h=bytearray(base);x=list(HEADER.unpack(h[:64]));x[4]=(x[4]+1)%3;h[:64]=HEADER.pack(*x);cases.append(('wrong_projection',bytes(h),INPUT[2],'identity'));h=bytearray(base);h[64]^=1;cases.append(('stale_crc',bytes(h),INPUT[2],'crc'))
  h=bytearray(base);word=int.from_bytes(h[64:69],'little');selected=None
  for slot in range(8):
   field=(word>>(5*slot))&31
   if field!=15:
    replacement=field-1 if field>15 else field+1
    if replacement<=30:selected=(slot,replacement);break
  if selected is None:raise RuntimeError('no_safe_code_mutation')
  slot,replacement=selected;word=(word&~(31<<(5*slot)))|(replacement<<(5*slot));h[64:69]=word.to_bytes(5,'little');x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('code_mutation',bytes(h),INPUT[2],'canonical_digest'))
  h=bytearray(base);h[655424]^=1;x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('scale_mutation',bytes(h),INPUT[2],'canonical_digest'));h=bytearray(base);five=int.from_bytes(h[64:69],'little');five=(five&~31)|31;h[64:69]=five.to_bytes(5,'little');x=list(HEADER.unpack(h[:64]));x[11]=zlib.crc32(h[655424:671808],zlib.crc32(h[64:655424]))&0xffffffff;h[:64]=HEADER.pack(*x);cases.append(('field31',bytes(h),INPUT[2],'field31'));cases.append(('wrong_input',base,'0'*64,'input_digest'))
  for control,data,digest,expected in cases:
   observed=check_record(data,spec,digest);rows.append({'record':name,'control':control,'expected':expected,'observed':observed,'pass':observed==expected,'predevice_counts':{'opencl_load':0,'context':0,'program':0,'kernel':0,'allocation':0,'launch':0}})
 wrong=bytearray(lut);wrong[0]^=1;observed='lut_digest' if sha(wrong)!=LUT_SHA else 'pass'
 rows.append({'record':'global','control':'wrong_lut_digest','expected':'lut_digest','observed':observed,'pass':observed=='lut_digest','presented_sha256':sha(wrong),'expected_sha256':LUT_SHA,'predevice_counts':{'opencl_load':0,'context':0,'program':0,'kernel':0,'allocation':0,'launch':0}})
 if len(rows)!=22 or not all(x['pass'] for x in rows):raise RuntimeError('controls')
 return rows
def package():
 inp=rr(D2,*INPUT[:2]);lut=(CPU/'bf16_silu_lut.bin').read_bytes()
 if sha(inp)!=INPUT[2] or sha(lut)!=LUT_SHA:raise ValueError('input_lut')
 records={}
 for spec in SPECS:records[spec[0]]=record(spec,rr(SHARD,spec[3][0],spec[3][1]-spec[3][0]))
 return records,inp,lut,controls(records,inp,lut)


