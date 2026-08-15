#!/usr/bin/env python3
"""PH0-R3 wire/checker/oracle/transaction primitives. No device imports."""
from __future__ import annotations
import hashlib, json, math, os, struct, zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; REPORTS=ROOT/'reports/streamq5_moe'
RUN_DIR=ROOT/'reports/runs/streamq5_moe/het_next_l0_ph0r3_single_projection'
SHARD=Path(r'C:/Users/de_do/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a19358a7659bd1f564300250ee189120c49a562f/model-00001-of-00040.safetensors')
D2=ROOT/'reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors'
SOURCE_OFFSET=3_498_051_416; SOURCE_BYTES=2_097_152; SOURCE_SHA='05bd679bceacfd4818103bcfdfe83d17cb288986655598f649a5fe0562d58c9c'
INPUT_OFFSET=155_138_788; INPUT_BYTES=4_096; INPUT_SHA='5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f'
SHARD_BYTES=3_999_619_288; SHARD_SHA='8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a'
D2_BYTES=171_696_126; D2_SHA='f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd'
HEADER=struct.Struct('<4sHHHBBIIH2xIII28s'); HEADER_BYTES=64; CODE_BYTES=655_360; SCALE_BYTES=16_384; PAD_BYTES=4_032; RECORD_BYTES=675_840
ROWS=512; COLS=2048; GROUP=128; COUNTER_BYTES=2_048
CODES_SHA='20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b'; SCALES_SHA='658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8'
DECODED_SHA='9fd43163f4933920168ec9d356db90615a09ecac71198bcc7d3ae373fd995c77'; RECORD_SHA='e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9'
PRISTINE_CRC=1_976_639_022; REQUEST=(0,50,0,ROWS,COLS)

def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def file_digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def canonical(obj)->bytes:return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n'
def read_exact(path:Path,offset:int,size:int)->bytes:
 with path.open('rb') as f:f.seek(offset); data=f.read(size)
 if len(data)!=size:raise EOFError(f'{path}: short range')
 return data
def bf16_to_f32(words):return (np.asarray(words,dtype=np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def f32_to_bf16(values):
 b=np.asarray(values,dtype=np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)

def quantize(source:bytes)->tuple[bytes,bytes,bytes]:
 words=np.frombuffer(source,'<u2'); values=bf16_to_f32(words).reshape(ROWS,COLS)
 blocks=values.reshape(ROWS,COLS//GROUP,GROUP); maximum=np.max(np.abs(blocks),axis=-1,keepdims=True)
 scale=np.where(maximum>0,np.asarray(maximum/np.float32(15),dtype=np.float32),np.float32(1))
 q=np.where(maximum>0,np.clip(np.rint(np.asarray(blocks/scale,dtype=np.float32)),-15,15),0).astype(np.int16)
 fields=(q+15).astype(np.uint64).reshape(-1,8); word=np.bitwise_or.reduce(fields<<(np.arange(8,dtype=np.uint64)*5),axis=1)
 codes=np.stack([(word>>(8*i))&255 for i in range(5)],axis=1).astype(np.uint8).tobytes()
 scale_words=f32_to_bf16(scale.reshape(-1)); scales=scale_words.astype('<u2',copy=False).tobytes()
 decoded=f32_to_bf16(q.reshape(ROWS,-1).astype(np.float32)*bf16_to_f32(scale_words).reshape(ROWS,-1).repeat(GROUP,axis=1)).astype('<u2').tobytes()
 return codes,scales,decoded

def build_record(source:bytes)->tuple[bytes,dict]:
 if digest(source)!=SOURCE_SHA:raise ValueError('source_digest')
 codes,scales,decoded=quantize(source)
 if (digest(codes),digest(scales),digest(decoded))!=(CODES_SHA,SCALES_SHA,DECODED_SHA):raise ValueError('codec_evidence')
 crc=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff
 header=HEADER.pack(b'SQ5M',1,0,50,0,5,ROWS,COLS,GROUP,len(codes),len(scales),crc,bytes(28)); record=header+codes+scales+bytes(PAD_BYTES)
 if crc!=PRISTINE_CRC or digest(record)!=RECORD_SHA:raise ValueError('record_evidence')
 return record,{'source_sha256':SOURCE_SHA,'codes_sha256':digest(codes),'scales_sha256':digest(scales),'decoded_sha256':digest(decoded),'header_sha256':digest(header),'record_sha256':digest(record),'crc32':crc,'q_fields':ROWS*COLS,'record_bytes':len(record)}

def header(record:bytes):return HEADER.unpack(record[:HEADER_BYTES])
def split_record(record:bytes):return record[64:64+CODE_BYTES],record[64+CODE_BYTES:64+CODE_BYTES+SCALE_BYTES]
def unpack_fields(codes:bytes)->np.ndarray:
 p=np.frombuffer(codes,np.uint8).reshape(-1,5).astype(np.uint64); w=p[:,0]|p[:,1]<<8|p[:,2]<<16|p[:,3]<<24|p[:,4]<<32
 return np.stack([(w>>(5*i))&31 for i in range(8)],axis=1).astype(np.uint8).reshape(ROWS,COLS)

class Reject(ValueError):
 def __init__(self,stage):super().__init__(stage);self.stage=stage
def safe_check(record:bytes,input_bytes:bytes,source_sha=SOURCE_SHA)->dict:
 trace=[]
 def gate(ok,stage):
  trace.append(stage)
  if not ok:raise Reject(stage)
 gate(len(record)==RECORD_BYTES,'size')
 try:v=header(record)
 except Exception:raise Reject('structural_header')
 gate((v[0],v[1],v[2],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[12])==(b'SQ5M',1,0,0,5,ROWS,COLS,GROUP,CODE_BYTES,SCALE_BYTES,bytes(28)),'structural_header')
 codes,scales=split_record(record); gate((zlib.crc32(scales,zlib.crc32(codes))&0xffffffff)==v[11],'crc')
 fields=unpack_fields(codes); gate(bool(np.all(fields<=30)),'field_range')
 gate(source_sha==SOURCE_SHA and digest(codes)==CODES_SHA and digest(scales)==SCALES_SHA,'payload_digests')
 gate((v[2],v[3],v[4],v[6],v[7])==REQUEST,'requested_identity')
 gate(len(input_bytes)==INPUT_BYTES and digest(input_bytes)==INPUT_SHA,'input_digest')
 gate(digest(record)==RECORD_SHA,'full_record_digest');trace.append('dispatch')
 return {'trace':trace,'q_min':int(fields.min())-15,'q_max':int(fields.max())-15}

def _rshift_even(n:int,s:int)->int:
 if s<=0:return n<<(-s)
 q,r=divmod(n,1<<s);half=1<<(s-1)
 return q+(r>half or (r==half and (q&1)))
def _finite(bits:int):
 sign=-1 if bits>>31 else 1; exp=(bits>>23)&255; frac=bits&0x7fffff
 if exp==255:raise ValueError('nonfinite')
 if exp==0:return sign*frac,-149
 return sign*((1<<23)|frac),exp-127-23
def _pack_exact(n:int,e:int)->int:
 if n==0:return 0
 sign=0x80000000 if n<0 else 0;n=abs(n);top=n.bit_length()-1+e
 if top>127:return sign|0x7f800000
 if top>=-126:
  shift=n.bit_length()-24;sig=_rshift_even(n,shift)
  if sig==(1<<24):sig>>=1;shift+=1
  unbiased=e+shift+23
  if unbiased>127:return sign|0x7f800000
  return sign|((unbiased+127)<<23)|(sig&0x7fffff)
 frac=_rshift_even(n,-149-e)
 if frac==0:return sign
 if frac>=(1<<23):return sign|(1<<23)
 return sign|frac
def soft_fma_bits(a:int,b:int,c:int)->int:
 an,ae=_finite(a);bn,be=_finite(b);cn,ce=_finite(c);pn,pe=an*bn,ae+be
 e=min(pe,ce);return _pack_exact((pn<<(pe-e))+(cn<<(ce-e)),e)
def soft_add_bits(a:int,b:int)->int:return soft_fma_bits(a,0x3f800000,b)
def bf16_word_to_f32_bits(word:int)->int:return int(word)<<16
def round_f32_bits_to_bf16(bits:int)->int:
 if (bits&0x7f800000)==0x7f800000 and bits&0x7fffff:raise ValueError('nonfinite')
 return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff

def cpu_oracle(record:bytes,input_bytes:bytes)->np.ndarray:
 codes,scales=split_record(record); fields=unpack_fields(codes)
 if np.any(fields>30):raise ValueError('field31')
 q=fields.astype(np.int16)-15; sw=np.frombuffer(scales,'<u2').reshape(ROWS,COLS//GROUP); x=np.frombuffer(input_bytes,'<u2')
 out=np.empty(ROWS,dtype='<u2')
 for row in range(ROWS):
  partial=[[0]*32 for _ in range(8)]
  for lane in range(8):
   for virtual in range(32):
    pack=lane+8*virtual; col=pack*8; acc=0
    scale=bf16_to_f32(np.asarray([sw[row,col//GROUP]],dtype=np.uint16))[0]
    for part in range(8):
     weight_word=int(f32_to_bf16(np.asarray([np.float32(int(q[row,col+part]))*scale]))[0])
     acc=soft_fma_bits(bf16_word_to_f32_bits(weight_word),bf16_word_to_f32_bits(int(x[col+part])),acc)
    partial[lane][virtual]=acc
  for distance in (16,8,4,2,1):
   for lane in range(8):
    for i in range(distance):partial[lane][i]=soft_add_bits(partial[lane][i],partial[lane][i+distance])
  lane=[partial[i][0] for i in range(8)]
  for off in (4,2,1):
   old=lane.copy()
   for i in range(off):lane[i]=soft_add_bits(old[i],old[i+off])
  out[row]=round_f32_bits_to_bf16(lane[0])
 return out

def controls(record:bytes,input_bytes:bytes)->list[dict]:
 rows=[]
 def attempt(name,r,x,expect):
  try:safe_check(r,x);observed='accepted'
  except Reject as e:observed=e.stage
  rows.append({'name':name,'expected':expect,'observed':observed,'pass':observed==expect,'record_sha256':digest(r),'input_sha256':digest(x)})
 attempt('truncation',record[:-1],input_bytes,'size')
 v=list(header(record));v[3]=51;wrong=HEADER.pack(*v)+record[64:];attempt('wrong_identity',wrong,input_bytes,'requested_identity')
 c,s=split_record(record);m=bytearray(c);m[5]^=1;attempt('stale_crc',record[:64]+bytes(m)+s+record[-PAD_BYTES:],input_bytes,'crc')
 def rebuilt(codes,scales,expert=50):
  crc=zlib.crc32(scales,zlib.crc32(codes))&0xffffffff;return HEADER.pack(b'SQ5M',1,0,expert,0,5,ROWS,COLS,GROUP,CODE_BYTES,SCALE_BYTES,crc,bytes(28))+codes+scales+bytes(PAD_BYTES)
 attempt('valid_crc_code_digest',rebuilt(bytes(m),s),input_bytes,'payload_digests')
 sm=bytearray(s);sm[0]^=1;attempt('wrong_scale',rebuilt(c,bytes(sm)),input_bytes,'payload_digests')
 xm=bytearray(input_bytes);xm[0]^=1;attempt('wrong_input',record,bytes(xm),'input_digest')
 fm=bytearray(c);w=int.from_bytes(fm[:5],'little');fm[:5]=((w&~31)|31).to_bytes(5,'little');attempt('field31',rebuilt(bytes(fm),s),input_bytes,'field_range')
 fields=unpack_fields(c);flat=fields.reshape(-1);idx=int(np.flatnonzero(flat!=15)[0]);stored=int(flat[idx]);step=stored-1 if stored>15 else stored+1
 qm=bytearray(c);pack=idx//8;slot=idx%8;w=int.from_bytes(qm[pack*5:pack*5+5],'little');w=(w&~(31<<(5*slot)))|(step<<(5*slot));qm[pack*5:pack*5+5]=w.to_bytes(5,'little');qrecord=rebuilt(bytes(qm),s);attempt('q_sensitivity_safe',qrecord,input_bytes,'payload_digests')
 activation=np.zeros(COLS,dtype='<u2');activation[0]=0x3b80
 original=np.zeros(ROWS,dtype='<u2');mutated=np.zeros(ROWS,dtype='<u2')
 scale0=bf16_to_f32(np.asarray([np.frombuffer(s,'<u2')[0]],dtype=np.uint16))[0]
 q0,q1=stored-15,step-15;ow=int(f32_to_bf16(np.asarray([np.float32(q0)*scale0]))[0]);mw=int(f32_to_bf16(np.asarray([np.float32(q1)*scale0]))[0])
 original[0]=round_f32_bits_to_bf16(soft_fma_bits(ow<<16,0x3b800000,0));mutated[0]=round_f32_bits_to_bf16(soft_fma_bits(mw<<16,0x3b800000,0))
 rows.append({'name':'q_sensitivity_witness','expected':'one_word_3894_to_3882','observed':'one_word_3894_to_3882' if (original[0],mutated[0],np.count_nonzero(original!=mutated))==(0x3894,0x3882,1) else 'mismatch','pass':(digest(activation.tobytes()),digest(original.tobytes()),digest(mutated.tobytes()))==('2498a04e393ec5eb0ec88b7f098523dd5f3a1cbaf9803fa7ace4b4776c17f561','ca913e50693d83329869fb61dabb75467df7091d39e9a5dd9e17e8480bbeb9f6','1868bd78f7059362bed974138ae89c4efa7b930fdf3d07db11db6cd94677ee23')})
 return rows

def fsync_file(path:Path):
 with path.open('r+b' if os.name=='nt' else 'rb') as f:os.fsync(f.fileno())
def write_atomic_new(path:Path,data:bytes):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.inprogress')
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fsync_file(path)
 try:
  fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
 except OSError:pass
