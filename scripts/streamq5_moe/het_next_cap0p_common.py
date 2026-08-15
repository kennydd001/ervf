#!/usr/bin/env python3
"""Pure CAP0P fixtures and child line protocol."""
import hashlib,json,struct,time
SEED=0x4845544E45585430;COUNT=1024
def words():
 x=[SEED&0xffffffff]
 for _ in range(1,COUNT):x.append((1664525*x[-1]+1013904223)&0xffffffff)
 return x
def expected(device,x):
 if device=='intel':return [(((((v^0xa5a5a5a5)<<7)|((v^0xa5a5a5a5)>>25))+0x3c6ef372)&0xffffffff) for v in x]
 return [((((((v+0x9e3779b9)&0xffffffff)>>11)|(((v+0x9e3779b9)&0xffffffff)<<21))&0xffffffff)^0xc3c3c3c3 for v in x]
def digest(x):return hashlib.sha256(struct.pack('<1024I',*x)).hexdigest()
def emit(obj):print(json.dumps(obj,sort_keys=True,separators=(',',':')),flush=True)
def command():
 line=input().strip().split()
 if not line:raise EOFError('coordinator_pipe_closed')
 return line
def now():return time.perf_counter_ns()
