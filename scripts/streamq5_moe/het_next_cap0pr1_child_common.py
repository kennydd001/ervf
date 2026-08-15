#!/usr/bin/env python3
import argparse,json,os,sys
from het_next_cap0pr1_protocol import frame
def args():
 p=argparse.ArgumentParser();p.add_argument('--lp',type=int,required=True);p.add_argument('--nonce',required=True);p.add_argument('--role',required=True);return p.parse_args()
def send(a,seq,kind,payload):sys.stdout.buffer.write(frame(a.nonce,a.role,seq,kind,payload));sys.stdout.buffer.flush()
def receive(expected):
 raw=sys.stdin.buffer.readline()
 if not raw:raise EOFError('control_pipe_closed')
 row=json.loads(raw)
 if row!=expected:raise RuntimeError(f'control_schema:{row!r}')
