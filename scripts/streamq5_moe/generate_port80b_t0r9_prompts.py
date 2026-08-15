#!/usr/bin/env python3
"""Deterministic R8 prompt source. No model output or route is read or filtered."""
import argparse, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer
SEED="PORT80B-T0R9-FRESH-PROMPTS-2026-08-13-v1"
REV="a19358a7659bd1f564300250ee189120c49a562f"
CANDIDATES={
 "code":["Implement a stable merge sort for signed integers and explain its invariants, complexity, and edge cases carefully."],
 "science":["Explain why coastal temperatures vary less than inland temperatures across seasons, using heat capacity and circulation."],
 "legal":["Summarize the difference between a warranty and an indemnity in plain language for a small software contract."],
 "dutch":["Beschrijf hoe een warmtepomp warmte verplaatst zonder zelf warmte te produceren, met een praktisch voorbeeld voor thuis."],
}
def generate():
 snap=Path.home()/f".cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/{REV}"
 tokenizer=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False);rows=[]
 for domain,items in CANDIDATES.items():
  index=int.from_bytes(hashlib.sha256(f"{SEED}:{domain}".encode()).digest()[:8],"little")%len(items);text=items[index]
  ids=tokenizer(text,add_special_tokens=False)["input_ids"][:16]
  packed=b"".join(int(x).to_bytes(4,"little") for x in ids)
  rows.append({"domain":domain,"candidate_index":index,"utf8_text":text,"token_ids":ids,"token_ids_le_u32_sha256":hashlib.sha256(packed).hexdigest()})
 return {"algorithm":"ordered_domains; SHA256(seed:domain) little-u64 modulo declared candidate count; tokenize; take first 16; no rejection/filtering","seed":SEED,"revision":REV,"no_output_dependent_filtering":True,"prompts":rows}
if __name__=="__main__":
 argparse.ArgumentParser().parse_args();print(json.dumps(generate(),ensure_ascii=False,indent=2))
