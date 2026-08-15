#!/usr/bin/env python3
"""Frozen natural inputs for T0Q5-R1; no model output/route read or filtering."""
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
SEED="PORT80B-T0Q5R1-FRESH-DISJOINT-2026-08-13-v1";REV="a19358a7659bd1f564300250ee189120c49a562f"
CANDIDATES={
 "code_review":["Review a concurrent queue implementation for lost wakeups, explain the race precisely, and propose a minimal robust correction."],
 "biology":["Describe how a cell repairs double strand DNA breaks, contrasting homologous recombination with end joining and their tradeoffs."],
 "commercial":["Draft a concise explanation of liability caps, exclusions, and service credits for a negotiated cloud hosting agreement."],
 "dutch_infra":["Leg uit hoe een beweegbare brug veilig wordt aangestuurd bij harde wind, scheepvaart en druk wegverkeer."],
}
def generate():
 snap=Path.home()/f'.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/{REV}';tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False);rows=[]
 for domain,items in CANDIDATES.items():
  index=int.from_bytes(hashlib.sha256(f'{SEED}:{domain}'.encode()).digest()[:8],'little')%len(items);text=items[index];ids=tok(text,add_special_tokens=False)['input_ids'][:16];packed=b''.join(int(x).to_bytes(4,'little') for x in ids);rows.append({'domain':domain,'candidate_index':index,'utf8_text':text,'token_ids':ids,'token_ids_le_u32_sha256':hashlib.sha256(packed).hexdigest()})
 return {'kind':'port80b_t0q5r1_prompt_lock','algorithm':'ordered domains; SHA256(seed:domain) little-u64 modulo candidate count; first 16 tokens; no rejection/filtering','seed':SEED,'revision':REV,'no_output_dependent_filtering':True,'prompts':rows}
if __name__=='__main__':print(json.dumps(generate(),ensure_ascii=False,sort_keys=True,separators=(',',':')))
