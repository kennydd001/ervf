from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import DOMAINS, MAX_CONTEXT, Runtime


MODEL = ROOT / "models/qwen3-30b-a3b-base"
R = ROOT / "reports/streamq5_moe"
INPUT_LOCK = R / "p6a_end_to_end_input_lock.json"
VERIFICATION = R / "p6b_end_to_end_verification.json"
TEST_RESULT = R / "p6b_strict_end_to_end_test.json"
SESSION_LOG = R / "p6b_server_sessions.jsonl"
MODEL_ID = "streamq5-qwen3-30b-a3b-q5q8"


INDEX_HTML = r'''<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>STREAMQ5 · Qwen3-30B-A3B</title>
  <style>
    :root{color-scheme:dark;--bg:#090b10;--panel:#121722;--line:#273043;--text:#eef2ff;--muted:#9aa7bd;--accent:#6ee7b7;--accent2:#60a5fa}
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 10% 0,#182235 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
    main{width:min(920px,calc(100% - 28px));margin:34px auto}.top{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:18px}
    h1{font-size:clamp(26px,5vw,42px);line-height:1.05;margin:0}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;text-transform:uppercase;font-size:12px}
    .badge{border:1px solid #285b4b;background:#102820;color:var(--accent);padding:7px 11px;border-radius:999px;white-space:nowrap}
    .card{background:rgba(18,23,34,.94);border:1px solid var(--line);border-radius:17px;padding:18px;box-shadow:0 20px 60px #0007}
    textarea{width:100%;min-height:150px;resize:vertical;border:1px solid #344057;border-radius:12px;padding:14px;background:#0b1018;color:var(--text);font:inherit;outline:none}
    textarea:focus{border-color:var(--accent2)} .controls{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;margin-top:12px;align-items:end}
    label{display:grid;gap:5px;color:var(--muted);font-size:12px} select,input{height:40px;border:1px solid #344057;border-radius:9px;background:#0b1018;color:var(--text);padding:0 10px}
    button{height:42px;border:0;border-radius:10px;padding:0 18px;font-weight:750;background:linear-gradient(135deg,var(--accent),#34d399);color:#052016;cursor:pointer}
    button:disabled{opacity:.45;cursor:wait}.result{white-space:pre-wrap;min-height:90px;margin-top:16px;padding:15px;border-radius:12px;background:#0b1018;border:1px solid #222c3d}
    .metrics{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;color:var(--muted);font-size:12px}.metrics span{border:1px solid #293348;border-radius:999px;padding:5px 9px}
    .note{color:var(--muted);margin:10px 1px 0;font-size:12px}@media(max-width:700px){.controls{grid-template-columns:1fr 1fr}.top{align-items:start;flex-direction:column}}
  </style>
</head>
<body><main>
  <div class="top"><div><div class="eyebrow">Lokale fysieke runtime</div><h1>STREAMQ5 · 30B MoE</h1></div><div class="badge" id="status">● model gereed</div></div>
  <section class="card">
    <textarea id="prompt">Kunstmatige intelligentie wordt efficiënter wanneer</textarea>
    <div class="controls">
      <label>Modus<select id="mode"><option value="raw">Ruwe completion (aanbevolen)</option><option value="chat">Chat-template (experimenteel)</option></select></label>
      <label>Domeincache<select id="domain"><option>general</option><option>code</option><option>math</option><option>multilingual</option><option>instruction</option></select></label>
      <label>Max. nieuwe tokens<input id="tokens" type="number" min="1" max="512" value="128"></label>
      <button id="run">Genereren</button>
    </div>
    <div class="result" id="result">Het antwoord verschijnt hier.</div>
    <div class="metrics" id="metrics"></div>
    <p class="note">Dit is het Qwen3-30B-A3B Base-checkpoint: ruwe tekstaanvulling werkt het best; chat/instructiegedrag is niet gegarandeerd. Batch 1 · maximaal 4.096 contexttokens · één generatie tegelijk.</p>
  </section>
</main><script>
const run=document.querySelector('#run'), result=document.querySelector('#result'), metrics=document.querySelector('#metrics'), status=document.querySelector('#status');
run.onclick=async()=>{run.disabled=true;result.textContent='Model genereert…';metrics.textContent='';status.textContent='● bezig';const started=performance.now();
try{const response=await fetch('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:document.querySelector('#prompt').value,mode:document.querySelector('#mode').value,domain:document.querySelector('#domain').value,max_new_tokens:Number(document.querySelector('#tokens').value)})});
const data=await response.json();if(!response.ok)throw new Error(data.error||'Onbekende fout');result.textContent=data.text||'(lege uitvoer)';metrics.innerHTML=`<span>${data.generated_tokens} tokens</span><span>${data.tokens_per_second.toFixed(2)} tok/s</span><span>mean ${data.mean_ms.toFixed(1)} ms</span><span>p95 ${data.p95_ms.toFixed(1)} ms</span><span>${data.finish_reason}</span><span>${((performance.now()-started)/1000).toFixed(1)} s totaal</span>`;status.textContent='● model gereed';}
catch(error){result.textContent='Fout: '+error.message;status.textContent='● fout';}finally{run.disabled=false;}};
</script></body></html>'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(np.percentile(array, 95))


class ModelService:
    def __init__(self) -> None:
        verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
        test = json.loads(TEST_RESULT.read_text(encoding="utf-8"))
        if verification["status"] != "p6b_end_to_end_verification_pass":
            raise RuntimeError("verified P6B result required")
        if test["status"] != "p6b_strict_end_to_end_eureka_pass":
            raise RuntimeError("P6B Eureka pass required")
        self.lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.runtime = Runtime(self.lock)
        self.mutex = threading.Lock()
        self.active_domain: str | None = None
        self.started_utc = datetime.now(timezone.utc).isoformat()

    def prepare_prompt(self, payload: dict[str, Any]) -> tuple[str, list[int]]:
        mode = str(payload.get("mode", "chat"))
        if "messages" in payload:
            messages = payload["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            if mode == "chat":
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            elif mode == "raw":
                text = prompt
            else:
                raise ValueError("mode must be chat or raw")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("prompt tokenized to an empty sequence")
        return text, ids

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        domain = str(payload.get("domain", "general"))
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {', '.join(DOMAINS)}")
        max_new = int(payload.get("max_new_tokens", payload.get("max_tokens", 128)))
        if not 1 <= max_new <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        prompt_text, prompt_ids = self.prepare_prompt(payload)
        if len(prompt_ids) + max_new > MAX_CONTEXT:
            raise ValueError(f"prompt ({len(prompt_ids)}) + generation ({max_new}) exceeds {MAX_CONTEXT} tokens")

        request_id = uuid.uuid4().hex
        request_started = time.perf_counter_ns()
        with self.mutex:
            queue_ms = (time.perf_counter_ns() - request_started) / 1e6
            cache_init_ms = 0.0
            if self.active_domain != domain:
                cache_init_ms = self.runtime.activate_domain(domain)
                self.active_domain = domain
            self.runtime.reset_context()
            prefill_ms: list[float] = []
            for position, token in enumerate(prompt_ids[:-1]):
                started = time.perf_counter_ns()
                self.runtime.decode(int(token), position, -1)
                prefill_ms.append((time.perf_counter_ns() - started) / 1e6)

            current = int(prompt_ids[-1])
            position = len(prompt_ids) - 1
            generated: list[int] = []
            decode_ms: list[float] = []
            misses: list[int] = []
            finish_reason = "length"
            stop_ids = {int(self.tokenizer.eos_token_id)} if self.tokenizer.eos_token_id is not None else set()
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end, int) and im_end >= 0:
                stop_ids.add(im_end)
            for _step in range(max_new):
                started = time.perf_counter_ns()
                row = self.runtime.decode(current, position, -1)
                decode_ms.append((time.perf_counter_ns() - started) / 1e6)
                next_token = int(row["prediction"])
                misses.append(int(row["misses"]))
                if next_token in stop_ids:
                    finish_reason = "stop"
                    break
                generated.append(next_token)
                current = next_token
                position += 1

        mean_ms, p95_ms = stats(decode_ms)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        total_ms = (time.perf_counter_ns() - request_started) / 1e6
        response = {
            "id": request_id,
            "model": MODEL_ID,
            "text": text,
            "generated_ids": generated,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "finish_reason": finish_reason,
            "domain": domain,
            "queue_ms": queue_ms,
            "cache_initialization_ms": cache_init_ms,
            "prefill_ms": float(sum(prefill_ms)),
            "decode_ms": float(sum(decode_ms)),
            "total_ms": total_ms,
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "tokens_per_second": 1000.0 / mean_ms,
            "expert_misses": int(sum(misses)),
        }
        log_row = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model": MODEL_ID,
            "domain": domain,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "finish_reason": finish_reason,
            "timing": {key: response[key] for key in ("queue_ms", "cache_initialization_ms", "prefill_ms", "decode_ms", "total_ms", "mean_ms", "p95_ms", "tokens_per_second")},
            "expert_misses": response["expert_misses"],
            "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")),
            "generated_ids_sha256": sha256_bytes(np.asarray(generated, dtype=np.int32).tobytes()),
        }
        with SESSION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row, ensure_ascii=False, sort_keys=True) + "\n")
        return response

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": MODEL_ID,
            "verified": True,
            "started_utc": self.started_utc,
            "max_context_tokens": MAX_CONTEXT,
            "domains": list(DOMAINS),
            "physical": self.runtime.physical(),
        }


SERVICE: ModelService


class Handler(BaseHTTPRequestHandler):
    server_version = "STREAMQ5/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Any, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("invalid request body length")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self.send_json(SERVICE.health())
        elif self.path == "/v1/models":
            self.send_json({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/api/generate":
                self.send_json(SERVICE.generate(payload))
                return
            if self.path == "/v1/completions":
                result = SERVICE.generate({**payload, "mode": "raw"})
                self.send_json({
                    "id": f"cmpl-{result['id']}", "object": "text_completion", "created": int(time.time()),
                    "model": MODEL_ID, "choices": [{"index": 0, "text": result["text"], "finish_reason": result["finish_reason"]}],
                    "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["generated_tokens"], "total_tokens": result["prompt_tokens"] + result["generated_tokens"]},
                    "streamq5_metrics": result,
                })
                return
            if self.path == "/v1/chat/completions":
                if payload.get("stream"):
                    raise ValueError("stream=true is not yet supported")
                result = SERVICE.generate(payload)
                self.send_json({
                    "id": f"chatcmpl-{result['id']}", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL_ID, "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]}, "finish_reason": result["finish_reason"]}],
                    "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["generated_tokens"], "total_tokens": result["prompt_tokens"] + result["generated_tokens"]},
                    "streamq5_metrics": result,
                })
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            print(f"request failed: {error!r}", file=sys.stderr, flush=True)
            self.send_json({"error": f"generation failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("this unauthenticated test server may only bind to localhost")
    global SERVICE
    print("Loading verified P6B runtime; this pins the banks and may take about 20 seconds...", flush=True)
    SERVICE = ModelService()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{args.port}", "model": MODEL_ID}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
