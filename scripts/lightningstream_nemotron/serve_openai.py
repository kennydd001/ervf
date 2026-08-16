"""OpenAI-compatible HTTP server in front of the Lightning runtime.

Purpose: stop re-inventing a chat UI. Any client that speaks the OpenAI API --
llama.cpp's own web UI, Open WebUI, LM Studio, Continue, a plain `curl` -- can
point at this and drive the real V18 stack. That gives us a chat surface without
porting anything into llama.cpp, which (see agents/LLAMA_CPP_INTEROP.md) cannot
currently run this architecture at all and would not carry our kernels anyway.

Runs the **V18 stack**, i.e. the 51.0 tok/s record path, not the bare runtime:

    selective ERVF  +  batched MoE (V6)  +  H-SCALE + B3 overlap (combined)

`chat_lightning.py` historically built only the plain graph runtime, so chatting
measured ~35 tok/s while the record was 51. Both entry points now share
`build_v18_runtime()` so a chat session and a benchmark exercise the same code.

Stdlib only -- no new dependency in .venv-nemotron.

    .venv-nemotron/Scripts/python.exe scripts/lightningstream_nemotron/serve_openai.py
    curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"lightning","messages":[{"role":"user","content":"hi"}],"stream":true}'

One GPU, one runtime, one sequence: requests are serialised behind a lock and
each one resets model state. There are no KV slots and no parallel decode --
that is a deliberate limit of this server, not of the API surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pro_research"))

MODEL = REPO / "models" / "nemotron_3_5_lightning_v35"
EOS_IDS = {2, 11}          # generation_config.json

_LOCK = threading.Lock()
_RT = None
_TOK = None
_STACK = "v18"

# Served at "/" so the whole thing is one command and a browser -- no client to
# install. Any OpenAI-compatible UI still works against /v1 if you prefer one.
CHAT_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron Lightning</title><style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0da;--card:#fff;--acc:#b4522b}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#eceae5;--mut:#9a978f;--line:#33312d;--card:#232220;--acc:#e08050}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
display:flex;flex-direction:column;height:100dvh}
header{padding:.7rem 1rem;border-bottom:1px solid var(--line);display:flex;
gap:.75rem;align-items:baseline;flex-wrap:wrap}
h1{font-size:1rem;margin:0;font-weight:600}
.meta{color:var(--mut);font-size:.8rem}
#log{flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:.9rem}
.msg{max-width:min(46rem,100%);padding:.7rem .9rem;border-radius:.7rem;
white-space:pre-wrap;word-wrap:break-word}
.user{align-self:flex-end;background:var(--acc);color:#fff}
.bot{align-self:flex-start;background:var(--card);border:1px solid var(--line)}
.tps{font-size:.75rem;color:var(--mut);margin-top:.35rem}
footer{border-top:1px solid var(--line);padding:.75rem 1rem;display:flex;gap:.5rem}
textarea{flex:1;resize:none;padding:.6rem .7rem;border-radius:.6rem;
border:1px solid var(--line);background:var(--card);color:var(--fg);
font:inherit;min-height:2.7rem;max-height:9rem}
button{padding:.6rem 1.1rem;border:0;border-radius:.6rem;background:var(--acc);
color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
</style></head><body>
<header><h1>Nemotron 3.5 Lightning</h1>
<span class="meta" id="hd">connecting...</span>
<span class="meta" style="margin-left:auto"><button id="clr"
style="background:none;color:var(--mut);padding:.2rem .5rem;font-weight:400">clear</button></span>
</header>
<div id="log"></div>
<footer><textarea id="in" placeholder="Message... (Enter to send, Shift+Enter for newline)"></textarea>
<button id="go">Send</button></footer>
<script>
const log=document.getElementById('log'),inp=document.getElementById('in'),
go=document.getElementById('go'),hd=document.getElementById('hd');
let msgs=[],busy=false;
fetch('/v1/models').then(r=>r.json()).then(d=>{
  hd.textContent='stack '+(d.data[0].stack||'?')+' - local, one sequence';}).catch(()=>hd.textContent='offline');
function add(cls,txt){const d=document.createElement('div');d.className='msg '+cls;
  d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
document.getElementById('clr').onclick=()=>{msgs=[];log.innerHTML='';};
async function send(){
  const text=inp.value.trim(); if(!text||busy) return;
  busy=true; go.disabled=true; inp.value='';
  add('user',text); msgs.push({role:'user',content:text});
  const box=add('bot',''); const tps=document.createElement('div');
  tps.className='tps'; box.after(tps);
  let acc='';
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'lightning',messages:msgs,stream:true,max_tokens:512})});
    const rd=r.body.getReader(),dec=new TextDecoder(); let buf='';
    for(;;){const{done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\n\n'); buf=parts.pop();
      for(const p of parts){
        const line=p.trim(); if(!line.startsWith('data:'))continue;
        const data=line.slice(5).trim(); if(data==='[DONE]')continue;
        let j; try{j=JSON.parse(data)}catch(e){continue}
        const c=j.choices&&j.choices[0];
        if(c&&c.delta&&c.delta.content){acc+=c.delta.content;box.textContent=acc;
          log.scrollTop=log.scrollHeight;}
        if(j.x_tokens_per_second) tps.textContent=j.x_tokens_per_second+' tok/s';
      }}
    msgs.push({role:'assistant',content:acc});
  }catch(e){box.textContent=acc+'\n\n[error: '+e+']';}
  busy=false; go.disabled=false; inp.focus();
}
go.onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
inp.focus();
</script></body></html>"""


def build_v18_runtime(capacity: int = 72, stack: str = "v18"):
    """The record path. `stack='v6'` stops before H-SCALE/B3 for comparison."""
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    from down_proj_batch_kernels import DownProjBatchKernels
    from ervf_dense import DenseERVF
    from layer_capacity import apply_nonuniform_capacity
    from moe_dev_batched import install_batched_moe_dev
    from selective_ervf_v3 import _install_selective
    from up_proj_batch_kernels import UpProjBatchKernels

    t0 = time.perf_counter()
    # Must match the configuration the 51.0 tok/s record was measured under
    # (pro_research/graph_e1f22.py:_new_runtime). embed_on_host defaults to
    # False, which parks the 704.6 MB BF16 embedding table in VRAM -- on an
    # 8 GiB card that is competing directly with the 4.33 GiB expert cache and
    # H-SCALE's 492 MiB scale planes. Building with the defaults measured
    # 24.7 tok/s here against a 51.0 tok/s record.
    rt = LightningRuntime(str(MODEL), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(capacity)
    rt.load_routed_bank()
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True

    dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
    _install_selective(rt, dense)
    install_batched_moe_dev(rt, down, up)

    if stack == "v18":
        # H-SCALE (resident down_proj scale planes) + B3 (PCIe/compute overlap),
        # fused. Must be installed BEFORE setup_graph: the capture binds the
        # _moe_dev that is live at capture time.
        #
        # free_all_blocks() first is LOAD-BEARING, not tidiness. H-SCALE needs a
        # contiguous 492.4 MiB for the scale planes against ~607 MiB free. CuPy's
        # pool holds every block it has ever grown into, so without returning the
        # unused ones the driver has far less than that available and the
        # allocation degrades into pool thrash instead of failing cleanly.
        # Omitting this one call is what made this server measure 24.8 tok/s
        # where the same stack benchmarks at 50.1 (scripts/.../bench_stacks.py),
        # and it showed up as setup_graph reporting 0 MiB of extra graph VRAM
        # instead of 524 MiB.
        import cupy as cp

        from moe_dev_combined import install_combined_moe_dev
        from moe_dev_scale_resident import planned_plane_bytes
        from scale_resident_kernels import ScaleResidentKernels

        cp.get_default_memory_pool().free_all_blocks()
        planned = planned_plane_bytes(rt)
        free_b = int(cp.cuda.Device(0).mem_info[0])
        print(f"[runtime] H-SCALE planes {planned / 2**20:.1f} MiB, "
              f"free {free_b / 2**20:.1f} MiB", flush=True)
        if planned > free_b:
            print("[runtime] WARNING: scale planes do not fit; falling back to v6",
                  flush=True)
            stack = "v6"
        else:
            install_combined_moe_dev(rt, down, up, ScaleResidentKernels())

    rt.setup_graph()
    print(f"[runtime] {stack} ready in {time.perf_counter() - t0:.1f}s "
          f"(graph extra VRAM {rt.graph_extra_vram_bytes / 2**20:.0f} MiB)",
          flush=True)
    return rt


def _harvest_last(rt) -> int:
    return int(rt.ring_harvest((rt._ring_i - 1) % rt._ring_size, 1)[0])


def _prefill(rt, prompt_ids):
    """Stage prompt tokens. The pinned staging ring is 256 slots, so a prompt
    longer than that must sync before it laps itself -- feeding more without a
    sync is exactly the race that produced garbage==garbage 'bit-exact' passes
    in the multi-sequence prototypes."""
    for i, tid in enumerate(prompt_ids):
        rt.step_graph(int(tid))
        if (i + 1) % 128 == 0:
            rt._graph_stream.synchronize()
    rt._graph_stream.synchronize()


def generate(rt, tok, prompt_ids, max_new: int, on_piece=None):
    rt.reset()
    _prefill(rt, prompt_ids)
    out_ids, pieces = [], []
    t0 = time.perf_counter()
    for _ in range(max_new):
        rt.step_graph()
        tid = _harvest_last(rt)
        if tid in EOS_IDS:
            break
        out_ids.append(tid)
        piece = tok.decode([tid], skip_special_tokens=True)
        pieces.append(piece)
        if on_piece is not None:
            on_piece(piece)
    dt = time.perf_counter() - t0
    return "".join(pieces), out_ids, (len(out_ids) / dt if dt > 0 else 0.0)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quieter than the default
        sys.stderr.write("[http] " + fmt % a + "\n")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/chat", "/index.html"):
            body = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": "lightning", "object": "model", "owned_by": "local",
                 "stack": _STACK}]})
        if self.path.rstrip("/") in ("/health", "/healthz"):
            return self._json(200, {"status": "ok", "stack": _STACK})
        return self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": f"no route {self.path}"}})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._json(400, {"error": {"message": f"bad JSON: {exc}"}})

        messages = req.get("messages") or []
        if not messages:
            return self._json(400, {"error": {"message": "messages is required"}})
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)
        stream = bool(req.get("stream"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        try:
            prompt_ids = _TOK.apply_chat_template(messages, add_generation_prompt=True)
        except Exception as exc:
            return self._json(400, {"error": {"message": f"chat template failed: {exc}"}})
        if len(prompt_ids) + max_new > _RT.max_ctx:
            return self._json(400, {"error": {
                "message": f"prompt {len(prompt_ids)} + max_tokens {max_new} exceeds "
                           f"context {_RT.max_ctx}"}})

        if not stream:
            with _LOCK:
                text, out_ids, tps = generate(_RT, _TOK, prompt_ids, max_new)
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created,
                "model": "lightning",
                "choices": [{"index": 0, "finish_reason":
                             "stop" if len(out_ids) < max_new else "length",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": len(prompt_ids),
                          "completion_tokens": len(out_ids),
                          "total_tokens": len(prompt_ids) + len(out_ids)},
                "x_tokens_per_second": round(tps, 2), "x_stack": _STACK,
            })

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        sse({"id": cid, "object": "chat.completion.chunk", "created": created,
             "model": "lightning",
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        try:
            with _LOCK:
                _, out_ids, tps = generate(
                    _RT, _TOK, prompt_ids, max_new,
                    on_piece=lambda p: sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": "lightning",
                        "choices": [{"index": 0, "delta": {"content": p},
                                     "finish_reason": None}]}))
            sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": "lightning",
                 "choices": [{"index": 0, "delta": {},
                              "finish_reason": "stop" if len(out_ids) < max_new else "length"}],
                 "x_tokens_per_second": round(tps, 2)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    global _RT, _TOK, _STACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--stack", choices=["v18", "v6"], default="v18")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    _STACK = args.stack
    _TOK = AutoTokenizer.from_pretrained(str(MODEL))
    _RT = build_v18_runtime(args.capacity, args.stack)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] OpenAI-compatible endpoint on http://{args.host}:{args.port}/v1\n"
          f"[serve] stack={args.stack}  model=lightning  ctx={_RT.max_ctx}\n"
          f"[serve] point any OpenAI client at it; one request at a time.",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
