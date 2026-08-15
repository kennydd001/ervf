# P6B lokale testserver

Status op 2026-08-12: actief op `http://127.0.0.1:8765`, proces-ID 24616.

## Browser

Open `http://127.0.0.1:8765/`. De pagina gebruikt standaard ruwe completion,
omdat het geladen checkpoint Qwen3-30B-A3B **Base** is en geen instruction-tuned
chatmodel. De chat-template is beschikbaar als experimentele modus, maar kan
rollen of prompttekst herhalen.

## HTTP-API

- `GET /health`
- `GET /v1/models`
- `POST /api/generate`
- `POST /v1/completions`
- `POST /v1/chat/completions` (`stream=true` nog niet ondersteund)

Voorbeeldbody voor `/api/generate`:

```json
{
  "prompt": "The future of efficient artificial intelligence is",
  "mode": "raw",
  "domain": "general",
  "max_new_tokens": 128
}
```

Geldige domeincaches zijn `general`, `code`, `math`, `multilingual` en
`instruction`. De som van prompt- en generatietokens mag maximaal 4.096 zijn;
per aanvraag zijn maximaal 512 nieuwe tokens toegestaan. Generaties worden
geserialiseerd omdat één fysieke GPU-runtime wordt gedeeld.

## Proces en logs

- server: `scripts/streamq5_moe/serve_p6b.py`;
- stdout: `reports/streamq5_moe/p6b_server_stdout.log`;
- stderr: `reports/streamq5_moe/p6b_server_stderr.log`;
- append-only aanvraagrapport: `reports/streamq5_moe/p6b_server_sessions.jsonl`.

De service bindt bewust uitsluitend aan localhost en heeft geen authenticatie.
Start hem daarom niet met een extern bindadres.

Veilig stoppen vanuit PowerShell:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
if ($process.CommandLine -like '*serve_p6b.py*') {
    Stop-Process -Id $listener.OwningProcess
}
```

Handmatig herstarten vanuit de projectmap:

```powershell
.\.venv\Scripts\python.exe .\scripts\streamq5_moe\serve_p6b.py --host 127.0.0.1 --port 8765
```

## Gecontroleerde serverprobe

De post-restartprobe genereerde 16 tokens voor het vaste P6-prompt met
18,223 tok/s, mean 54,874 ms en p95 66,835 ms. Uitvoer:

> here, and it's called the AI Chip. This revolutionary technology is set to
