"""Package every research line into one small analysable archive.

Includes the LIGHTNINGSTREAM_NEMOTRON line plus the protected Qwen3-30B
(STREAMQ5), Qwen3-Coder-Next (PORT80B), HET-NEXT and all earlier families
(CRAFT, RSIV, FLEQ, E2GQ, HERA, CORETAIL, BITFLOW, ERVF, TierFlow, DeepSeek
baseline).

Protected files are READ ONLY -- they are copied into a new archive inside the
Nemotron allowlist and never modified. The archive is written under
reports/lightningstream_nemotron/ rather than the repository root, so it cannot
be confused with the other agent's own research_docs_*.zip bundles.

Excluded: model shards, capture tensors, GGUF, existing zips, virtualenvs,
vendored third_party, __pycache__, and any single file over MAX_FILE_BYTES.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

MAX_FILE_BYTES = 2 * 1024 * 1024          # keep the bundle analysable
TEXT_EXT = {".md", ".yaml", ".yml", ".py", ".txt", ".toml", ".cfg", ".ini"}
DATA_EXT = {".json"}
JSON_MAX = 512 * 1024                      # result JSONs can be huge; cap them

SKIP_DIRS = {".git", ".venv", ".venv-next-ref", ".venv-nemotron", "__pycache__",
             ".pytest_cache", "third_party", "models", ".cache", "data"}
SKIP_EXT = {".safetensors", ".gguf", ".bin", ".zip", ".pt", ".pth", ".npy",
            ".npz", ".pyc", ".log", ".lock", ".jinja"}

ROOT_FILES = {"README.md", "pyproject.toml", "requirements.txt", ".gitignore"}


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def classify(rel: str) -> str:
    parts = rel.split("/")
    if "lightningstream_nemotron" in rel:
        return "lightningstream_nemotron (this line)"
    if "streamq5_moe" in rel:
        return "streamq5 / port80b / het-next (protected)"
    if "streamq4_moe" in rel:
        return "streamq4 (protected)"
    for fam in ("craft", "rsiv", "fleq", "e2gq", "hera", "dhera", "dchera",
                "ldhera", "adhera", "coretail", "bitflow", "offload_roofline",
                "qwen_gptq_bank"):
        if fam in rel.lower():
            return f"{fam} (protected)"
    if parts[0] == "docs":
        return "docs"
    if parts[0] == "info":
        return "info / analysis notes"
    if parts[0] == "reports" and len(parts) > 1 and parts[1] == "baseline":
        return "deepseek-v2-lite baseline (protected)"
    if parts[0] == "reports":
        return "reports (top level)"
    if parts[0] in ("scripts", "src", "tests"):
        return f"{parts[0]} (source)"
    return "other"


def should_include(path: Path, rel: str) -> tuple[bool, str]:
    ext = path.suffix.lower()
    if ext in SKIP_EXT:
        return False, "binary/archive extension"
    size = path.stat().st_size
    if ext in DATA_EXT:
        if size > JSON_MAX:
            return False, f"json over {JSON_MAX} bytes"
        return True, ""
    if ext in TEXT_EXT:
        if size > MAX_FILE_BYTES:
            return False, f"over {MAX_FILE_BYTES} bytes"
        return True, ""
    return False, "extension not in include set"


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_zip = OUT_DIR / f"ALL_RESEARCH_LINES_{stamp}.zip"

    included, excluded = [], []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        rel_dir = base.relative_to(REPO_ROOT).as_posix()
        if rel_dir == ".":
            filenames = [f for f in filenames if f in ROOT_FILES]
        for name in filenames:
            p = base / name
            rel = p.relative_to(REPO_ROOT).as_posix()
            try:
                ok, why = should_include(p, rel)
            except OSError:
                continue
            if ok:
                included.append((p, rel, p.stat().st_size))
            elif p.stat().st_size > 64 * 1024:
                excluded.append({"path": rel, "bytes": p.stat().st_size, "reason": why})

    groups: dict[str, dict] = {}
    for _, rel, size in included:
        g = classify(rel)
        groups.setdefault(g, {"files": 0, "bytes": 0})
        groups[g]["files"] += 1
        groups[g]["bytes"] += size

    manifest = {
        "kind": "lightningstream_nemotron_research_archive_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": out_zip.name,
        "purpose": ("All research lines in one analysable bundle: this Nemotron "
                    "line plus the protected Qwen3-30B / Qwen3-Coder-Next / "
                    "HET-NEXT / earlier-family work."),
        "protected_files_are_read_only": True,
        "protected_note": ("Protected artifacts are copied, never modified. This "
                           "archive lives in the Nemotron allowlist, not the "
                           "repository root, so it cannot be confused with the "
                           "other agent's research_docs_*.zip bundles."),
        "include_rules": {
            "text_extensions": sorted(TEXT_EXT),
            "json_max_bytes": JSON_MAX,
            "max_file_bytes": MAX_FILE_BYTES,
            "skipped_dirs": sorted(SKIP_DIRS),
            "skipped_extensions": sorted(SKIP_EXT),
        },
        "file_count": len(included),
        "uncompressed_bytes": sum(s for _, _, s in included),
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])),
        "notable_exclusions_over_64kib": sorted(
            excluded, key=lambda e: -e["bytes"])[:40],
        "excluded_count_over_64kib": len(excluded),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, rel, _ in sorted(included, key=lambda r: r[1]):
            z.write(p, rel)
        z.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")

    manifest["archive_bytes"] = out_zip.stat().st_size
    manifest["archive_sha256"] = sha256_path(out_zip)
    (OUT_DIR / "research_archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"archive : {out_zip.name}")
    print(f"files   : {len(included):,}")
    print(f"raw     : {manifest['uncompressed_bytes']:,} B")
    print(f"zipped  : {manifest['archive_bytes']:,} B "
          f"({manifest['archive_bytes']/1024/1024:.2f} MiB)")
    print(f"sha256  : {manifest['archive_sha256']}")
    print()
    for g, v in manifest["groups"].items():
        print(f"  {g:<44} {v['files']:>5} files  {v['bytes']/1024:>9.1f} KiB")
    print(f"\nexcluded (>64 KiB): {len(excluded)}")
    for e in manifest["notable_exclusions_over_64kib"][:6]:
        print(f"  {e['bytes']/1024/1024:>8.2f} MiB  {e['path']}  [{e['reason']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
