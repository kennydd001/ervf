#!/usr/bin/env python3
"""Self-contained -I/-B bootstrap for device-free N4 production fixtures."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "preflight_het_next_l0_ph1_nvidia_n4_static.py"
spec = importlib.util.spec_from_file_location("ph1_nvidia_n4_preflight_isolated", PREFLIGHT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
row = module.isolated_suite()
sys.stdout.write(json.dumps(row, separators=(",", ":")))
raise SystemExit(0 if row == {"cleanup_faults": True, "verifier_mutations": True, "payload_bytes_read": 0} else 3)
