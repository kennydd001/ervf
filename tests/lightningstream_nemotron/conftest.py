"""Make ``src`` importable without installing the project.

An editable install would regenerate ``src/moe_lab.egg-info``, which is a
protected artifact of the 80B line.  Putting ``src`` on ``sys.path`` here keeps
the Nemotron tests self-contained and leaves every protected byte untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
