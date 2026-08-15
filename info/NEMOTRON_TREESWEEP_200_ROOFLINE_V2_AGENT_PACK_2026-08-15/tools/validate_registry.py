#!/usr/bin/env python3
"""Minimal structural validator for the TreeSweep registry.

Requires PyYAML. This intentionally validates only dependency references and
status vocabulary; scientific gates are independently verified later.
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

ALLOWED = {
    'queued','dependency_blocked','optional_dependency_blocked','running',
    'screen_positive','screen_negative','gate_passed','gate_failed',
    'falsified','inconclusive','queued_after_candidate','verified_breakthrough'
}


def main(path: str) -> None:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    exps = data.get('experiments', {})
    errors = []
    for name, exp in exps.items():
        status = exp.get('status')
        if status not in ALLOWED:
            errors.append(f'{name}: invalid status {status!r}')
        deps = []
        for key in ('depends_on','depends_on_any'):
            v = exp.get(key, [])
            if isinstance(v, str): v = [v]
            deps.extend(v)
        for dep in deps:
            if dep not in exps:
                errors.append(f'{name}: unknown dependency {dep}')
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print(f'OK: {len(exps)} experiments; all dependencies and statuses valid')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'EXPERIMENT_REGISTRY.yaml')
