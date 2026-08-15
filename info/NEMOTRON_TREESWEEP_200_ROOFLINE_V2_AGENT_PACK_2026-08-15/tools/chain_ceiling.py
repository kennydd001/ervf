#!/usr/bin/env python3
"""Diagnostic homogeneous-chain acceptance calculator.

This is not a model of the full MTP process. It solves for a constant
conditional acceptance p such that sum_{i=1..k} p**i equals an observed mean
accepted-draft count, then reports finite/infinite chain expectations.
"""
from __future__ import annotations
import argparse


def solve_p(k: int, observed: float, iters: int = 100) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        value = sum(mid ** i for i in range(1, k + 1))
        if value < observed:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft-positions', type=int, default=5)
    ap.add_argument('--mean-accepted-drafts', type=float, default=2.114)
    ap.add_argument('--max-chain', type=int, default=32)
    args = ap.parse_args()

    p = solve_p(args.draft_positions, args.mean_accepted_drafts)
    print(f'conditional p (diagnostic): {p:.12f}')
    print(f'infinite accepted drafts:   {p/(1-p):.6f}')
    print(f'infinite output/round:      {1+p/(1-p):.6f}')
    print('\nfinite chains')
    for k in range(1, args.max_chain + 1):
        accepted = sum(p ** i for i in range(1, k + 1))
        print(f'{k:3d} positions -> accepted drafts {accepted:8.5f}, output {1+accepted:8.5f}')


if __name__ == '__main__':
    main()
