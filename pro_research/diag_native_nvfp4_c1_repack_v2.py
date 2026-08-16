"""C1 pre-execution implementation fix.

The original C1 runner sampled large-shape `(m,sf)` coordinates *with
replacement* and then used uniqueness of mapped offsets as a diagnostic.
Duplicate input coordinates would therefore create duplicate offsets and could
falsely fail G4 even for a perfect bijection. No C1 result existed when this was
noticed. This wrapper changes only diagnostic sample construction: sampled
input coordinates are unique. The frozen swizzle formula, gates, representative
payload selection and thresholds are unchanged.
"""
from __future__ import annotations

import numpy as np

import diag_native_nvfp4_c1_repack as c1


def structural_shape_check_unique(M: int, SFK: int, rng: np.random.Generator) -> dict:
    natural_n = M * SFK
    native_n = c1.padded_count(M, SFK)
    boundary = {
        (0, 0), (max(M - 1, 0), 0), (0, max(SFK - 1, 0)),
        (max(M - 1, 0), max(SFK - 1, 0)),
    }
    for m in (31, 32, 63, 64, 95, 96, 127, 128, 129):
        if 0 <= m < M:
            for sf in (0, 3, 4, SFK - 1):
                if 0 <= sf < SFK:
                    boundary.add((m, sf))
    for sf in (3, 4, 7, 8, 11, 12):
        if 0 <= sf < SFK:
            for m in (0, min(31, M - 1), min(127, M - 1), M - 1):
                if m >= 0:
                    boundary.add((m, sf))

    enumerated = natural_n <= c1.MAX_ENUM_COORDS
    if enumerated:
        lin = np.arange(natural_n, dtype=np.int64)
    else:
        n = min(c1.RANDOM_COORDS_LARGE, natural_n)
        # Generator.choice without replacement makes input uniqueness a fact,
        # so output uniqueness measures the mapping rather than RNG collisions.
        lin = rng.choice(natural_n, size=n, replace=False).astype(np.int64, copy=False)
        if boundary:
            bb = np.asarray(sorted(boundary), dtype=np.int64)
            extra = bb[:, 0] * SFK + bb[:, 1]
            lin = np.unique(np.concatenate([lin, extra]))

    mm = lin // SFK
    ss = lin % SFK
    off = c1.swizzle_offset(mm, ss, M)
    im, isf = c1.inverse_offset(off, M)
    inverse_ok = bool(np.array_equal(mm, im) and np.array_equal(ss, isf))
    in_bounds = bool(off.size == 0 or ((off >= 0).all() and (off < native_n).all()))
    sampled_unique = bool(np.unique(off).size == off.size)

    return {
        "M": M, "SFK": SFK,
        "natural_count": natural_n, "native_padded_count": native_n,
        "padding_count": native_n - natural_n,
        "padding_fraction_of_natural": (native_n - natural_n) / natural_n if natural_n else 0.0,
        "enumerated_all_natural_coordinates": enumerated,
        "tested_coordinates": int(off.size),
        "input_coordinates_unique": bool(np.unique(lin).size == lin.size),
        "in_bounds": in_bounds,
        "inverse_exact": inverse_ok,
        "enumerated_unique": sampled_unique if enumerated else None,
        "sampled_unique": sampled_unique,
        "injective_by_exact_inverse": inverse_ok,
    }


c1.structural_shape_check = structural_shape_check_unique

if __name__ == "__main__":
    raise SystemExit(c1.main())
