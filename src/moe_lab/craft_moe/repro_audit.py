from __future__ import annotations

import hashlib
import json
import math
import mmap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODEL_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
AUDIT_SEED = 20260810


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    experiment: str
    category: str
    passed: bool
    observed: Any
    expected: Any
    evidence: str
    severity: str = "error"


class AuditCollector:
    """Collect machine-readable audit assertions without short-circuiting."""

    def __init__(self) -> None:
        self.checks: list[AuditCheck] = []

    def add(
        self,
        check_id: str,
        experiment: str,
        category: str,
        passed: bool,
        observed: Any,
        expected: Any,
        evidence: str,
        *,
        severity: str = "error",
    ) -> bool:
        if severity not in {"error", "warning"}:
            raise ValueError(f"unsupported severity: {severity}")
        self.checks.append(
            AuditCheck(
                check_id=check_id,
                experiment=experiment,
                category=category,
                passed=bool(passed),
                observed=observed,
                expected=expected,
                evidence=evidence,
                severity=severity,
            )
        )
        return bool(passed)

    def equal(
        self,
        check_id: str,
        experiment: str,
        category: str,
        observed: Any,
        expected: Any,
        evidence: str,
        *,
        severity: str = "error",
    ) -> bool:
        return self.add(
            check_id,
            experiment,
            category,
            observed == expected,
            observed,
            expected,
            evidence,
            severity=severity,
        )

    def close(
        self,
        check_id: str,
        experiment: str,
        category: str,
        observed: float,
        expected: float,
        evidence: str,
        *,
        abs_tol: float = 1e-12,
        rel_tol: float = 1e-12,
        severity: str = "error",
    ) -> bool:
        passed = math.isfinite(float(observed)) and math.isclose(
            float(observed), float(expected), abs_tol=abs_tol, rel_tol=rel_tol
        )
        return self.add(
            check_id,
            experiment,
            category,
            passed,
            observed,
            {"value": expected, "abs_tol": abs_tol, "rel_tol": rel_tol},
            evidence,
            severity=severity,
        )

    @property
    def failures(self) -> list[AuditCheck]:
        return [
            check
            for check in self.checks
            if not check.passed and check.severity == "error"
        ]

    @property
    def warnings(self) -> list[AuditCheck]:
        return [
            check
            for check in self.checks
            if not check.passed and check.severity == "warning"
        ]

    def summary(self) -> dict[str, Any]:
        passed = sum(check.passed for check in self.checks)
        return {
            "checks": len(self.checks),
            "passed": passed,
            "failed": len(self.failures),
            "warnings": len(self.warnings),
            "all_required_checks_pass": not self.failures,
        }

    def serializable_checks(self) -> list[dict[str, Any]]:
        return [asdict(check) for check in self.checks]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _top_level_member_spans(path: Path) -> dict[str, tuple[int, int]]:
    """Index pretty-printed top-level JSON values without loading large arrays.

    CRAFT result files are written with ``json.dumps(..., indent=2)``. A top-level
    key therefore begins immediately after a newline with two spaces. Searching
    those delimiters happens inside ``mmap.find`` (rather than a Python byte loop),
    which keeps the two ~1 GiB CRCQ reports cheap to inspect.
    """

    spans: dict[str, tuple[int, int]] = {}
    with path.open("rb") as handle:
        if handle.seek(0, 2) == 0:
            raise ValueError(f"empty JSON file: {path}")
        handle.seek(0)
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            if mapped[:1] != b"{":
                raise ValueError(f"expected a JSON object: {path}")
            positions: list[tuple[str, int, int]] = []
            cursor = 0
            marker = b'\n  "'
            while True:
                line_start = mapped.find(marker, cursor)
                if line_start < 0:
                    break
                key_start = line_start + len(b"\n  ")
                quote_end = mapped.find(b'":', key_start + 1)
                if quote_end < 0:
                    raise ValueError(f"malformed top-level key in {path}")
                raw_key = mapped[key_start : quote_end + 1]
                key = json.loads(raw_key.decode("utf-8"))
                value_start = quote_end + 2
                while mapped[value_start : value_start + 1] in b" \t\r\n":
                    value_start += 1
                positions.append((key, line_start, value_start))
                cursor = quote_end + 2

            if not positions:
                stripped = mapped[:].strip()
                if stripped == b"{}":
                    return {}
                raise ValueError(
                    f"JSON is not in the expected two-space pretty format: {path}"
                )

            object_end = mapped.rfind(b"}")
            for index, (key, _, value_start) in enumerate(positions):
                value_end = (
                    positions[index + 1][1]
                    if index + 1 < len(positions)
                    else object_end
                )
                while value_end > value_start and mapped[value_end - 1 : value_end] in b" \t\r\n":
                    value_end -= 1
                if value_end > value_start and mapped[value_end - 1 : value_end] == b",":
                    value_end -= 1
                while value_end > value_start and mapped[value_end - 1 : value_end] in b" \t\r\n":
                    value_end -= 1
                spans[key] = (value_start, value_end)
    return spans


def load_top_level_members(
    path: Path,
    keys: Iterable[str],
    *,
    maximum_member_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Load selected top-level members from a potentially multi-gigabyte JSON."""

    requested = set(keys)
    spans = _top_level_member_spans(path)
    missing = requested.difference(spans)
    if missing:
        raise KeyError(f"missing top-level keys in {path}: {sorted(missing)}")
    result: dict[str, Any] = {}
    with path.open("rb") as handle:
        for key in requested:
            start, end = spans[key]
            size = end - start
            if size > maximum_member_bytes:
                raise MemoryError(
                    f"refusing to materialize {key!r} ({size:,} bytes) from {path}"
                )
            handle.seek(start)
            result[key] = json.loads(handle.read(size))
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise KeyError(".".join(path))
        cursor = cursor[key]
    return cursor


def upper_empirical_quantile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("quantile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(int(value) for value in values)
    index = math.ceil(probability * (len(ordered) - 1))
    return ordered[index]


def ratio_reduction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("reduction denominator must be positive")
    return 1.0 - float(numerator) / float(denominator)


def gap_closure(q3: float, q4: float, candidate: float) -> float:
    denominator = float(q3) - float(q4)
    if denominator <= 0:
        raise ValueError("Q3-to-Q4 denominator must be positive")
    return (float(q3) - float(candidate)) / denominator


def disjoint(values_a: Iterable[int], values_b: Iterable[int]) -> bool:
    return set(map(int, values_a)).isdisjoint(map(int, values_b))


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def contains_value(value: Any, expected: Any) -> bool:
    if isinstance(value, Mapping):
        return any(contains_value(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_value(item, expected) for item in value)
    return value == expected


def values_for_key(value: Any, target_key: str) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == target_key:
                matches.append(item)
            matches.extend(values_for_key(item, target_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            matches.extend(values_for_key(item, target_key))
    return matches


def paired_load_bootstrap_independent(
    baseline_misses: Sequence[int],
    primary_avoided: Sequence[int],
    zero_avoided: Sequence[int],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_misses, dtype=np.float64)
    primary = np.asarray(primary_avoided, dtype=np.float64)
    zero = np.asarray(zero_avoided, dtype=np.float64)
    if not (
        baseline.ndim == primary.ndim == zero.ndim == 1
        and baseline.size == primary.size == zero.size
        and baseline.size > 0
    ):
        raise ValueError("paired block counts must be aligned and non-empty")
    if np.any(baseline <= 0) or np.any(primary < 0) or np.any(zero < 0):
        raise ValueError("invalid load count")
    if np.any(primary > baseline) or np.any(zero > baseline):
        raise ValueError("avoided loads exceed baseline")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(resamples, baseline.size))
    sampled_baseline = baseline[indices].sum(axis=1)
    raw = {
        "primary_miss_reduction_fraction": (
            primary[indices].sum(axis=1) / sampled_baseline
        ),
        "zero_fill_miss_reduction_fraction": (
            zero[indices].sum(axis=1) / sampled_baseline
        ),
    }
    raw["span_uplift_fraction"] = (
        raw["primary_miss_reduction_fraction"]
        - raw["zero_fill_miss_reduction_fraction"]
    )

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        return {"low": float(low), "high": float(high)}

    return {
        "method": "paired sequence-block percentile bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "sampling_units": int(baseline.size),
        "point_estimates": {
            "primary_miss_reduction_fraction": float(primary.sum() / baseline.sum()),
            "zero_fill_miss_reduction_fraction": float(zero.sum() / baseline.sum()),
            "span_uplift_fraction": float((primary.sum() - zero.sum()) / baseline.sum()),
        },
        "intervals_95": {name: interval(values) for name, values in raw.items()},
        "raw": {name: values.tolist() for name, values in raw.items()},
    }


def paired_gap_bootstrap_independent(
    q3_kl: Sequence[float],
    q4_kl: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
    *,
    block_size: int,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    q3 = np.asarray(q3_kl, dtype=np.float64)
    q4 = np.asarray(q4_kl, dtype=np.float64)
    candidate_arrays = {
        name: np.asarray(series, dtype=np.float64)
        for name, series in candidates.items()
    }
    if q3.ndim != 1 or q3.size == 0 or q3.size % block_size:
        raise ValueError("KL series do not form complete sequence blocks")
    if q4.shape != q3.shape or any(
        series.shape != q3.shape for series in candidate_arrays.values()
    ):
        raise ValueError("paired KL series are not aligned")
    if not (
        np.isfinite(q3).all()
        and np.isfinite(q4).all()
        and all(np.isfinite(series).all() for series in candidate_arrays.values())
    ):
        raise ValueError("KL series contain a non-finite value")
    blocks = q3.size // block_size

    def block_sums(series: np.ndarray) -> np.ndarray:
        return series.reshape(blocks, block_size).sum(axis=1)

    q3_sums = block_sums(q3)
    q4_sums = block_sums(q4)
    candidate_sums = {
        name: block_sums(series) for name, series in candidate_arrays.items()
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, blocks, size=(resamples, blocks))
    divisor = blocks * block_size
    q3_sample = q3_sums[indices].sum(axis=1) / divisor
    q4_sample = q4_sums[indices].sum(axis=1) / divisor
    denominator = q3_sample - q4_sample
    if np.any(denominator <= 0):
        raise ValueError("bootstrap encountered a non-positive Q3-to-Q4 gap")
    closures = {
        name: (q3_sample - sums[indices].sum(axis=1) / divisor) / denominator
        for name, sums in candidate_sums.items()
    }
    point_gap = float(q3.mean() - q4.mean())

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        return {"low": float(low), "high": float(high)}

    return {
        "method": "paired sequence-block percentile bootstrap with replacement",
        "seed": seed,
        "resamples": resamples,
        "sampling_units": blocks,
        "block_size": block_size,
        "point_gap": point_gap,
        "point_closure": {
            name: float((q3.mean() - series.mean()) / point_gap)
            for name, series in candidate_arrays.items()
        },
        "intervals_95": {
            name: interval(values) for name, values in closures.items()
        },
        "probability_ge_0_10": {
            name: float((values >= 0.10).mean())
            for name, values in closures.items()
        },
        "probability_ge_0_20": {
            name: float((values >= 0.20).mean())
            for name, values in closures.items()
        },
        "raw": {name: values.tolist() for name, values in closures.items()},
    }
