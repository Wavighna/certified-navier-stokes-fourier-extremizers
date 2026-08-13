"""Standalone exact audit of the six-amplitude static Fourier reduction.

This deliberately uses only the standard library.  The six Fourier basis
fields are written literally, and exact rational convolution recovers every
quadratic and cubic coefficient without importing the production basis,
polynomial tensor, symmetry code, or certificate machinery.
"""

from __future__ import annotations

from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
from typing import TypeAlias


C: TypeAlias = tuple[Q, Q]
V: TypeAlias = tuple[C, C, C]
F: TypeAlias = dict[tuple[int, int, int], V]
ZERO: C = (Q(0), Q(0))
NAMES = "abcdef"


def _cadd(x: C, y: C) -> C:
    return x[0] + y[0], x[1] + y[1]


def _cneg(x: C) -> C:
    return -x[0], -x[1]


def _cmul(x: C, y: C) -> C:
    return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def _cscale(x: C, value: int | Q) -> C:
    return x[0] * value, x[1] * value


def _cconj(x: C) -> C:
    return x[0], -x[1]


def _vadd(x: V, y: V) -> V:
    return tuple(_cadd(x[i], y[i]) for i in range(3))  # type: ignore[return-value]


def _vscale(x: V, value: C | int | Q) -> V:
    scalar = value if isinstance(value, tuple) else (Q(value), Q(0))
    return tuple(_cmul(scalar, x[i]) for i in range(3))  # type: ignore[return-value]


def _dot(x: V, y: V, *, conjugate_left: bool = False) -> C:
    total = ZERO
    for i in range(3):
        total = _cadd(total, _cmul(_cconj(x[i]) if conjugate_left else x[i], y[i]))
    return total


def _k2(k: tuple[int, int, int]) -> int:
    return sum(component * component for component in k)


def _project(k: tuple[int, int, int], value: V) -> V:
    kd = ZERO
    for i in range(3):
        kd = _cadd(kd, _cscale(value[i], k[i]))
    return tuple(
        _cadd(value[i], _cneg(_cscale(kd, Q(k[i], _k2(k))))) for i in range(3)
    )  # type: ignore[return-value]


def _basis_fields() -> tuple[F, ...]:
    """Literal Fourier fields for amplitudes (a,b,c,d,e,f)."""

    r = lambda value: (Q(value), Q(0))
    i = lambda value: (Q(0), Q(value))
    z = ZERO
    return (
        {(0, 0, -1): (r(1), r(1), z), (0, 0, 1): (r(1), r(1), z)},
        {(-2, 0, 0): (z, z, r(2)), (0, -2, 0): (z, z, r(-2)),
         (0, 2, 0): (z, z, r(-2)), (2, 0, 0): (z, z, r(2))},
        {(-1, 1, -1): (i(2), i(2), z), (-1, 1, 1): (i(-2), i(-2), z),
         (1, -1, -1): (i(2), i(2), z), (1, -1, 1): (i(-2), i(-2), z)},
        {(-1, 1, 0): (z, z, r(2)), (1, -1, 0): (z, z, r(2))},
        {(-1, -1, -1): (r(-2), r(2), z), (-1, -1, 1): (r(-2), r(2), z),
         (1, 1, -1): (r(-2), r(2), z), (1, 1, 1): (r(-2), r(2), z)},
        {(-1, -1, 0): (i(1), i(-1), z), (1, 1, 0): (i(-1), i(1), z)},
    )


def _field(amplitudes: tuple[Q, ...]) -> F:
    result: F = {}
    for amplitude, basis in zip(amplitudes, _basis_fields(), strict=True):
        for k, value in basis.items():
            result[k] = _vadd(result.get(k, (ZERO, ZERO, ZERO)), _vscale(value, amplitude))
    return {k: value for k, value in result.items() if value != (ZERO, ZERO, ZERO)}


def _nonlinear(field: F) -> F:
    raw: F = {}
    minus_i: C = (Q(0), Q(-1))
    for p, up in field.items():
        for q, uq in field.items():
            k = tuple(p[i] + q[i] for i in range(3))
            if k == (0, 0, 0):
                continue
            directional = ZERO
            for i in range(3):
                directional = _cadd(directional, _cscale(up[i], q[i]))
            raw[k] = _vadd(raw.get(k, (ZERO, ZERO, ZERO)), _vscale(uq, _cmul(minus_i, directional)))
    return {k: _project(k, value) for k, value in raw.items()}


def _weighted_inner(left: F, right: F, power: int) -> Q:
    total = ZERO
    for k in left.keys() & right.keys():
        total = _cadd(total, _cscale(_dot(left[k], right[k], conjugate_left=True), _k2(k) ** power))
    if total[1]:
        raise ArithmeticError("expected real Fourier inner product")
    return total[0]


def _invariants(amplitudes: tuple[Q, ...]) -> dict[str, Q]:
    field = _field(amplitudes)
    return {
        "enstrophy": _weighted_inner(field, field, 1) / 2,
        "palinstrophy": _weighted_inner(field, field, 2) / 2,
        "stretching": _weighted_inner(field, _nonlinear(field), 1),
    }


def _unit(index: int, sign: int = 1) -> tuple[Q, ...]:
    return tuple(Q(sign) if i == index else Q(0) for i in range(6))


def _add(*vectors: tuple[Q, ...]) -> tuple[Q, ...]:
    return tuple(sum((vector[i] for vector in vectors), Q(0)) for i in range(6))


def _coefficients() -> dict[str, dict[str, Q]]:
    """Recover all homogeneous degree-two and degree-three coefficients."""

    basis = [_unit(i) for i in range(6)]
    values = [_invariants(vector) for vector in basis]
    result: dict[str, dict[str, Q]] = {"enstrophy": {}, "palinstrophy": {}, "stretching": {}}
    for name in ("enstrophy", "palinstrophy"):
        for i in range(6):
            result[name][NAMES[i] * 2] = values[i][name]
        for i in range(6):
            for j in range(i + 1, 6):
                result[name][NAMES[i] + NAMES[j]] = _invariants(_add(basis[i], basis[j]))[name] - values[i][name] - values[j][name]
    name = "stretching"
    for i in range(6):
        result[name][NAMES[i] * 3] = values[i][name]
    for i in range(6):
        for j in range(i + 1, 6):
            plus = _invariants(_add(basis[i], basis[j]))[name] - values[i][name] - values[j][name]
            minus = _invariants(_add(basis[i], _unit(j, -1)))[name] - values[i][name] + values[j][name]
            result[name][NAMES[i] * 2 + NAMES[j]] = (plus - minus) / 2
            result[name][NAMES[i] + NAMES[j] * 2] = (plus + minus) / 2
    for i in range(6):
        for j in range(i + 1, 6):
            for k in range(j + 1, 6):
                triple = _invariants(_add(basis[i], basis[j], basis[k]))[name]
                pairs = sum((_invariants(_add(basis[x], basis[y]))[name] for x, y in ((i, j), (i, k), (j, k))), Q(0))
                result[name][NAMES[i] + NAMES[j] + NAMES[k]] = triple - pairs + values[i][name] + values[j][name] + values[k][name]
    return result


EXPECTED = {
    "enstrophy": {"aa": Q(2), "bb": Q(32), "cc": Q(48), "dd": Q(8), "ee": Q(48), "ff": Q(4)},
    "palinstrophy": {"aa": Q(2), "bb": Q(128), "cc": Q(144), "dd": Q(16), "ee": Q(144), "ff": Q(8)},
    "stretching": {"acd": Q(64), "aef": Q(32), "bdf": Q(-64)},
}


def audit() -> dict[str, object]:
    coefficients = _coefficients()
    nonzero = {name: {term: value for term, value in polynomial.items() if value} for name, polynomial in coefficients.items()}
    return {
        "truth_label": "standalone_exact_rational_six_amplitude_fourier_audit",
        "independence_statement": "Uses only Python standard-library modules and literal Fourier fields; imports no project implementation.",
        "basis_mode_counts": [len(field) for field in _basis_fields()],
        "all_coefficients": {name: {term: str(value) for term, value in polynomial.items()} for name, polynomial in coefficients.items()},
        "nonzero_coefficients": {name: {term: str(value) for term, value in polynomial.items()} for name, polynomial in nonzero.items()},
        "expected_nonzero_coefficients": {name: {term: str(value) for term, value in polynomial.items()} for name, polynomial in EXPECTED.items()},
        "all_coefficients_match": nonzero == EXPECTED,
        "source": {"path": "scripts/audit_reduced_static_standalone.py", "sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
        "scope": "exact audit of the displayed six-amplitude Fourier space only; not a full-64-variable global certificate",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = audit()
    if not payload["all_coefficients_match"]:
        raise AssertionError("standalone six-amplitude Fourier audit failed")
    output = root / "artifacts" / "proofs" / "reduced_static_standalone_algebra_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
