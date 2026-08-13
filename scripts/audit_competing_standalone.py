"""Standalone exact audit of the competing two-shell Fourier field.

This module intentionally imports only the Python standard library. It does
not use the production basis, polynomial tensor, symmetry implementation, or
certificate machinery. The 18 Fourier coefficients are written explicitly in
the fixed convention stated in the paper; exact rational convolution then
recomputes K, H, E, P, and A.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import TypeAlias


Q = Fraction
C: TypeAlias = tuple[Q, Q]
V: TypeAlias = tuple[C, C, C]
F: TypeAlias = dict[tuple[int, int, int], V]
ZERO: C = (Q(0), Q(0))


def cadd(x: C, y: C) -> C:
    return x[0] + y[0], x[1] + y[1]


def cneg(x: C) -> C:
    return -x[0], -x[1]


def cmul(x: C, y: C) -> C:
    return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def cscale(x: C, value: int | Q) -> C:
    return x[0] * value, x[1] * value


def cconj(x: C) -> C:
    return x[0], -x[1]


def vadd(x: V, y: V) -> V:
    return tuple(cadd(x[j], y[j]) for j in range(3))  # type: ignore[return-value]


def vscale(x: V, value: C | int | Q) -> V:
    scalar = value if isinstance(value, tuple) else (Q(value), Q(0))
    return tuple(cmul(scalar, x[j]) for j in range(3))  # type: ignore[return-value]


def dot(x: V, y: V, *, conjugate_left: bool = False) -> C:
    total = ZERO
    for j in range(3):
        left = cconj(x[j]) if conjugate_left else x[j]
        total = cadd(total, cmul(left, y[j]))
    return total


def k2(k: tuple[int, int, int]) -> int:
    return sum(component * component for component in k)


def project(k: tuple[int, int, int], value: V) -> V:
    """Exact Leray projection without importing production code."""

    kd = ZERO
    for j in range(3):
        kd = cadd(kd, cscale(value[j], k[j]))
    return tuple(
        cadd(value[j], cneg(cscale(kd, Q(k[j], k2(k))))) for j in range(3)
    )  # type: ignore[return-value]


def competing_field(a: Q, b: Q) -> F:
    """Explicit 18-mode field in the paper's fixed Fourier convention."""

    r = lambda x: (Q(x), Q(0))
    i = lambda x: (Q(0), Q(x))
    z: C = ZERO
    return {
        (-1, 0, 0): (z, z, r(a)), (1, 0, 0): (z, z, r(a)),
        (0, -1, 0): (r(-a), z, z), (0, 1, 0): (r(-a), z, z),
        (0, 0, -1): (z, r(a), z), (0, 0, 1): (z, r(a), z),
        (-1, -1, 0): (z, z, i(-2 * b)), (-1, 0, -1): (z, i(2 * b), z),
        (-1, 0, 1): (z, i(-2 * b), z), (-1, 1, 0): (z, z, i(-2 * b)),
        (0, -1, -1): (i(-2 * b), z, z), (0, -1, 1): (i(-2 * b), z, z),
        (0, 1, -1): (i(2 * b), z, z), (0, 1, 1): (i(2 * b), z, z),
        (1, -1, 0): (z, z, i(2 * b)), (1, 0, -1): (z, i(2 * b), z),
        (1, 0, 1): (z, i(-2 * b), z), (1, 1, 0): (z, z, i(2 * b)),
    }


PHYSICAL_SPACE_FORMULA = (
    "u(a,b)=(-2*a*cos(y)-8*b*sin(y)*cos(z), "
    "2*a*cos(z)+8*b*cos(x)*sin(z), "
    "2*a*cos(x)-8*b*sin(x)*cos(y))"
)


def nonlinear(field: F) -> F:
    """Compute -P[(u dot grad)u] by literal rational Fourier convolution."""

    raw: F = {}
    minus_i: C = (Q(0), Q(-1))
    for p, up in field.items():
        for q, uq in field.items():
            k = tuple(p[j] + q[j] for j in range(3))
            if k == (0, 0, 0):
                continue
            directional = ZERO
            for j in range(3):
                directional = cadd(directional, cscale(up[j], q[j]))
            term = vscale(uq, cmul(minus_i, directional))
            raw[k] = vadd(raw.get(k, (ZERO, ZERO, ZERO)), term)
    return {k: project(k, value) for k, value in raw.items()}


def weighted_inner(left: F, right: F, power: int) -> Q:
    """Exact real Fourier inner product weighted by |k|^(2*power)."""

    total = ZERO
    for k in left.keys() & right.keys():
        total = cadd(
            total,
            cscale(dot(left[k], right[k], conjugate_left=True), k2(k) ** power),
        )
    if total[1]:
        raise ArithmeticError("the allegedly real Fourier inner product is non-real")
    return total[0]


def curl(field: F) -> F:
    imaginary: C = (Q(0), Q(1))
    return {
        k: vscale((
            cadd(cscale(value[2], k[1]), cneg(cscale(value[1], k[2]))),
            cadd(cscale(value[0], k[2]), cneg(cscale(value[2], k[0]))),
            cadd(cscale(value[1], k[0]), cneg(cscale(value[0], k[1]))),
        ), imaginary)
        for k, value in field.items()
    }


def invariants(a: Q, b: Q) -> dict[str, Q]:
    field = competing_field(a, b)
    omega = curl(field)
    return {
        "kinetic_energy": weighted_inner(field, field, 0) / 2,
        "helicity": weighted_inner(field, omega, 0),
        "enstrophy": weighted_inner(field, field, 1) / 2,
        "palinstrophy": weighted_inner(field, field, 2) / 2,
        "stretching": weighted_inner(field, nonlinear(field), 1),
    }


def polynomial_coefficients() -> dict[str, tuple[Q, ...]]:
    """Recover all quadratic/cubic coefficients using only this module."""

    at10, at01 = invariants(Q(1), Q(0)), invariants(Q(0), Q(1))
    at11, at1m1 = invariants(Q(1), Q(1)), invariants(Q(1), Q(-1))

    def quadratic(name: str) -> tuple[Q, Q, Q]:
        return at10[name], at11[name] - at10[name] - at01[name], at01[name]

    c30, c03 = at10["stretching"], at01["stretching"]
    mixed_sum = at11["stretching"] - c30 - c03
    mixed_difference = at1m1["stretching"] - c30 + c03
    return {
        "kinetic_energy": quadratic("kinetic_energy"),
        "helicity": quadratic("helicity"),
        "enstrophy": quadratic("enstrophy"),
        "palinstrophy": quadratic("palinstrophy"),
        "stretching": (c30, (mixed_sum - mixed_difference) / 2,
                       (mixed_sum + mixed_difference) / 2, c03),
    }


EXPECTED = {
    "kinetic_energy": (Q(3), Q(0), Q(24)),
    "helicity": (Q(0), Q(0), Q(0)),
    "enstrophy": (Q(3), Q(0), Q(48)),
    "palinstrophy": (Q(3), Q(0), Q(96)),
    "stretching": (Q(0), Q(24), Q(0), Q(0)),
}


def audit() -> dict[str, object]:
    coefficients = polynomial_coefficients()
    return {
        "truth_label": "standalone_exact_rational_competing_field_audit",
        "independence_statement": (
            "This checker imports only Python standard-library modules and does "
            "not import extreme_flows or production certificate code."
        ),
        "mode_count": len(competing_field(Q(1), Q(1))),
        "physical_space_formula": PHYSICAL_SPACE_FORMULA,
        "coefficients": {name: [str(value) for value in row] for name, row in coefficients.items()},
        "expected_coefficients": {name: [str(value) for value in row] for name, row in EXPECTED.items()},
        "all_coefficients_match": coefficients == EXPECTED,
        "source": {
            "path": "scripts/audit_competing_standalone.py",
            "sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "scope": (
            "exact audit of the explicit two-amplitude field only; not a full "
            "64-variable KKT or global optimization certificate"
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = audit()
    if not payload["all_coefficients_match"]:
        raise AssertionError("standalone competing-field audit failed")
    output = root / "artifacts" / "proofs" / "competing_standalone_algebra_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
