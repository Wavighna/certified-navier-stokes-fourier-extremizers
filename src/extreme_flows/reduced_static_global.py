"""Global maximization certificate for the six-amplitude static ansatz.

The exact ansatz has three cubic monomials.  Sign choices make all three
nonnegative without changing E or P.  Every positive global maximizer is then
either fully interior, or is supported on exactly one of the three monomials:
two active monomials leave the third missing variable with a strictly improving
linear perturbation.  The interior KKT equations reduce to one quartic in the
Lagrange multiplier; each one-monomial face reduces to a quadratic.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import importlib.metadata
import json
from math import ceil
from pathlib import Path
import platform
import sys
from typing import Iterable, Sequence


Q = Fraction
Polynomial = tuple[Q, ...]  # coefficients in increasing degree order

_NU = Q(1, 100)
_INTERIOR_POLYNOMIAL: Polynomial = (
    Q(-127995727),
    Q(-3199658400),
    Q(-19990665000),
    Q(106000000),
    Q(431250000),
)
_FACE_TRIPLES = {
    "acd": (Q(64), (Q(2), Q(48), Q(8)), (Q(2), Q(144), Q(16))),
    "aef": (Q(32), (Q(2), Q(48), Q(4)), (Q(2), Q(144), Q(8))),
    "bdf": (Q(64), (Q(32), Q(8), Q(4)), (Q(128), Q(16), Q(8))),
}


def _trim(values: Iterable[Q]) -> Polynomial:
    output = list(values)
    while len(output) > 1 and output[-1] == 0:
        output.pop()
    return tuple(output or [Q(0)])


def _add(left: Polynomial, right: Polynomial) -> Polynomial:
    output = [Q(0)] * max(len(left), len(right))
    for index, value in enumerate(left):
        output[index] += value
    for index, value in enumerate(right):
        output[index] += value
    return _trim(output)


def _scale(values: Polynomial, scalar: Q) -> Polynomial:
    return _trim(scalar * value for value in values)


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    output = [Q(0)] * (len(left) + len(right) - 1)
    for i, value in enumerate(left):
        for j, other in enumerate(right):
            output[i + j] += value * other
    return _trim(output)


def _derivative(values: Polynomial) -> Polynomial:
    return _trim(index * values[index] for index in range(1, len(values)))


def _divide(left: Polynomial, right: Polynomial) -> tuple[Polynomial, Polynomial]:
    remainder = list(_trim(left))
    divisor = _trim(right)
    quotient = [Q(0)] * max(1, len(remainder) - len(divisor) + 1)
    while len(remainder) >= len(divisor) and any(remainder):
        coefficient = remainder[-1] / divisor[-1]
        shift = len(remainder) - len(divisor)
        quotient[shift] = coefficient
        for index, value in enumerate(divisor):
            remainder[index + shift] -= coefficient * value
        remainder = list(_trim(remainder))
    return _trim(quotient), _trim(remainder)


def _evaluate(values: Polynomial, point: Q) -> Q:
    total = Q(0)
    for value in reversed(values):
        total = total * point + value
    return total


def _sturm(values: Polynomial) -> tuple[Polynomial, ...]:
    sequence = [_trim(values), _derivative(values)]
    while any(sequence[-1]):
        _, remainder = _divide(sequence[-2], sequence[-1])
        if not any(remainder):
            break
        sequence.append(_scale(remainder, Q(-1)))
    return tuple(sequence)


def _variations(sequence: Sequence[Polynomial], point: Q) -> int:
    signs = []
    for polynomial in sequence:
        value = _evaluate(polynomial, point)
        if value:
            signs.append(1 if value > 0 else -1)
    return sum(left != right for left, right in zip(signs, signs[1:], strict=False))


def _root_count(sequence: Sequence[Polynomial], left: Q, right: Q) -> int:
    return _variations(sequence, left) - _variations(sequence, right)


def _positive_root_interval(
    values: Polynomial, *, bits: int = 100
) -> tuple[tuple[Q, Q], dict[str, object]]:
    """Isolate the unique positive root by exact Sturm bisection."""

    if values[-1] == 0 or values[0] == 0:
        raise ValueError("root isolation requires nonzero endpoint coefficients")
    bound = Q(1 + ceil(max(abs(value / values[-1]) for value in values[:-1])))
    sequence = _sturm(values)
    root_count = _root_count(sequence, Q(0), bound)
    if root_count != 1:
        raise ArithmeticError("polynomial does not have exactly one positive root")
    left, right = Q(0), bound
    for _ in range(bits):
        midpoint = (left + right) / 2
        if _evaluate(values, midpoint) == 0:
            return (midpoint, midpoint), {
                "positive_root_count": root_count,
                "cauchy_upper_bound": str(bound),
                "isolation_bits": bits,
            }
        if _root_count(sequence, left, midpoint):
            right = midpoint
        else:
            left = midpoint
    return (left, right), {
        "positive_root_count": root_count,
        "cauchy_upper_bound": str(bound),
        "isolation_bits": bits,
    }


def _arb_fraction(value: Q):
    from flint import arb

    return arb(value.numerator) / arb(value.denominator)


def _arb_interval(left: Q, right: Q):
    from flint import arb

    center = (left + right) / 2
    return arb(_arb_fraction(center), _arb_fraction((right - left) / 2))


def _ball(value) -> dict[str, object]:
    return {
        "ball": str(value),
        "lower": str(value.lower()),
        "upper": str(value.upper()),
        "finite": bool(value.is_finite()),
    }


def _interior_quantities(multiplier):
    """Return X=a², Y=d², Z=f² and the exact interior rate expression."""

    nu = _arb_fraction(_NU)
    b_factor = 1 / (multiplier + _arb_fraction(Q(2, 25)))
    c_factor = _arb_fraction(Q(2, 3)) / (multiplier + _arb_fraction(Q(3, 50)))
    e_factor = _arb_fraction(Q(1, 3)) / (multiplier + _arb_fraction(Q(3, 50)))
    x = (
        3
        * (multiplier + _arb_fraction(Q(3, 50)))
        * (
            3 * multiplier * multiplier
            + _arb_fraction(Q(12, 25)) * multiplier
            + _arb_fraction(Q(39, 2500))
        )
        / (32 * (multiplier + _arb_fraction(Q(2, 25))))
    )
    y = (
        5 * multiplier * multiplier
        + _arb_fraction(Q(12, 25)) * multiplier
        + _arb_fraction(Q(1, 100))
    ) / 64
    z = (multiplier * multiplier - _arb_fraction(Q(7, 2500))) / 16
    stretching = 64 * c_factor * x * y + 32 * e_factor * x * z + 64 * b_factor * y * z
    palinstrophy = (
        2 * x
        + 128 * b_factor * b_factor * y * z
        + 144 * c_factor * c_factor * x * y
        + 16 * y
        + 144 * e_factor * e_factor * x * z
        + 8 * z
    )
    return x, y, z, stretching - 2 * nu * palinstrophy


def _face_polynomial(
    coefficient: Q, enstrophy: Sequence[Q], palinstrophy: Sequence[Q]
) -> Polynomial:
    alpha = [
        (4 * _NU * palinstrophy[index], 2 * enstrophy[index])
        for index in range(3)
    ]
    output: Polynomial = (Q(-100) * coefficient * coefficient,)
    for index in range(3):
        first, second = alpha[(index + 1) % 3], alpha[(index + 2) % 3]
        output = _add(output, _scale(_multiply(first, second), enstrophy[index]))
    return output


def _face_rate(multiplier, coefficient: Q, enstrophy: Sequence[Q], palinstrophy: Sequence[Q]):
    nu = _arb_fraction(_NU)
    alpha = [
        2 * _arb_fraction(enstrophy[index]) * multiplier
        + 4 * nu * _arb_fraction(palinstrophy[index])
        for index in range(3)
    ]
    squared = [
        alpha[(index + 1) % 3] * alpha[(index + 2) % 3]
        / (_arb_fraction(coefficient) ** 2)
        for index in range(3)
    ]
    stretching = alpha[0] * alpha[1] * alpha[2] / (_arb_fraction(coefficient) ** 2)
    p = sum(
        (_arb_fraction(palinstrophy[index]) * squared[index] for index in range(3)),
        _arb_fraction(Q(0)),
    )
    return stretching - 2 * nu * p


def _dependency() -> dict[str, object]:
    distribution = importlib.metadata.distribution("python-flint")
    root = Path(distribution.locate_file(""))
    record = root / "python_flint-0.9.0.dist-info" / "RECORD"
    return {
        "package": "python-flint",
        "version": distribution.version,
        "python": sys.version,
        "platform": platform.platform(),
        "record_sha256": sha256(record.read_bytes()).hexdigest(),
    }


def _source_record(script_name: str) -> dict[str, object]:
    module = Path(__file__).resolve()
    root = module.parents[2]
    script = root / "scripts" / script_name
    def record(path: Path) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        }
    return {
        "module": record(module),
        "script": record(script),
        "reduced_formula": record(
            root / "src" / "extreme_flows" / "reduced_static.py"
        ),
        "symmetry_definition": record(
            root / "src" / "extreme_flows" / "static_symmetry.py"
        ),
    }


def evaluate_global_static_certificate(*, precision: int, bits: int = 100) -> dict[str, object]:
    """Certify the interior/face comparison at one Arb precision."""

    from flint import ctx

    ctx.prec = int(precision)
    interior_interval, interior_sturm = _positive_root_interval(
        _INTERIOR_POLYNOMIAL, bits=bits
    )
    multiplier = _arb_interval(*interior_interval)
    x, y, z, interior_rate = _interior_quantities(multiplier)
    if not all(value.lower() > 0 for value in (x, y, z)):
        raise ArithmeticError("positive interior multiplier did not yield positive squares")
    faces: dict[str, object] = {}
    dominates = True
    for name, (coefficient, e_weights, p_weights) in _FACE_TRIPLES.items():
        polynomial = _face_polynomial(coefficient, e_weights, p_weights)
        interval, sturm = _positive_root_interval(polynomial, bits=bits)
        face_rate = _face_rate(_arb_interval(*interval), coefficient, e_weights, p_weights)
        dominates = dominates and interior_rate.lower() > face_rate.upper()
        faces[name] = {
            "multiplier_interval": [str(interval[0]), str(interval[1])],
            "sturm": sturm,
            "polynomial_low_to_high": [str(value) for value in polynomial],
            "rate": _ball(face_rate),
            "strictly_below_interior": bool(interior_rate.lower() > face_rate.upper()),
        }
    return {
        "precision_bits": precision,
        "interior": {
            "multiplier_interval": [str(interior_interval[0]), str(interior_interval[1])],
            "sturm": interior_sturm,
            "polynomial_low_to_high": [str(value) for value in _INTERIOR_POLYNOMIAL],
            "squared_amplitudes": {"a2": _ball(x), "d2": _ball(y), "f2": _ball(z)},
            "rate": _ball(interior_rate),
        },
        "faces": faces,
        "strict_interior_dominance": dominates,
    }


def build_global_static_certificate(
    *, precisions: Sequence[int] = (256, 512), bits: int = 100
) -> dict[str, object]:
    """Build the exact global maximum certificate inside the six-mode ansatz."""

    precision_set = tuple(int(value) for value in precisions)
    if len(set(precision_set)) < 2 or not {256, 512}.issubset(precision_set):
        raise ValueError("proof-grade builders require distinct 256-bit and 512-bit runs")
    runs = [
        evaluate_global_static_certificate(precision=precision, bits=bits)
        for precision in precision_set
    ]
    certified = all(bool(run["strict_interior_dominance"]) for run in runs)
    return {
        "schema_version": 1,
        "truth_label": "proof_grade_global_static_ansatz_maximum" if certified else "global_static_certificate_incomplete",
        "problem": {
            "functional": "R=A-2*nu*P at E=100",
            "viscosity": "1/100",
            "ansatz_dimension": 6,
            "interior_multiplier_polynomial_low_to_high": [str(value) for value in _INTERIOR_POLYNOMIAL],
        },
        "proof_structure": {
            "sign_gauge": "three independent sign constraints make all cubic monomials nonnegative while preserving E and P",
            "interior": "all-nonzero KKT equations reduce to the displayed quartic",
            "boundary": "a positive local maximum cannot have exactly two active monomials; one-monomial faces reduce to three quadratics",
            "zero_monomial_faces": "have R<=0 or are dominated by their one-monomial full-energy face maximum",
        },
        "arb": {"dependency": _dependency(), "runs": runs},
        "checker_source": _source_record("certify_reduced_static_global.py"),
        "claims": {
            "unique_positive_interior_stationary_branch": certified,
            "all_positive_boundary_face_maxima_strictly_lower": certified,
            "global_maximum_within_six_amplitude_static_ansatz": certified,
            "global_maximum_in_full_64d_static_class": False,
            "navier_stokes_trajectory_or_regularity_statement": False,
        },
        "limitations": {
            "six_amplitude_static_ansatz_only": True,
            "no_full_64d_global_claim": True,
            "no_dynamical_invariance_or_regularization_claim": True,
        },
    }


__all__ = ["build_global_static_certificate", "evaluate_global_static_certificate"]
