"""Exact degree-six target for the full 64-control inviscid cubic conjecture.

This module deliberately does *not* assert that the target polynomial is
nonnegative.  It builds the exact sparse polynomial whose positivity would
prove the proposed full-space extension of the six-amplitude extremizer:

    (8832/42849) E(x)^3 - A(x)^2 >= 0.

Here ``E`` and ``A`` are the exact finite Fourier forms defined by
``StaticLowModePolynomial``.  The output is a proof target for a future SOS,
representation-block, or exact branch-and-bound certificate, not a
certificate itself.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import factorial
from hashlib import sha256
from itertools import permutations
from typing import Mapping, Sequence

import numpy as np

from .certify import StaticLowModePolynomial
from .static_symmetry import IDENTITY_GENERATOR, static_symmetry_matrix


Monomial = tuple[int, ...]
SparsePolynomial = dict[Monomial, Fraction]
INVISCID_CUBIC_CONSTANT_SQUARED = Fraction(8832, 42849)


def _add_term(polynomial: SparsePolynomial, monomial: Monomial, coefficient: Fraction) -> None:
    if coefficient:
        polynomial[monomial] = polynomial.get(monomial, Fraction(0)) + coefficient
        if not polynomial[monomial]:
            del polynomial[monomial]


def _multiply(left: Mapping[Monomial, Fraction], right: Mapping[Monomial, Fraction]) -> SparsePolynomial:
    result: SparsePolynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            _add_term(
                result,
                tuple(sorted(left_monomial + right_monomial)),
                left_coefficient * right_coefficient,
            )
    return result


def exact_stretching_cubic_terms(model: StaticLowModePolynomial | None = None) -> SparsePolynomial:
    """Return the canonical-monomial form of the exact stretching cubic ``A``.

    ``third_derivative_terms`` stores the symmetric third derivative.  A
    canonical monomial has coefficient ``multiplicity/6`` times that derivative.
    """

    model = StaticLowModePolynomial() if model is None else model
    result: SparsePolynomial = {}
    for monomial, derivative in model.third_derivative_terms.items():
        multiplicity = len(set(permutations(monomial)))
        _add_term(result, monomial, Fraction(multiplicity, 6) * derivative)
    return result


def exact_enstrophy_terms(model: StaticLowModePolynomial | None = None) -> SparsePolynomial:
    """Return the diagonal quadratic form ``E`` in canonical monomial form."""

    model = StaticLowModePolynomial() if model is None else model
    return {
        (index, index): coefficient / 2
        for index, coefficient in enumerate(model.enstrophy_diagonal_exact)
        if coefficient
    }


def diagonal_cauchy_stretching_bound(
    model: StaticLowModePolynomial | None = None,
) -> dict[str, Fraction]:
    """Return the exact monomial-diagonal Cauchy bound for ``A^2/E^3``.

    Write ``E^3=sum w_alpha z_alpha^2`` in the full degree-three monomial
    basis and ``A=sum a_alpha z_alpha``. Weighted Cauchy gives
    ``A^2 <= (sum a_alpha^2/w_alpha) E^3``. This is a valid but very loose
    bound; recording it exactly rules out a term-by-term diagonal proof of
    the sharp conjectured constant.
    """

    model = StaticLowModePolynomial() if model is None else model
    stretching = exact_stretching_cubic_terms(model)
    diagonal = model.enstrophy_diagonal_exact
    constant = Fraction(0)
    for monomial, coefficient in stretching.items():
        multiplicities = Counter(monomial)
        weight = Fraction(factorial(3))
        for index, multiplicity in multiplicities.items():
            weight /= factorial(multiplicity)
            weight *= (diagonal[index] / 2) ** multiplicity
        constant += coefficient * coefficient / weight
    return {
        "diagonal_cauchy_constant": constant,
        "sharp_target_constant": INVISCID_CUBIC_CONSTANT_SQUARED,
        "ratio_diagonal_over_sharp": constant / INVISCID_CUBIC_CONSTANT_SQUARED,
    }


def exact_inviscid_target_terms(model: StaticLowModePolynomial | None = None) -> SparsePolynomial:
    """Build ``(8832/42849)E^3-A^2`` exactly in sparse degree-six form."""

    model = StaticLowModePolynomial() if model is None else model
    energy = exact_enstrophy_terms(model)
    stretching = exact_stretching_cubic_terms(model)
    energy_cubed = _multiply(_multiply(energy, energy), energy)
    stretching_squared = _multiply(stretching, stretching)
    result: SparsePolynomial = dict(stretching_squared)
    for monomial in tuple(result):
        result[monomial] = -result[monomial]
    for monomial, coefficient in energy_cubed.items():
        _add_term(result, monomial, INVISCID_CUBIC_CONSTANT_SQUARED * coefficient)
    return result


def evaluate_sparse_polynomial(terms: Mapping[Monomial, Fraction], controls: Sequence[float]) -> float:
    """Evaluate a canonical sparse polynomial in floating point for audits."""

    x = np.asarray(controls, dtype=float)
    return float(
        sum(
            float(coefficient) * float(np.prod(x[np.asarray(monomial, dtype=int)]))
            for monomial, coefficient in terms.items()
        )
    )


def inviscid_target_metadata(model: StaticLowModePolynomial | None = None) -> dict[str, object]:
    """Return a compact reproducibility record for the exact target.

    The SHA-256 binds every sparse coefficient without pretending the target
    has been proved nonnegative.
    """

    model = StaticLowModePolynomial() if model is None else model
    stretching = exact_stretching_cubic_terms(model)
    target = exact_inviscid_target_terms(model)
    diagonal_bound = diagonal_cauchy_stretching_bound(model)
    encoded = "\n".join(
        f"{','.join(str(index) for index in monomial)}:{coefficient.numerator}/{coefficient.denominator}"
        for monomial, coefficient in sorted(target.items())
    ).encode("ascii")
    return {
        "truth_label": "exact_sparse_sos_target_not_a_nonnegativity_certificate",
        "dimension": model.dimension,
        "inequality_target": "A(x)^2 <= (8832/42849) E(x)^3",
        "constant_squared": "8832/42849",
        "constant": "8*sqrt(138)/207",
        "stretching_cubic_monomial_count": len(stretching),
        "degree_six_target_monomial_count": len(target),
        "degree_six_target_sha256": sha256(encoded).hexdigest(),
        "diagonal_cauchy_obstruction": {
            "bound": str(diagonal_bound["diagonal_cauchy_constant"]),
            "sharp_target": str(diagonal_bound["sharp_target_constant"]),
            "ratio": str(diagonal_bound["ratio_diagonal_over_sharp"]),
            "interpretation": (
                "A monomial-diagonal weighted Cauchy/SOS proof is too loose; "
                "a sharp proof must retain cross-triad cancellation or symmetry."
            ),
        },
        "next_required_step": (
            "Produce an independently checkable SOS, symmetry-block positivity, "
            "or rigorous global certificate for this exact polynomial."
        ),
        "claims": {
            "target_constructed_exactly": True,
            "target_nonnegative_proved": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }


def translation_sos_basis_structure(
    model: StaticLowModePolynomial | None = None,
) -> dict[str, object]:
    """Analyze the exact quarter-translation action on a sparse cubic basis.

    The basis comprises the 512 cubic stretching monomials and all repeated
    monomials ``x_i x_j^2`` needed to represent the diagonal ``E^3`` terms.
    It is closed under all 64 quarter-period translations. The resulting small
    orbits are useful for an eventual symmetry-adapted Gram certificate.
    This reports structure only, not an SOS or positivity result.
    """

    model = StaticLowModePolynomial() if model is None else model
    stretching = exact_stretching_cubic_terms(model)
    basis = set(stretching) | {
        tuple(sorted((first, second, second)))
        for first in range(model.dimension)
        for second in range(model.dimension)
    }

    actions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for shift_x in range(4):
        for shift_y in range(4):
            for shift_z in range(4):
                matrix = static_symmetry_matrix(
                    (IDENTITY_GENERATOR[0], (shift_x, shift_y, shift_z))
                )
                destinations: list[int] = []
                signs: list[int] = []
                for column in range(model.dimension):
                    rows = [
                        row
                        for row in range(model.dimension)
                        if matrix[row][column]
                    ]
                    if len(rows) != 1:
                        raise ArithmeticError("translation action is not signed-permutational")
                    row = rows[0]
                    destinations.append(row)
                    signs.append(int(matrix[row][column]))
                actions.append((tuple(destinations), tuple(signs)))

    transformed = {
        tuple(sorted(destination[index] for index in monomial))
        for destination, _ in actions
        for monomial in basis
    }
    if transformed != basis:
        raise ArithmeticError("sparse cubic basis is not closed under quarter translations")

    unseen = set(basis)
    orbit_histogram: dict[int, int] = {}
    while unseen:
        representative = unseen.pop()
        orbit = {
            tuple(sorted(destination[index] for index in representative))
            for destination, _ in actions
        }
        unseen -= orbit
        orbit_histogram[len(orbit)] = orbit_histogram.get(len(orbit), 0) + 1

    first_quarter_signs = {1: 0, -1: 0}
    for monomial in basis:
        _, signs = actions[16]  # (1,0,0) in the lexicographic shift order.
        sign = 1
        for index in monomial:
            sign *= signs[index]
        first_quarter_signs[sign] += 1
    return {
        "truth_label": "exact_translation_symmetry_structure_not_an_sos_certificate",
        "basis_definition": "stretching cubic monomials union {x_i x_j^2}",
        "basis_size": len(basis),
        "translation_group_size": len(actions),
        "basis_closed_under_all_quarter_translations": True,
        "translation_orbit_count": sum(orbit_histogram.values()),
        "translation_orbit_size_histogram": {
            str(size): count for size, count in sorted(orbit_histogram.items())
        },
        "first_quarter_translation_basis_sign_counts": {
            str(sign): count for sign, count in sorted(first_quarter_signs.items())
        },
        "next_required_step": (
            "Use these exact translation orbits to construct and certify a "
            "symmetry-adapted positive-semidefinite Gram matrix."
        ),
        "claims": {
            "exact_sparse_basis_translation_closed": True,
            "sharp_global_sos_certificate_constructed": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }
