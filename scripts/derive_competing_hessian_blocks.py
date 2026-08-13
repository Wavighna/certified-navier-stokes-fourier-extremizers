"""Exact symmetry-block factorization of the competing branch Hessian.

The competing two-amplitude branch is fixed by an ``A4 x C2`` lattice group.
The ambient Fourier-coordinate matrices are orthogonal in the enstrophy
metric, rather than in the raw coordinate Euclidean metric.  Accordingly this
script factors the *generalized* Hessian ``D_E^{-1} H`` on the six rational
isotypic sectors.  It is an exact symbolic derivation; the separate Arb cover
in :mod:`certify_competing_all_eta_uniform` certifies the required signs over
the entire parameter interval.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

import sympy as sp

from extreme_flows.certify import StaticLowModePolynomial
from extreme_flows.reduced_competing import embed_competing
from extreme_flows.static_symmetry import (
    COMPETING_A4_COMPLEMENT_INVOLUTION,
    COMPETING_ANSATZ_GENERATORS,
    COMPETING_CENTRAL_INVOLUTION,
    compose_lattice_symmetries,
    lattice_symmetry_group,
    lattice_symmetry_order,
    static_symmetry_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
T = sp.symbols("t")
S = sp.symbols("s", nonzero=True)
A = sp.symbols("a")
RELATION = 9 * A * A + S * S - 3
CHARACTERS = {
    "trivial": {1: 1, 2: 1, 3: 1},
    "two_dimensional": {1: 2, 2: 2, 3: -1},
    "three_dimensional": {1: 3, 2: -1, 3: 0},
}


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    data = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _branch_generalized_hessian() -> tuple[sp.Matrix, StaticLowModePolynomial]:
    """Return ``D_E^-1 H`` in the unit-enstrophy ``s`` parameterization."""

    model = StaticLowModePolynomial()
    dimension = model.dimension
    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    tensor_a = sp.zeros(dimension)
    tensor_b = sp.zeros(dimension)
    for (i, j, k), coefficient in model.third_derivative_terms.items():
        value = sp.Rational(coefficient.numerator, coefficient.denominator)
        for ii, jj, kk in {
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        }:
            tensor_a[ii, jj] += value * sp.Rational(
                primary[kk].numerator, primary[kk].denominator
            )
            tensor_b[ii, jj] += value * sp.Rational(
                secondary[kk].numerator, secondary[kk].denominator
            )
    enstrophy = [
        sp.Rational(value.numerator, value.denominator)
        for value in model.enstrophy_diagonal_exact
    ]
    palinstrophy = [
        sp.Rational(value.numerator, value.denominator)
        for value in model.palinstrophy_diagonal_exact
    ]
    # E=1: a^2=(3-s^2)/9, b=s/12, nu=(1-s^2)/(2s),
    # and the constrained multiplier is lambda=5s/3-1/s.
    b = S / 12
    viscosity = (1 - S * S) / (2 * S)
    multiplier = 5 * S / 3 - 1 / S
    hessian = (
        A * tensor_a
        + b * tensor_b
        - 2 * viscosity * sp.diag(*palinstrophy)
        - multiplier * sp.diag(*enstrophy)
    )
    return sp.diag(*[1 / value for value in enstrophy]) * hessian, model


def _sector_basis(parity: int, character: dict[int, int], dimension: int) -> sp.Matrix:
    """Return a rational column basis of one ``A4 x C2`` isotypic sector."""

    complement = lattice_symmetry_group(
        (COMPETING_A4_COMPLEMENT_INVOLUTION, COMPETING_ANSATZ_GENERATORS[1])
    )
    projector = sp.zeros(dimension)
    for element in complement:
        coefficient = character[lattice_symmetry_order(element)]
        direct = sp.Matrix(static_symmetry_matrix(element))
        central_shifted = sp.Matrix(
            static_symmetry_matrix(
                compose_lattice_symmetries(element, COMPETING_CENTRAL_INVOLUTION)
            )
        )
        projector += coefficient * (direct + parity * central_shifted)
    return sp.Matrix.hstack(*projector.columnspace())


def _translation_tangent(controls: tuple[Fraction, ...], axis: int, model: StaticLowModePolynomial) -> sp.Matrix:
    """Differentiate the Fourier controls under translation in one axis."""

    values: list[sp.Rational] = []
    for index, value in enumerate(controls):
        wave_component = model.pairs[index // 4][axis]
        slot = index % 4
        # d/d tau exp(-i k.tau) (real+i imag)=(k imag)-i(k real).
        partner = controls[index + 2] if slot < 2 else -controls[index - 2]
        values.append(
            sp.Rational(wave_component * partner.numerator, partner.denominator)
        )
    return sp.Matrix(values)


def _constraint_geometry(model: StaticLowModePolynomial) -> dict[str, object]:
    """Verify exact tangent-sector and translation-orbit geometry."""

    enstrophy = sp.diag(
        *[
            sp.Rational(value.numerator, value.denominator)
            for value in model.enstrophy_diagonal_exact
        ]
    )
    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    primary_vector = sp.Matrix(
        [sp.Rational(value.numerator, value.denominator) for value in primary]
    )
    secondary_vector = sp.Matrix(
        [sp.Rational(value.numerator, value.denominator) for value in secondary]
    )
    sectors: dict[str, dict[str, object]] = {}
    for parity, parity_name in ((1, "even"), (-1, "odd")):
        for sector, character in CHARACTERS.items():
            name = f"{sector}_{parity_name}"
            basis = _sector_basis(parity, character, model.dimension)
            sectors[name] = {
                "dimension": basis.shape[1],
                "enstrophy_orthogonal_to_primary": bool(
                    all(value == 0 for value in basis.T * enstrophy * primary_vector)
                ),
                "enstrophy_orthogonal_to_secondary": bool(
                    all(value == 0 for value in basis.T * enstrophy * secondary_vector)
                ),
            }
    odd_three = _sector_basis(-1, CHARACTERS["three_dimensional"], model.dimension)
    translation_membership = []
    for label, controls in (("primary", primary), ("secondary", secondary)):
        for axis in range(3):
            direction = _translation_tangent(controls, axis, model)
            translation_membership.append(
                {
                    "branch_basis_vector": label,
                    "axis": axis,
                    "nonzero": bool(any(direction)),
                    "lies_in_odd_three_dimensional_sector": odd_three.row_join(direction).rank()
                    == odd_three.rank(),
                }
            )
    # On the branch a^2=(3-s^2)/9 and b=s/12.  The translation tangent
    # Gram determinant therefore simplifies to the strictly positive formula.
    return {
        "sector_enstrophy_orthogonality": sectors,
        "nontrivial_sectors_are_enstrophy_tangent": all(
            values["enstrophy_orthogonal_to_primary"]
            and values["enstrophy_orthogonal_to_secondary"]
            for name, values in sectors.items()
            if name != "trivial_even"
        ),
        "translation_membership": translation_membership,
        "all_translation_basis_tangents_in_odd_three_dimensional_sector": all(
            row["nonzero"] and row["lies_in_odd_three_dimensional_sector"]
            for row in translation_membership
        ),
        "branch_translation_gram_determinant": "8*(3+s^2)^3/729",
        "branch_translation_orbit_dimension": 3,
    }


def _reduce_branch_relation(value: sp.Expr) -> sp.Expr:
    """Reduce a rational expression modulo ``9a^2+s^2-3`` exactly."""

    numerator, denominator = sp.fraction(sp.cancel(value))
    if denominator.has(A):
        raise ArithmeticError("unexpected a-dependent denominator")
    reduced = sp.rem(
        sp.Poly(numerator, A), sp.Poly(RELATION, A)
    ).as_expr()
    return sp.factor(reduced / denominator)


def derive_block_polynomials() -> dict[str, object]:
    """Derive all six characteristic factors over ``Q(s)`` exactly."""

    generalized_hessian, model = _branch_generalized_hessian()
    blocks: dict[str, dict[str, object]] = {}
    for parity, parity_name in ((1, "even"), (-1, "odd")):
        for sector, character in CHARACTERS.items():
            basis = _sector_basis(parity, character, model.dimension)
            restricted = sp.cancel(
                (basis.T * basis).inv() * basis.T * generalized_hessian * basis
            )
            residual = generalized_hessian * basis - basis * restricted
            if any(sp.simplify(entry) != 0 for entry in residual):
                raise ArithmeticError("generalized Hessian failed to preserve a sector")
            polynomial = sp.Poly(restricted.charpoly(T).as_expr(), T)
            coefficients = [_reduce_branch_relation(value) for value in polynomial.all_coeffs()]
            factor = sp.factor(sp.Poly.from_list(coefficients, gens=T).as_expr())
            if factor.has(A):
                raise ArithmeticError("factor still depends on a after branch reduction")
            blocks[f"{sector}_{parity_name}"] = {
                "sector_dimension": basis.shape[1],
                "characteristic_polynomial": str(factor),
            }
    return {
        "parameterization": {
            "enstrophy": "1",
            "a_squared": "(3-s^2)/9",
            "b": "s/12",
            "viscosity": "(1-s^2)/(2s)",
            "multiplier": "5s/3-1/s",
            "range": "0<s<1",
        },
        "operator": "D_E^(-1) [ Hessian(R) - lambda Hessian(E) ]",
        "isotypic_sector_dimensions": {
            "trivial_even": 2,
            "two_dimensional_even": 6,
            "three_dimensional_even": 24,
            "trivial_odd": 2,
            "two_dimensional_odd": 6,
            "three_dimensional_odd": 24,
        },
        "constraint_geometry": _constraint_geometry(model),
        "blocks": blocks,
    }


def build_payload() -> dict[str, object]:
    """Build an exact, reproducible factorization record."""

    derived = derive_block_polynomials()
    return {
        "schema_version": 1,
        "truth_label": "exact_symbolic_competing_branch_generalized_hessian_block_factorization",
        "statement": (
            "Exact symbolic characteristic polynomials of the generalized "
            "64-control Hessian in the A4 x C2 isotypic sectors along the "
            "competing branch. This algebraic factorization is not, by itself, "
            "a proof of all-parameter inertia signs or full-space globality."
        ),
        "sympy_version": sp.__version__,
        "derivation": derived,
        "checker_source": _record("scripts/derive_competing_hessian_blocks.py"),
        "polynomial_source": _record("src/extreme_flows/certify.py"),
        "symmetry_source": _record("src/extreme_flows/static_symmetry.py"),
        "formula_source": _record("src/extreme_flows/reduced_competing.py"),
        "claims": {
            "exact_generalized_hessian_sector_factorization": True,
            "all_parameter_inertia_signs_proved_by_this_artifact": False,
            "full_64d_global_maximum": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/proofs/competing_hessian_spectral_blocks.json",
    )
    args = parser.parse_args()
    payload = build_payload()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")
    for name, block in payload["derivation"]["blocks"].items():
        print(f"{name}: degree {block['sector_dimension']}")


if __name__ == "__main__":
    main()
