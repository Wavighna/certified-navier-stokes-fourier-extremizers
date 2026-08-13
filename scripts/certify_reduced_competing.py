"""Record exact algebra for the two-amplitude competing static branch."""

from __future__ import annotations

import json
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from extreme_flows.reduced_competing import (
    SECONDARY_SIGNS,
    PRIMARY_INDICES,
    exact_formula_coefficients_from_full_polynomial,
    exact_formula_coefficients_match_full_polynomial,
    exact_energy_helicity_coefficients_from_full_polynomial,
    exact_energy_helicity_match_full_polynomial,
    exact_shell_support,
    exact_formula_matches_full_polynomial,
    optimizer_formula,
    optimized_rate_formula,
)
from extreme_flows.static_symmetry import (
    COMPETING_ANSATZ_GENERATORS,
    competing_ansatz_fixed_space_dimension,
    reduced_competing_ansatz_is_exact_fixed_space,
)


def exact_full_kkt_reduction() -> dict[str, object]:
    """Reduce all 64 full KKT equations for symbolic positive ``E, nu``."""

    import sympy as sp
    from itertools import permutations

    from extreme_flows.certify import StaticLowModePolynomial
    from extreme_flows.reduced_competing import embed_competing

    model = StaticLowModePolynomial(
        viscosity=Fraction(1, 10), phase_polarizations=(0, 1, 0)
    )
    a, b, enstrophy, viscosity = sp.symbols("a b E nu", positive=True)
    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    controls = [
        sp.Rational(x.numerator, x.denominator) * a
        + sp.Rational(y.numerator, y.denominator) * b
        for x, y in zip(primary, secondary, strict=True)
    ]
    # Assemble the exact full gradient straight from the sparse cubic tensor.
    gradient = [
        -2 * viscosity
        * sp.Rational(value.numerator, value.denominator)
        * controls[index]
        for index, value in enumerate(model.palinstrophy_diagonal_exact)
    ]
    for indices, value in model.third_derivative_terms.items():
        coefficient = sp.Rational(value.numerator, value.denominator)
        for i, j, k in set(permutations(indices)):
            gradient[i] += coefficient * controls[j] * controls[k] / 2
    e_diagonal = [
        sp.Rational(value.numerator, value.denominator)
        for value in model.enstrophy_diagonal_exact
    ]
    multiplier = 8 * b - 2 * viscosity
    ideal = sp.groebner(
        [
            3 * a * a + 48 * b * b - enstrophy,
            144 * b * b + 24 * viscosity * b - enstrophy,
        ],
        a,
        b,
        order="lex",
        domain=sp.QQ.frac_field(enstrophy, viscosity),
    )
    stationarity_remainders = [
        ideal.reduce(sp.expand(value - multiplier * e_diagonal[index] * controls[index]))[1]
        for index, value in enumerate(gradient)
    ]
    enstrophy_remainder = ideal.reduce(
        sum(e_diagonal[index] * value * value for index, value in enumerate(controls))
        / 2 - enstrophy
    )[1]
    return {
        "basis": "canonical 64-control exact Fourier polynomial",
        "phase_chart": {"waves": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "polarizations": [0, 1, 0]},
        "kkt_multiplier": "lambda=8*b-2*nu; three phase multipliers vanish",
        "ideal_generators": ["3*a^2+48*b^2-E", "144*b^2+24*nu*b-E"],
        "stationarity_remainder_count": sum(value != 0 for value in stationarity_remainders),
        "enstrophy_constraint_remainder": str(enstrophy_remainder),
        "all_full_kkt_remainders_vanish": all(value == 0 for value in stationarity_remainders)
        and enstrophy_remainder == 0,
    }


def exact_optimized_rate_reduction() -> dict[str, object]:
    """Verify the closed optimum-rate formula and its sign threshold symbolically."""

    import sympy as sp

    eta, enstrophy, viscosity = sp.symbols("eta E nu", positive=True)
    root = sp.sqrt(enstrophy + viscosity**2)
    b = (root - viscosity) / 12
    reduced_rate = 8 * enstrophy * b - 384 * b**3 - 2 * viscosity * enstrophy - 96 * viscosity * b**2
    closed_rate = sp.Rational(4, 9) * (
        (enstrophy + viscosity**2) ** sp.Rational(3, 2)
        - 6 * enstrophy * viscosity
        - viscosity**3
    )
    dimensionless = sp.simplify(
        (closed_rate / viscosity**3).subs(enstrophy, eta * viscosity**2)
    )
    y = sp.symbols("y", positive=True)
    factored_y = sp.factor(
        ((eta + 1) ** sp.Rational(3, 2) - 6 * eta - 1).subs(eta, y**2 - 1)
    )
    threshold = (33 + 15 * sp.sqrt(5)) / 2
    return {
        "substitution_remainder": str(sp.simplify(reduced_rate - closed_rate)),
        "dimensionless_rate_times_9_over_4": str(dimensionless * sp.Rational(9, 4)),
        "factor_after_y_equals_sqrt_eta_plus_1": str(factored_y),
        "positive_eta_threshold": str(threshold),
        "threshold_proved": (
            sp.simplify(reduced_rate - closed_rate) == 0
            and sp.expand(factored_y) == sp.expand((y - 1) * (y**2 - 5 * y - 5))
            and sp.simplify(
                ((5 + 3 * sp.sqrt(5)) / 2) ** 2 - 1 - threshold
            )
            == 0
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = ((Fraction(1), Fraction(1)), (Fraction(2), Fraction(1)), (Fraction(1), Fraction(3)))
    viscosity = Fraction(1, 10)
    if not exact_formula_matches_full_polynomial(samples, viscosity):
        raise AssertionError("exact full-polynomial check failed")
    if not exact_formula_coefficients_match_full_polynomial(viscosity):
        raise AssertionError("restricted exact polynomial coefficient check failed")
    if not exact_energy_helicity_match_full_polynomial():
        raise AssertionError("restricted exact energy/helicity check failed")
    kkt_reduction = exact_full_kkt_reduction()
    if not kkt_reduction["all_full_kkt_remainders_vanish"]:
        raise AssertionError("exact full KKT reduction failed")
    rate_reduction = exact_optimized_rate_reduction()
    if not rate_reduction["threshold_proved"]:
        raise AssertionError("closed optimized-rate reduction failed")
    fixed_space_dimension = competing_ansatz_fixed_space_dimension()
    fixed_space_exact = reduced_competing_ansatz_is_exact_fixed_space()
    if fixed_space_dimension != 2 or not fixed_space_exact:
        raise AssertionError("competing ansatz is not the claimed exact fixed space")
    optimum = optimizer_formula(100.0, 0.1)
    source = root / "src" / "extreme_flows" / "reduced_competing.py"
    symmetry_source = root / "src" / "extreme_flows" / "static_symmetry.py"
    certificate = root / "artifacts" / "proofs" / "high_branch_eta10000_arb_certificate.json"
    certificate_payload = json.loads(certificate.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "truth_label": "exact_two_amplitude_symmetry_formula_and_parameter_uniform_full_kkt_branch",
        "scope": "global only on the displayed sign-gauged two-amplitude plane",
        "embedding": {
            "primary_indices_with_coefficient_a": list(PRIMARY_INDICES),
            "secondary_indices_with_signed_coefficient_b": SECONDARY_SIGNS,
        },
        "spectral_support": {
            "a_wave_square_shells": list(exact_shell_support()["a"]),
            "b_wave_square_shells": list(exact_shell_support()["b"]),
            "interpretation": "an exact nonhelical coupling of the |k|^2=1 and |k|^2=2 shells",
        },
        "symmetry_fixed_space": {
            "generators": [
                {"Q": [list(row) for row in matrix], "quarter_period_shift_m": list(shift)}
                for matrix, shift in COMPETING_ANSATZ_GENERATORS
            ],
            "exact_common_fixed_space_dimension": fixed_space_dimension,
            "embedding_spans_exact_common_fixed_space": fixed_space_exact,
        },
        "exact_formulas": {
            "stretching": "24*a^2*b",
            "enstrophy": "3*a^2 + 48*b^2",
            "palinstrophy": "3*a^2 + 96*b^2",
            "rate": "24*a^2*b - 2*nu*(3*a^2 + 96*b^2)",
        },
        "maximization": {
            "reduced_rate_after_eliminating_a": "8*E*b - 384*b^3 - 2*nu*E - 96*nu*b^2",
            "critical_point": "b=(sqrt(E+nu^2)-nu)/12",
            "derivative": "8*E - 1152*b^2 - 192*nu*b",
            "strict_concavity": "-2304*b - 192*nu < 0 for b>=0, nu>0",
            "endpoint_signs": "f'(0)=8E>0 and f'(sqrt(E/48))<0",
            "conclusion": "the stated critical point is the unique global maximum on 0<=b<=sqrt(E/48)",
            "optimized_rate_closed_form": "(4/9)*((E+nu^2)^(3/2)-6*E*nu-nu^3)",
            "positive_rate_exactly_when": "E/nu^2 > (33+15*sqrt(5))/2 for E,nu>0",
            "symbolic_rate_reduction": rate_reduction,
        },
        "eta_10000_point": {**optimum, "closed_form_rate": optimized_rate_formula(100.0, 0.1)},
        "relation_to_full_certificate": {
            "high_branch_certificate_sha256": sha256(certificate.read_bytes()).hexdigest(),
            "strictly_beats_six_amplitude_ansatz": certificate_payload["claims"][
                "strictly_beats_six_amplitude_ansatz"
            ],
        },
        "exact_full_polynomial_samples": [[str(a), str(b)] for a, b in samples],
        "exact_restricted_polynomial_coefficients_recovered_from_full_64_mode_convolution": {
            name: [str(value) for value in coefficients]
            for name, coefficients in exact_formula_coefficients_from_full_polynomial(viscosity).items()
        },
        "exact_auxiliary_coefficients_recovered_from_full_64_mode_convolution": {
            name: [str(value) for value in coefficients]
            for name, coefficients in exact_energy_helicity_coefficients_from_full_polynomial().items()
        },
        "exact_full_kkt_reduction_for_symbolic_positive_E_and_nu": kkt_reduction,
        "source": {
            "reduced_competing": {
                "path": "src/extreme_flows/reduced_competing.py",
                "sha256": sha256(source.read_bytes()).hexdigest(),
            },
            "static_symmetry": {
                "path": "src/extreme_flows/static_symmetry.py",
                "sha256": sha256(symmetry_source.read_bytes()).hexdigest(),
            },
        },
        "limitations": {
            "full_64d_global_maximum": False,
            "symmetry_fixed_space_characterization": True,
            "navier_stokes_trajectory_or_regularity_claim": False,
        },
    }
    output = root / "artifacts" / "proofs" / "reduced_competing_formula.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"rate={optimum['rate']:.15f}")


if __name__ == "__main__":
    main()
