"""Exact high-eta global comparison inside the six-amplitude static ansatz.

Together with the face exhaustion argument, the positive-coefficient
resultants here prove that the unique admissible interior branch cannot cross
any one-monomial face branch when eta=E0/nu**2 exceeds 1867/4.  The strict
ordering is anchored at eta=10**6 by the independent Arb certificate.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from .reduced_static_parameterized import UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD


Q = Fraction

# Coefficients in increasing powers of the interior dimensionless multiplier
# mu.  Each is the nontrivial factor of the exact resultant of
# (eta_interior(mu)-eta_face(t), rate_interior(mu)-rate_face(t)) with respect
# to the face multiplier t.  Strict positivity for mu>0 excludes a crossing.
FACE_CROSSING_RESULTANTS: Mapping[str, tuple[int, ...]] = {
    "acd": (
        33611390720,
        45891581952,
        26841508864,
        8786443264,
        1762843872,
        222383744,
        17260704,
        755136,
        14283,
    ),
    "aef": (
        35539083008,
        47603404800,
        27470241280,
        8909605888,
        1776452064,
        223188224,
        17280576,
        755136,
        14283,
    ),
    "bdf": (
        33012768512,
        45179445248,
        26501192192,
        8703757312,
        1752023008,
        221655040,
        17240832,
        755136,
        14283,
    ),
}


def interior_eta(mu: Q) -> Q:
    """Return eta along the admissible interior branch parameterized by mu."""

    mu = Q(mu)
    return (
        Q(69) * mu**4
        + Q(1696) * mu**3
        + Q(14936) * mu**2
        + Q(54656) * mu
        + Q(68368)
    ) / (Q(32) * (mu + Q(8)) ** 2)


def interior_dimensionless_rate(mu: Q) -> Q:
    """Return R/nu**3 along the interior branch."""

    mu = Q(mu)
    return (
        Q(23) * mu**5
        + Q(516) * mu**4
        + Q(3840) * mu**3
        + Q(8864) * mu**2
        - Q(8432) * mu
        - Q(33728)
    ) / (Q(16) * (mu + Q(8)) ** 2)


def face_eta_and_rate(name: str, multiplier: Q) -> tuple[Q, Q]:
    """Return eta and R/nu**3 on a one-monomial stationary face branch."""

    t = Q(multiplier)
    if name == "acd":
        return (
            Q(3) * (Q(3) * t * t + Q(24) * t + Q(44)) / Q(4),
            Q(3) * (t**3 + Q(6) * t * t - Q(24)) / Q(2),
        )
    if name == "aef":
        return (
            Q(3) * (Q(3) * t * t + Q(24) * t + Q(44)) / Q(2),
            Q(3) * (t**3 + Q(6) * t * t - Q(24)),
        )
    if name == "bdf":
        return (
            (t + Q(4)) * (Q(3) * t + Q(20)),
            Q(2) * (t + Q(4)) * (t * t + Q(4) * t - Q(16)),
        )
    raise ValueError(f"unknown one-monomial face {name!r}")


def high_eta_global_ansatz_theorem() -> dict[str, object]:
    """Return the exact logical ingredients of the parameter-uniform theorem."""

    return {
        "threshold": str(UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD),
        "interior": {
            "unique_admissible_positive_branch": True,
            "eta_of_mu": "(69 mu^4+1696 mu^3+14936 mu^2+54656 mu+68368)/(32(mu+8)^2)",
            "rate_over_nu_cubed": "(23 mu^5+516 mu^4+3840 mu^3+8864 mu^2-8432 mu-33728)/(16(mu+8)^2)",
        },
        "faces": {
            name: {
                "unique_positive_multiplier_for_each_eta_above_threshold": True,
                "crossing_resultant_low_to_high": [str(value) for value in polynomial],
                "all_nontrivial_resultant_coefficients_strictly_positive": all(
                    value > 0 for value in polynomial
                ),
            }
            for name, polynomial in FACE_CROSSING_RESULTANTS.items()
        },
        "connectedness_argument": (
            "The interior and each face branch are continuous and unique above the threshold. "
            "A rate equality would force its positive-coefficient resultant to vanish at mu>0, "
            "which is impossible. The independently Arb-certified eta=10^6 ordering fixes the "
            "strict interior-greater-than-face sign on the whole connected high-eta interval."
        ),
        "claims": {
            "global_static_maximum_within_six_amplitude_ansatz_for_eta_gt_1867_over_4": True,
            "full_64d_global_maximum": False,
            "navier_stokes_trajectory_or_regularity_statement": False,
        },
    }


def symmetry_branch_large_eta_efficiency() -> dict[str, object]:
    """Return the exact high-``eta`` efficiency comparison of both branches.

    The six-amplitude formulas are rational in its multiplier ``mu``.  Their
    leading terms give ``eta ~ (69/32) mu**2`` and
    ``R/nu**3 ~ (23/16) mu**3``.  The competing two-amplitude formula is
    ``(4/9)[(eta+1)**(3/2)-6 eta-1]``.  This returns an exact, readily
    auditable version of the resulting asymptotic comparison, rather than a
    floating-point reading of the branch plot.
    """

    six_squared = Q(64 * 138, 207 * 207)
    competing_squared = Q(16, 81)
    return {
        "normalization": "R/(nu^3 eta^(3/2))",
        "six_amplitude": {
            "eta_leading_coefficient_in_mu_squared": "69/32",
            "rate_leading_coefficient_in_mu_cubed": "23/16",
            "limit": "8*sqrt(138)/207",
            "limit_squared": str(six_squared),
        },
        "competing_two_amplitude": {
            "limit": "4/9",
            "limit_squared": str(competing_squared),
        },
        "strict_ordering": {
            "squared_difference_six_minus_competing": str(
                six_squared - competing_squared
            ),
            "six_amplitude_branch_eventually_higher": bool(
                six_squared > competing_squared
            ),
        },
        "scope": "comparison only along the two displayed symmetry branches",
        "full_64d_global_ordering": False,
    }


def six_amplitude_inviscid_extremizer() -> dict[str, object]:
    """Return the exact inviscid limit of the high-``eta`` six branch.

    At unit enstrophy, ``nu*mu -> sqrt(32/69)`` in the rational branch
    parameterization.  Substitution in the exact amplitude recovery formulas
    gives the displayed signed limit.  High-eta ansatz globality then promotes
    this limit to a global maximizer of the pure cubic stretching polynomial
    inside the six-amplitude fixed space.  It does *not* make a full
    64-coordinate cubic-globality assertion.
    """

    return {
        "normalization": "E=1, nu->0+ along the high-eta interior branch",
        "multiplier_limit": "nu*mu -> sqrt(32/69)",
        "signed_amplitudes_a_b_c_d_e_f": (
            "(sqrt(69)/23,-sqrt(690)/552,sqrt(345)/276,"
            "sqrt(690)/138,sqrt(69)/276,sqrt(138)/69)"
        ),
        "squared_amplitudes_a_b_c_d_e_f": {
            "a2": "3/23",
            "b2": "5/2208",
            "c2": "5/1104",
            "d2": "5/138",
            "e2": "1/1104",
            "f2": "2/69",
        },
        "invariants": {
            "enstrophy": "1",
            "stretching": "8*sqrt(138)/207",
            "palinstrophy": "148/69",
        },
        "globality_logic_inside_six_amplitude_space": (
            "For every unit-enstrophy y in the six-amplitude space, high-eta "
            "globality gives A(y)-2nu P(y)<=R_nu(x_nu). Let nu->0+. "
            "Finite dimensionality and the explicit branch limit give "
            "A(y)<=8*sqrt(138)/207."
        ),
        "claims": {
            "exact_inviscid_six_amplitude_cubic_maximizer": True,
            "global_maximum_within_six_amplitude_inviscid_cubic_problem": True,
            "full_64d_inviscid_cubic_global_maximum": False,
        },
    }


def symmetry_branch_crossover_asymptotics() -> dict[str, object]:
    """Return exact two-term large-``eta`` expansions for the branch rates.

    Put ``t=eta**(-1/2)``.  Inverting the rational six-branch relation gives
    ``mu=(4 sqrt(138)/69)t^-1-296/69+O(t)``.  Substitution into its exact rate
    rational function, and a binomial expansion of the competing closed form,
    yield the fields recorded here.  In particular the first two terms of the
    difference predict the otherwise non-obvious scale of the certified
    crossover without being used as a replacement for its interval proof.
    """

    return {
        "small_parameter": "t=eta^(-1/2)",
        "six_multiplier": {
            "t_inverse_coefficient": "4*sqrt(138)/69",
            "constant_coefficient": "-296/69",
        },
        "six_normalized_rate": (
            "8*sqrt(138)/207-(296/69)t+"
            "(1913*sqrt(138)/4761)t^2+O(t^3)"
        ),
        "competing_normalized_rate": "4/9-(8/3)t+(2/3)t^2+O(t^3)",
        "difference_six_minus_competing": (
            "8*sqrt(138)/207-4/9-(112/69)t+"
            "(1913*sqrt(138)/4761-2/3)t^2+O(t^3)"
        ),
        "two_term_crossover_predictor": {
            "formula": "7056/(23-2*sqrt(138))^2",
            "decimal": "28834.3012031766043515912662467",
            "meaning": (
                "zero of the constant-plus-t term only; it is explanatory, "
                "not a replacement for the interval-certified crossover"
            ),
        },
        "scope": "comparison only along the two displayed symmetry branches",
        "full_64d_global_ordering": False,
    }


__all__ = [
    "FACE_CROSSING_RESULTANTS",
    "face_eta_and_rate",
    "high_eta_global_ansatz_theorem",
    "interior_dimensionless_rate",
    "interior_eta",
    "six_amplitude_inviscid_extremizer",
    "symmetry_branch_large_eta_efficiency",
    "symmetry_branch_crossover_asymptotics",
]
