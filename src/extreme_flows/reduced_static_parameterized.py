"""Exact parameterized interior KKT reduction for the six-amplitude ansatz.

The static rate is cubic-minus-quadratic.  After fixing enstrophy, its only
dimensionless parameter is eta = E / nu**2.  The interior KKT equations reduce
to one quartic in mu = lambda / nu.  This module contains only exact rational
algebra and the elementary high-eta branch theorem; it does not assert global
maximality outside the separately certified normalization.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


Q = Fraction

# eta > 14936 / 32 makes the dimensionless quartic have coefficient signs
# (+,+,-,-,-), hence exactly one positive root by Descartes plus IVT.
UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD = Q(1867, 4)


def dimensionless_interior_quartic(eta: Q) -> tuple[Q, Q, Q, Q, Q]:
    """Return F_eta(mu), in increasing powers of the dimensionless multiplier.

    F_eta(mu) = 0 is the interior enstrophy constraint after eliminating
    b, c, and e from the six KKT equations, where eta=E/nu**2 and mu=lambda/nu.
    """

    eta = Q(eta)
    return (
        Q(68368) - Q(2048) * eta,
        Q(54656) - Q(512) * eta,
        Q(14936) - Q(32) * eta,
        Q(1696),
        Q(69),
    )


def interior_multiplier_quartic(
    viscosity: Q, target_enstrophy: Q
) -> tuple[Q, Q, Q, Q, Q]:
    """Return the exact lambda-quartic in increasing powers of lambda."""

    viscosity = Q(viscosity)
    target_enstrophy = Q(target_enstrophy)
    if viscosity <= 0 or target_enstrophy <= 0:
        raise ValueError("viscosity and target enstrophy must be positive")
    nu = viscosity
    energy = target_enstrophy
    return (
        Q(68368) * nu**4 - Q(2048) * energy * nu**2,
        Q(54656) * nu**3 - Q(512) * energy * nu,
        Q(14936) * nu**2 - Q(32) * energy,
        Q(1696) * nu,
        Q(69),
    )


def interior_squared_amplitudes(mu: Q, viscosity: Q) -> dict[str, Q]:
    """Return a²,d²,f² on the dimensionless interior KKT branch.

    The remaining amplitudes are recovered from
    b=-df/(lambda+8nu), c=2ad/[3(lambda+6nu)], and
    e=af/[3(lambda+6nu)].
    """

    mu = Q(mu)
    viscosity = Q(viscosity)
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    prefactor = viscosity * viscosity
    return {
        "a2": prefactor
        * Q(3)
        * (mu + Q(6))
        * (Q(3) * mu * mu + Q(48) * mu + Q(156))
        / (Q(32) * (mu + Q(8))),
        "d2": prefactor
        * (Q(5) * mu * mu + Q(48) * mu + Q(100))
        / Q(64),
        "f2": prefactor * (mu * mu - Q(28)) / Q(16),
    }


def high_eta_interior_branch_theorem(eta: Q) -> dict[str, object]:
    """Return the exact elementary proof conditions for the high-eta branch.

    For eta>1867/4, F_eta has sign pattern (+,+,-,-,-) in descending
    powers, so Descartes gives at most one positive root. Its negative constant
    term and positive leading coefficient give existence. Furthermore
    F_eta(sqrt(28)) is strictly negative, so that root exceeds sqrt(28) and
    all three squared-amplitude formulas are positive.
    """

    eta = Q(eta)
    threshold_met = eta > UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD
    coefficients = dimensionless_interior_quartic(eta)
    signs_descending = tuple(
        1 if value > 0 else -1 if value < 0 else 0 for value in reversed(coefficients)
    )
    return {
        "eta": str(eta),
        "threshold": str(UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD),
        "threshold_met": threshold_met,
        "dimensionless_quartic_low_to_high": [str(value) for value in coefficients],
        "descending_sign_pattern": list(signs_descending),
        "descartes_positive_root_count": 1 if threshold_met else None,
        "positive_root_exists_by_sign_change": bool(threshold_met),
        "root_exceeds_sqrt_28": bool(threshold_met),
        "positive_a2_d2_f2": bool(threshold_met),
        "sqrt_28_bound": (
            "F_eta(sqrt(28)) <= -833440-273664*sqrt(7) < 0 for "
            "eta >= 1867/4"
            if threshold_met
            else None
        ),
        "claim": (
            "unique admissible positive interior KKT branch"
            if threshold_met
            else "no high-eta branch conclusion"
        ),
        "global_maximum_statement": False,
    }


__all__ = [
    "UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD",
    "dimensionless_interior_quartic",
    "high_eta_interior_branch_theorem",
    "interior_multiplier_quartic",
    "interior_squared_amplitudes",
]
