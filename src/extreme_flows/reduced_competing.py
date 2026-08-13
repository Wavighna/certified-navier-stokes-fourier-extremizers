"""Exact two-amplitude reduction for the competing static local branch.

The vectors below are expressed in the canonical 64-real-control basis used by
``StaticLowModePolynomial``. They were recognized from a certified full-space
KKT root, but every identity here is checked directly by rational Fourier
convolution. ``static_symmetry`` independently proves that this plane is the
complete common fixed space of two explicit lattice symmetries.
"""

from __future__ import annotations

from fractions import Fraction
from math import sqrt
from typing import Sequence

from .certify import QComplex, StaticLowModePolynomial, weighted_real_inner


PRIMARY_INDICES = (0, 13, 40)
SECONDARY_SIGNS = {
    11: -1,
    19: -1,
    31: -1,
    39: -1,
    47: 1,
    55: -1,
}


def embed_competing(a: Fraction, b: Fraction) -> tuple[Fraction, ...]:
    """Return the exact 64-control vector with coordinates ``(a,b)``."""

    controls = [Fraction(0) for _ in range(64)]
    for index in PRIMARY_INDICES:
        controls[index] = a
    for index, sign in SECONDARY_SIGNS.items():
        controls[index] = Fraction(sign) * b
    return tuple(controls)


def reduced_invariants(a: Fraction, b: Fraction, viscosity: Fraction) -> dict[str, Fraction]:
    """Exact restricted invariants and static rate.

    ``a`` can be sign-gauged nonnegative; the rate-maximizing branch has
    ``b>0``.  The formula is an exact identity on this specified linear plane.
    """

    enstrophy = 3 * a * a + 48 * b * b
    palinstrophy = 3 * a * a + 96 * b * b
    stretching = 24 * a * a * b
    return {
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "stretching": stretching,
        "rate": stretching - 2 * viscosity * palinstrophy,
    }


def optimizer_formula(enstrophy: float, viscosity: float) -> dict[str, float]:
    """The unique global rate maximizer on the sign-gauged two-amplitude plane.

    Eliminating ``a`` by ``E=3a²+48b²`` leaves a strictly concave cubic on
    ``0 <= b <= sqrt(E/48)``.  Its only critical point is therefore the global
    maximizer on that interval.
    """

    if enstrophy <= 0 or viscosity <= 0:
        raise ValueError("enstrophy and viscosity must be positive")
    b = (sqrt(enstrophy + viscosity * viscosity) - viscosity) / 12.0
    a_squared = (enstrophy - 48.0 * b * b) / 3.0
    if a_squared <= 0:
        raise ArithmeticError("derived interior branch is not admissible")
    a = sqrt(a_squared)
    stretching = 24.0 * a_squared * b
    palinstrophy = 3.0 * a_squared + 96.0 * b * b
    return {
        "a": a,
        "b": b,
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "stretching": stretching,
        "rate": stretching - 2.0 * viscosity * palinstrophy,
    }


def optimized_rate_formula(enstrophy: float, viscosity: float) -> float:
    """Closed form for the two-amplitude constrained optimum.

    Substitution of ``optimizer_formula`` into the reduced static rate gives

    ``R* = 4/9 * ((E+nu²)^(3/2) - 6 E nu - nu³)``.

    For positive enstrophy, this branch has positive rate exactly when
    ``E/nu² > (33 + 15 sqrt(5))/2``.  This is a statement on the
    two-amplitude plane, not on the unrestricted Fourier class.
    """

    if enstrophy <= 0 or viscosity <= 0:
        raise ValueError("enstrophy and viscosity must be positive")
    return (4.0 / 9.0) * (
        (enstrophy + viscosity * viscosity) ** 1.5
        - 6.0 * enstrophy * viscosity
        - viscosity**3
    )


def exact_formula_matches_full_polynomial(
    samples: Sequence[tuple[Fraction, Fraction]], viscosity: Fraction
) -> bool:
    """Check formula values against independent exact 64-mode convolution."""

    model = StaticLowModePolynomial(viscosity=viscosity)
    return all(
        model.exact_invariants(embed_competing(a, b))
        == reduced_invariants(a, b, viscosity)
        for a, b in samples
    )


def exact_formula_coefficients_from_full_polynomial(
    viscosity: Fraction,
) -> dict[str, tuple[Fraction, ...]]:
    """Recover the restricted coefficients directly from the 64-mode model.

    Enstrophy and palinstrophy are homogeneous quadratics in ``(a,b)``;
    stretching is homogeneous cubic. Exact evaluation at four integer points
    determines every coefficient, making this an audit of the complete
    restricted polynomial identity rather than a numerical spot check.

    Coefficient order is ``(a², ab, b²)`` for quadratic invariants and
    ``(a³, a²b, ab², b³)`` for stretching.
    """

    model = StaticLowModePolynomial(viscosity=viscosity)

    def values(a: int, b: int) -> dict[str, Fraction]:
        return model.exact_invariants(embed_competing(Fraction(a), Fraction(b)))

    at10, at01, at11, at1m1 = values(1, 0), values(0, 1), values(1, 1), values(1, -1)

    def quadratic(name: str) -> tuple[Fraction, Fraction, Fraction]:
        return (
            at10[name],
            at11[name] - at10[name] - at01[name],
            at01[name],
        )

    c30, c03 = at10["stretching"], at01["stretching"]
    sum_mixed = at11["stretching"] - c30 - c03
    difference_mixed = at1m1["stretching"] - c30 + c03
    return {
        "enstrophy": quadratic("enstrophy"),
        "palinstrophy": quadratic("palinstrophy"),
        "stretching": (
            c30,
            (sum_mixed - difference_mixed) / 2,
            (sum_mixed + difference_mixed) / 2,
            c03,
        ),
    }


def exact_formula_coefficients_match_full_polynomial(viscosity: Fraction) -> bool:
    """Return whether every restricted polynomial coefficient matches exactly."""

    return exact_formula_coefficients_from_full_polynomial(viscosity) == {
        "enstrophy": (Fraction(3), Fraction(0), Fraction(48)),
        "palinstrophy": (Fraction(3), Fraction(0), Fraction(96)),
        "stretching": (Fraction(0), Fraction(24), Fraction(0), Fraction(0)),
    }


def exact_energy_helicity_coefficients_from_full_polynomial() -> dict[str, tuple[Fraction, ...]]:
    """Recover kinetic-energy and helicity coefficients from the full field.

    The returned order is (a squared, ab, b squared). The calculation uses an
    exact Fourier curl and is independent of the reduced invariant formula.
    """

    model = StaticLowModePolynomial()
    imaginary = QComplex(Fraction(0), Fraction(1))

    def values(a: int, b: int) -> tuple[Fraction, Fraction]:
        field = model.exact_field(embed_competing(Fraction(a), Fraction(b)))
        curl = {
            k: tuple(
                imaginary * value
                for value in (
                    k[1] * coefficient[2] - k[2] * coefficient[1],
                    k[2] * coefficient[0] - k[0] * coefficient[2],
                    k[0] * coefficient[1] - k[1] * coefficient[0],
                )
            )
            for k, coefficient in field.items()
        }
        return (
            weighted_real_inner(field, field, 0) / 2,
            weighted_real_inner(field, curl, 0),
        )

    at10, at01, at11 = values(1, 0), values(0, 1), values(1, 1)

    def coefficients(component: int) -> tuple[Fraction, Fraction, Fraction]:
        return (
            at10[component],
            at11[component] - at10[component] - at01[component],
            at01[component],
        )

    return {"kinetic_energy": coefficients(0), "helicity": coefficients(1)}


def exact_energy_helicity_match_full_polynomial() -> bool:
    """Check the exact zero-helicity two-amplitude structure."""

    return exact_energy_helicity_coefficients_from_full_polynomial() == {
        "kinetic_energy": (Fraction(3), Fraction(0), Fraction(24)),
        "helicity": (Fraction(0), Fraction(0), Fraction(0)),
    }


def exact_shell_support() -> dict[str, tuple[int, ...]]:
    """Return the exact wave-square shells occupied by each branch amplitude."""

    model = StaticLowModePolynomial()
    return {
        "a": tuple(sorted({sum(value * value for value in model.pairs[index // 4]) for index in PRIMARY_INDICES})),
        "b": tuple(sorted({sum(value * value for value in model.pairs[index // 4]) for index in SECONDARY_SIGNS})),
    }
