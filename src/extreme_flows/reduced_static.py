"""Exact six-amplitude static ansatz for low-mode enstrophy production.

The numerical 64-real-variable KKT representative used to seed the certified
local root is supported, to its displayed tolerance, in this explicit
coordinate subspace.  The formulas here are finite-dimensional Fourier
algebra, not a Navier--Stokes trajectory reduction: the ansatz is introduced only for the static functional
``R = A - 2 nu P``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


Amplitude = Fraction | float


# A six-parameter embedding into StaticLowModePolynomial's canonical 64
# controls.  The signs are part of the definition.  They give the exact cubic
# formula in ``reduced_static_invariants`` below.
_EMBEDDING: dict[int, tuple[int, int]] = {
    0: (0, 1),
    1: (0, -1),
    20: (1, 1),
    60: (1, 1),
    26: (2, -1),
    27: (2, -1),
    34: (2, -1),
    35: (2, 1),
    29: (3, -1),
    48: (4, -1),
    49: (4, 1),
    56: (4, 1),
    57: (4, 1),
    54: (5, -1),
}


def embed_reduced_static(amplitudes: Sequence[Amplitude]) -> tuple[Amplitude, ...]:
    """Embed ``(a,b,c,d,e,f)`` into the canonical 64-real low-mode basis."""

    if len(amplitudes) != 6:
        raise ValueError("expected the six amplitudes (a,b,c,d,e,f)")
    zero: Amplitude = Fraction(0) if all(
        isinstance(value, Fraction) for value in amplitudes
    ) else 0.0
    controls: list[Amplitude] = [zero] * 64
    for index, (amplitude_index, sign) in _EMBEDDING.items():
        controls[index] = sign * amplitudes[amplitude_index]
    return tuple(controls)


def reduced_static_invariants(
    amplitudes: Sequence[Amplitude], *, viscosity: Amplitude = Fraction(1, 100)
) -> dict[str, Amplitude]:
    """Return exact static invariants of the six-amplitude ansatz.

    With the canonical-coordinate embedding,

    ``A = 64*a*c*d + 32*a*e*f - 64*b*d*f``,

    ``E = 2*a^2 + 32*b^2 + 48*c^2 + 8*d^2 + 48*e^2 + 4*f^2``, and

    ``P = 2*a^2 + 128*b^2 + 144*c^2 + 16*d^2 + 144*e^2 + 8*f^2``.
    """

    if len(amplitudes) != 6:
        raise ValueError("expected the six amplitudes (a,b,c,d,e,f)")
    a, b, c, d, e, f = amplitudes
    stretching = 64 * a * c * d + 32 * a * e * f - 64 * b * d * f
    enstrophy = (
        2 * a * a
        + 32 * b * b
        + 48 * c * c
        + 8 * d * d
        + 48 * e * e
        + 4 * f * f
    )
    palinstrophy = (
        2 * a * a
        + 128 * b * b
        + 144 * c * c
        + 16 * d * d
        + 144 * e * e
        + 8 * f * f
    )
    return {
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "stretching": stretching,
        "rate": stretching - 2 * viscosity * palinstrophy,
    }


def reduced_amplitudes_from_controls(
    controls: Sequence[float], *, tolerance: float = 1.0e-10
) -> tuple[float, ...]:
    """Recover the ansatz amplitudes and reject controls outside its support."""

    if len(controls) != 64:
        raise ValueError("expected 64 canonical low-mode controls")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    amplitudes = (
        float(controls[0]),
        float(controls[20]),
        float(-controls[26]),
        float(-controls[29]),
        float(-controls[48]),
        float(-controls[54]),
    )
    expected = embed_reduced_static(amplitudes)
    residual = max(
        abs(float(actual) - float(expected_value))
        for actual, expected_value in zip(controls, expected, strict=True)
    )
    if residual > tolerance:
        raise ValueError(
            f"controls are not in the reduced static ansatz (max residual {residual})"
        )
    return amplitudes


__all__ = [
    "embed_reduced_static",
    "reduced_amplitudes_from_controls",
    "reduced_static_invariants",
]
