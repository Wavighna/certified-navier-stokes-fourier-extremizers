"""Interval-certify high-eta local maximality of the competing KKT branch.

The exact branch is parameterized at unit enstrophy by ``s=12 b``:

    a=sqrt(3-s^2)/3,  b=s/12,  nu=(1-s^2)/(2s),

with ``eta=4s^2/(1-s^2)^2``.  A rational interval cover of
``9049/10000 <= s <= 1`` contains every positive-viscosity point with
``eta >= 100``.  On each interval this checker builds the exact bordered
Hessian in the same translation phase chart as the point certificates and
certifies inertia ``(4+,64-,0)`` using outward-rounded Arb arithmetic.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import importlib.metadata
import json
from itertools import permutations
from pathlib import Path
from typing import Sequence

from extreme_flows.certify import (
    StaticLowModePolynomial,
    _arb_fraction,
    _certify_symmetric_inertia_arb,
    _python_flint_dependency_record,
    _validated_proof_precisions,
)
from extreme_flows.reduced_competing import embed_competing


ROOT = Path(__file__).resolve().parents[1]
LOWER_S = Fraction(9049, 10_000)
UPPER_S = Fraction(1)
INTERVAL_COUNT = 238
PHASE_WAVES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
PHASE_POLARIZATIONS = (0, 1, 0)


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    data = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _interval(index: int) -> tuple[Fraction, Fraction]:
    if not 0 <= index < INTERVAL_COUNT:
        raise IndexError(index)
    width = (UPPER_S - LOWER_S) / INTERVAL_COUNT
    return LOWER_S + index * width, LOWER_S + (index + 1) * width


def _branch_bordered_hessian(model: StaticLowModePolynomial, lower: Fraction, upper: Fraction):
    """Return the exact-coefficient Arb bordered Hessian on one ``s`` box."""

    from flint import arb, arb_mat

    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    s = arb(_arb_fraction(center), _arb_fraction(radius))
    a = (arb(3) - s * s).sqrt() / 3
    b = s / 12
    viscosity = (arb(1) - s * s) / (2 * s)
    multiplier = 5 * s / 3 - 1 / s

    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    controls = [
        _arb_fraction(a_coefficient) * a + _arb_fraction(b_coefficient) * b
        for a_coefficient, b_coefficient in zip(primary, secondary, strict=True)
    ]
    dimension = model.dimension
    size = dimension + 4
    bordered = [[arb(0) for _ in range(size)] for _ in range(size)]
    for indices, value in model.third_derivative_terms.items():
        coefficient = _arb_fraction(value)
        for i, j, k in set(permutations(indices)):
            bordered[i][j] += coefficient * controls[k]
    for index in range(dimension):
        bordered[index][index] -= (
            2 * viscosity * _arb_fraction(model.palinstrophy_diagonal_exact[index])
            + multiplier * _arb_fraction(model.enstrophy_diagonal_exact[index])
        )
        derivative = _arb_fraction(model.enstrophy_diagonal_exact[index]) * controls[index]
        bordered[index][dimension] = derivative
        bordered[dimension][index] = derivative
    for row, index in enumerate(model.phase_indices, start=dimension + 1):
        bordered[index][row] = arb(1)
        bordered[row][index] = arb(1)
    return arb_mat(bordered)


def _run(precision: int) -> dict[str, object]:
    from flint import ctx

    model = StaticLowModePolynomial(
        phase_waves=PHASE_WAVES, phase_polarizations=PHASE_POLARIZATIONS
    )
    old_precision = ctx.prec
    try:
        ctx.prec = precision
        certified = 0
        margins = []
        failures = []
        for index in range(INTERVAL_COUNT):
            lower, upper = _interval(index)
            inertia = _certify_symmetric_inertia_arb(
                _branch_bordered_hessian(model, lower, upper)
            )
            valid = bool(
                inertia["verified"]
                and inertia["positive"] == 4
                and inertia["negative"] == 64
                and inertia["zero_or_unresolved"] == 0
            )
            if valid:
                certified += 1
                margins.append(inertia["minimum_signed_gershgorin_margin"])
            else:
                failures.append(
                    {
                        "index": index,
                        "lower_s": str(lower),
                        "upper_s": str(upper),
                        "positive": inertia["positive"],
                        "negative": inertia["negative"],
                        "zero_or_unresolved": inertia["zero_or_unresolved"],
                    }
                )
        minimum_margin = min(margins, key=float) if margins else None
        return {
            "precision_bits": precision,
            "certified_intervals": certified,
            "total_intervals": INTERVAL_COUNT,
            "all_intervals_certified": certified == INTERVAL_COUNT,
            "minimum_signed_gershgorin_margin": str(minimum_margin) if minimum_margin else None,
            "failures": failures,
        }
    finally:
        ctx.prec = old_precision


def build_payload(precisions: Sequence[int] = (256, 512)) -> dict[str, object]:
    """Build the proof-grade uniform high-eta local-maximality certificate."""

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires python-flint==0.9.0")
    precisions = _validated_proof_precisions(precisions)
    runs = [_run(precision) for precision in precisions]
    certified = all(run["all_intervals_certified"] for run in runs)
    # Exact inequality: sqrt(101/100)-1/10 > 9049/10000. Squaring the
    # positive sides yields the recorded positive rational difference.
    threshold_difference = Fraction(101, 100) - Fraction(10049, 10_000) ** 2
    if threshold_difference <= 0:
        raise AssertionError("rational cover lower endpoint misses eta=100 threshold")
    return {
        "schema_version": 1,
        "truth_label": (
            "proved_uniform_high_eta_competing_full_space_local_maximum"
            if certified
            else "uniform_high_eta_competing_local_maximum_certificate_incomplete"
        ),
        "statement": (
            "For every E,nu>0 with eta=E/nu^2>=100, the exact competing "
            "two-amplitude KKT branch is a strict local maximum modulo "
            "translations in the 64-real-control class. This is not a "
            "full-space global-maximality or PDE regularity statement."
        ),
        "normalization_and_cover": {
            "unit_enstrophy_parameterization": "a=sqrt(3-s^2)/3, b=s/12, nu=(1-s^2)/(2s)",
            "eta_relation": "eta=4s^2/(1-s^2)^2",
            "rational_s_cover": [str(LOWER_S), str(UPPER_S)],
            "interval_count": INTERVAL_COUNT,
            "eta_100_threshold_lower_bound": "sqrt(101/100)-1/10 > 9049/10000",
            "squared_positive_side_difference": str(threshold_difference),
            "phase_chart": {
                "waves": [list(wave) for wave in PHASE_WAVES],
                "polarizations": list(PHASE_POLARIZATIONS),
                "regularity": "a^2=(3-s^2)/9>=2/9 on the covered range",
            },
        },
        "arb": {"dependency": _python_flint_dependency_record(), "runs": runs},
        "checker_source": _record("scripts/certify_competing_high_eta_uniform.py"),
        "formula_source": _record("src/extreme_flows/reduced_competing.py"),
        "polynomial_source": _record("src/extreme_flows/certify.py"),
        "claims": {
            "strict_full_space_local_maximum_modulo_translations_for_all_eta_ge_100": certified,
            "full_space_global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/proofs/competing_high_eta_uniform.json")
    parser.add_argument("--precisions", nargs="+", type=int, default=(256, 512))
    args = parser.parse_args()
    payload = build_payload(args.precisions)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["truth_label"])
    for run in payload["arb"]["runs"]:
        print(run["precision_bits"], run["certified_intervals"], "/", run["total_intervals"])
    if not payload["claims"]["strict_full_space_local_maximum_modulo_translations_for_all_eta_ge_100"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
