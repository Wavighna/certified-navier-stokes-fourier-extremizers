"""Attempt a uniform full-space stability certificate for the six branch.

The high-eta six-amplitude KKT branch has a compact one-parameter
description after putting ``r=1/mu=s/10`` with ``0 <= s <= 1``.  This
script evaluates the *full 64-coordinate bordered Hessian* with Arb interval
arithmetic over subintervals of that compact parameter.  A successful cover
would prove strict local maximality of the branch, modulo translations, for
every ``eta > 1867/4``.  Until a complete cover closes, the emitted payload is
explicitly labelled as an incomplete proof attempt.

Unlike the earlier floating-point atlas, this keeps the exact Fourier tensor,
the physical Helmholtz projection, and every off-ansatz Hessian direction.
The chosen translated three-phase slice remains regular at both compactified
endpoints, so no limiting chart argument is hidden in the calculation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
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
from extreme_flows.reduced_static import embed_reduced_static
from extreme_flows.reduced_static_parameterized import (
    UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD,
    dimensionless_interior_quartic,
)
from extreme_flows.static_symmetry import IDENTITY_GENERATOR, static_symmetry_matrix


ROOT = Path(__file__).resolve().parents[1]
TARGET_ENSTROPHY = Fraction(100)
PHASE_WAVES = ((1, -1, -1), (1, -1, 1), (1, 1, 0))
PHASE_POLARIZATIONS = (0, 0, 0)
QUARTER_TRANSLATION = (IDENTITY_GENERATOR[0], (1, 0, 0))


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    data = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


@lru_cache(maxsize=1)
def _model() -> StaticLowModePolynomial:
    # Viscosity is supplied as an Arb interval below, so the model only holds
    # the exact Fourier tensor, diagonal E/P forms, and phase indices.
    return StaticLowModePolynomial(
        viscosity=Fraction(0),
        phase_waves=PHASE_WAVES,
        phase_polarizations=PHASE_POLARIZATIONS,
    )


def _branch_arb_controls(s):
    """Return translated controls, viscosity, and lambda on an Arb ``s`` box.

    With ``r=s/10`` and

    ``n=69+1696r+14936r^2+54656r^3+68368r^4``,

    the displayed radicals are algebraically equivalent to the high-eta
    formula in ``reduced_static_parameterized`` but have no singularity at
    the inviscid compactification ``s=0``. At the actual high-eta threshold,
    the unique positive quartic root obeys ``mu>10`` because its quartic is
    negative at 10. Thus the closed auxiliary cover ``0<=s<=1`` contains
    every physical point ``eta>1867/4`` while avoiding an algebraic endpoint
    root in the interval parameter.
    """

    from flint import arb

    energy = arb(_arb_fraction(TARGET_ENSTROPHY))
    r = s / 10
    one = arb(1)
    n = 69 + 1696 * r + 14936 * r * r + 54656 * r**3 + 68368 * r**4
    d_factor = 5 + 48 * r + 100 * r * r
    g_factor = 3 + 48 * r + 156 * r * r
    f_factor = one - 28 * r * r
    a = (
        energy * 3 * (one + 8 * r) * (one + 6 * r) * g_factor / n
    ).sqrt()
    d = (energy * (one + 8 * r) ** 2 * d_factor / (2 * n)).sqrt()
    f = (2 * energy * (one + 8 * r) ** 2 * f_factor / n).sqrt()
    b = -(energy * d_factor * f_factor / (32 * n)).sqrt()
    c = (
        energy * (one + 8 * r) * g_factor * d_factor
        / (48 * n * (one + 6 * r))
    ).sqrt()
    e = (
        energy * (one + 8 * r) * g_factor * f_factor
        / (48 * n * (one + 6 * r))
    ).sqrt()
    viscosity = (32 * energy).sqrt() * r * (one + 8 * r) / n.sqrt()
    multiplier = (32 * energy).sqrt() * (one + 8 * r) / n.sqrt()
    amplitudes = (a, b, c, d, e, f)
    embedding = [
        embed_reduced_static(
            tuple(Fraction(int(index == amplitude)) for index in range(6))
        )
        for amplitude in range(6)
    ]
    raw = [
        sum(
            (
                _arb_fraction(embedding[amplitude][coordinate]) * amplitudes[amplitude]
                for amplitude in range(6)
            ),
            arb(0),
        )
        for coordinate in range(64)
    ]
    translation = static_symmetry_matrix(QUARTER_TRANSLATION)
    controls = [
        sum(
            (
                _arb_fraction(translation[row][column]) * raw[column]
                for column in range(64)
            ),
            arb(0),
        )
        for row in range(64)
    ]
    return controls, viscosity, multiplier


def _bordered_hessian(s):
    """Construct the symmetric bordered Hessian on an Arb parameter box."""

    from flint import arb, arb_mat

    model = _model()
    controls, viscosity, multiplier = _branch_arb_controls(s)
    dimension = model.dimension
    hessian = [[arb(0) for _ in range(dimension)] for _ in range(dimension)]
    for indices, coefficient in model.third_derivative_terms.items():
        value = _arb_fraction(coefficient)
        for i, j, k in set(permutations(indices)):
            hessian[i][j] += value * controls[k]
    e_diagonal = [_arb_fraction(value) for value in model.enstrophy_diagonal_exact]
    p_diagonal = [_arb_fraction(value) for value in model.palinstrophy_diagonal_exact]
    for index in range(dimension):
        hessian[index][index] -= 2 * viscosity * p_diagonal[index]
        hessian[index][index] -= multiplier * e_diagonal[index]

    size = dimension + 4
    bordered = [[arb(0) for _ in range(size)] for _ in range(size)]
    for row in range(dimension):
        for column in range(dimension):
            bordered[row][column] = hessian[row][column]
        constraint = e_diagonal[row] * controls[row]
        bordered[row][dimension] = constraint
        bordered[dimension][row] = constraint
    for row, index in enumerate(model.phase_indices, start=dimension + 1):
        bordered[index][row] = arb(1)
        bordered[row][index] = arb(1)
    return arb_mat(bordered)


def _interval(index: int, count: int) -> tuple[Fraction, Fraction]:
    if not 0 <= index < count:
        raise IndexError(index)
    return Fraction(index, count), Fraction(index + 1, count)


def _frobenius_upper(matrix):
    """Return a rigorous upper bound for the Frobenius norm of an Arb matrix."""

    from flint import arb

    return sum(
        (
            matrix[row, column].abs_upper() ** 2
            for row in range(matrix.nrows())
            for column in range(matrix.ncols())
        ),
        arb(0),
    ).sqrt()


def _certify_interval_by_midpoint_perturbation(s, midpoint):
    """Certify inertia on ``s`` from a point matrix and a Neumann bound.

    Direct interval LDL/Gershgorin elimination is excessively pessimistic for
    a 68-by-68 nonlinear bordered Hessian.  Let ``B0=B(midpoint)`` and let
    ``Delta=B(s)-B0``.  If ``||B0^-1 Delta||_F<1``, every matrix on the
    straight homotopy from ``B0`` to any member of the interval family is
    nonsingular.  Symmetric inertia is then constant along that homotopy.
    This is a standard verified continuation argument, and the Frobenius norm
    is used only as the certified upper bound on the spectral norm.
    """

    from flint import arb_mat

    midpoint_matrix = _bordered_hessian(midpoint)
    midpoint_inertia = _certify_symmetric_inertia_arb(midpoint_matrix)
    midpoint_valid = bool(
        midpoint_inertia["verified"]
        and midpoint_inertia["positive"] == 4
        and midpoint_inertia["negative"] == 64
        and midpoint_inertia["zero_or_unresolved"] == 0
    )
    inverse = arb_mat(
        [
            [
                midpoint_matrix[row, column].mid()
                for column in range(midpoint_matrix.ncols())
            ]
            for row in range(midpoint_matrix.nrows())
        ]
    ).inv()
    difference = _bordered_hessian(s) - midpoint_matrix
    perturbation_upper = _frobenius_upper(inverse * difference)
    controls, _, _ = _branch_arb_controls(s)
    phase_regular = all(
        # The phase constraints set selected *imaginary* coordinates to zero;
        # their neighbouring real amplitudes must stay nonzero to give a
        # regular translation slice.
        not controls[index - 2].contains(0) for index in _model().phase_indices
    )
    valid = bool(
        midpoint_valid and phase_regular and perturbation_upper < 1
    )
    return valid, midpoint_inertia, perturbation_upper, phase_regular


def _run(count: int, precision: int) -> dict[str, object]:
    """Try one uniform interval cover at a specified Arb precision."""

    from flint import arb, ctx

    old_precision = ctx.prec
    try:
        ctx.prec = int(precision)
        failures: list[dict[str, object]] = []
        margins = []
        for index in range(count):
            lower, upper = _interval(index, count)
            center = (lower + upper) / 2
            s = arb(_arb_fraction(center), _arb_fraction((upper - lower) / 2))
            midpoint = arb(_arb_fraction(center))
            valid, inertia, perturbation_upper, phase_regular = (
                _certify_interval_by_midpoint_perturbation(s, midpoint)
            )
            if valid:
                margins.append(inertia["minimum_signed_gershgorin_margin"])
            else:
                failures.append(
                    {
                        "index": index,
                        "s_range": [str(lower), str(upper)],
                        "positive": int(inertia["positive"]),
                        "negative": int(inertia["negative"]),
                        "zero_or_unresolved": int(inertia["zero_or_unresolved"]),
                        "midpoint_to_box_frobenius_perturbation_upper": str(
                            perturbation_upper
                        ),
                        "phase_chart_regular_on_box": phase_regular,
                    }
                )
        return {
            "precision_bits": int(precision),
            "interval_count": count,
            "all_intervals_certified": not failures,
            "method": "midpoint_inertia_plus_frobenius_neumann_continuation",
            "failures": failures,
            "minimum_signed_gershgorin_margin": (
                str(min(margins, key=float)) if margins else None
            ),
        }
    finally:
        ctx.prec = old_precision


def build_payload(
    *, count: int = 128, precisions: Sequence[int] = (256, 512)
) -> dict[str, object]:
    """Run an interval-cover attempt and truth-label its outcome."""

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires python-flint==0.9.0")
    if count < 1:
        raise ValueError("count must be positive")
    precisions = _validated_proof_precisions(precisions)
    threshold_polynomial = dimensionless_interior_quartic(
        UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD
    )
    threshold_at_ten = sum(
        coefficient * 10**power
        for power, coefficient in enumerate(threshold_polynomial)
    )
    if threshold_at_ten >= 0:
        raise AssertionError("the compact-cover containment witness changed")
    runs = [_run(count, precision) for precision in precisions]
    proved = all(run["all_intervals_certified"] for run in runs)
    return {
        "schema_version": 1,
        "truth_label": (
            "proved_uniform_high_eta_six_branch_full_space_local_maximum"
            if proved
            else "uniform_high_eta_six_branch_interval_cover_incomplete"
        ),
        "statement": (
            "A complete certified cover proves that the exact high-eta six-amplitude "
            "branch is a strict local maximum of A-2*nu*P at fixed enstrophy in "
            "the complete 64-real-control class, modulo translations, for every "
            "eta>1867/4. It is not a global or PDE regularity statement."
        ),
        "compact_parameterization": {
            "s_range": "0<=s<=1",
            "r": "s/10=1/mu",
            "physical_open_branch": "eta>1867/4 implies 0<s<1 (the cover is a safe superset)",
            "containment_witness": (
                "At eta=1867/4, F_eta(10)="
                f"{threshold_at_ten}<0. The high-eta theorem gives one positive "
                "root and F_eta(+infinity)>0, so mu>10 on the whole physical branch."
            ),
            "endpoints": {
                "s=0": "inviscid six-branch limit",
                "s=1": "mu=10 auxiliary endpoint; the physical threshold has mu>10",
            },
            "phase_chart": {
                "translation": "(pi/2,0,0)",
                "waves": [list(wave) for wave in PHASE_WAVES],
                "polarizations": list(PHASE_POLARIZATIONS),
                "regularity_note": "selected phase amplitudes remain nonzero on the compact cover",
            },
        },
        "arb": {"dependency": _python_flint_dependency_record(), "runs": runs},
        "checker_source": _record("scripts/certify_six_high_eta_uniform.py"),
        "polynomial_source": _record("src/extreme_flows/certify.py"),
        "formula_source": _record("src/extreme_flows/reduced_static_parameterized.py"),
        "claims": {
            "strict_full_space_local_maximum_modulo_translations_for_eta_gt_1867_over_4": proved,
            "full_64d_global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--precisions", type=int, nargs="+", default=(256, 512))
    parser.add_argument(
        "--output", default="artifacts/proofs/six_branch_high_eta_uniform.json"
    )
    args = parser.parse_args()
    payload = build_payload(count=args.count, precisions=args.precisions)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["truth_label"])
    for run in payload["arb"]["runs"]:
        print(run["precision_bits"], run["all_intervals_certified"], len(run["failures"]))
    if not payload["claims"][
        "strict_full_space_local_maximum_modulo_translations_for_eta_gt_1867_over_4"
    ]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
