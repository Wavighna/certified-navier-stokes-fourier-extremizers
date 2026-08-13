"""Certify the full-64D local maximality of the exact inviscid six branch.

The displayed radical formula is first translated by ``(pi/2,0,0)`` to a
regular three-phase slice.  Krawczyk inclusion and bordered-Hessian inertia
then prove a strict constrained local maximum of stretching in the complete
64-real-coordinate Fourier class.  This is local maximality only: it does not
prove the full-space cubic-global conjecture.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from extreme_flows.certify import (
    StaticLowModePolynomial,
    _arb_fraction,
    _certify_symmetric_inertia_arb,
    _checker_source_record,
    _exact_tensor_sha256,
    _krawczyk_static_box,
    _python_flint_dependency_record,
    _refine_static_root_arb,
    _validated_proof_precisions,
)
from extreme_flows.reduced_static import embed_reduced_static
from extreme_flows.static_symmetry import IDENTITY_GENERATOR, static_symmetry_matrix


TARGET_ENSTROPHY = Fraction(100)
PHASE_WAVES = ((1, -1, -1), (1, -1, 1), (1, 1, 0))
PHASE_POLARIZATIONS = (0, 0, 0)
QUARTER_TRANSLATION = (IDENTITY_GENERATOR[0], (1, 0, 0))


def _model() -> StaticLowModePolynomial:
    return StaticLowModePolynomial(
        viscosity=Fraction(0),
        phase_waves=PHASE_WAVES,
        phase_polarizations=PHASE_POLARIZATIONS,
    )


def _source_record(relative_path: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    data = path.read_bytes()
    return {
        "path": relative_path.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def exact_formula_arb_point():
    """Return the exact radical branch after a quarter-period translation."""

    from flint import arb

    # Unit-enstrophy amplitudes are the inviscid six-space extremizer. Scale
    # them by ten to put the Krawczyk calculation on E=100.
    amplitudes = (
        10 * arb(69).sqrt() / 23,
        -10 * arb(690).sqrt() / 552,
        10 * arb(345).sqrt() / 276,
        10 * arb(690).sqrt() / 138,
        10 * arb(69).sqrt() / 276,
        10 * arb(138).sqrt() / 69,
    )
    basis = [
        embed_reduced_static(
            tuple(Fraction(int(index == amplitude)) for index in range(6))
        )
        for amplitude in range(6)
    ]
    raw_controls = [
        sum(
            (
                _arb_fraction(basis[amplitude][coordinate]) * amplitudes[amplitude]
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
                _arb_fraction(translation[row][column]) * raw_controls[column]
                for column in range(64)
            ),
            arb(0),
        )
        for row in range(64)
    ]
    # Euler homogeneity gives lambda=3A/(2E)=40 sqrt(138)/69 at E=100.
    lagrange_multiplier = 40 * arb(138).sqrt() / 69
    return controls + [lagrange_multiplier, arb(0), arb(0), arb(0)]


def _run(precision: int) -> dict[str, object]:
    from flint import ctx

    model = _model()
    old_precision = ctx.prec
    try:
        ctx.prec = int(precision)
        formula = exact_formula_arb_point()
        center, residual, correction = _refine_static_root_arb(
            model, [float(value) for value in formula]
        )
        krawczyk = _krawczyk_static_box(model, center)
        if not krawczyk["verified"]:
            return {
                "precision_bits": int(precision),
                "krawczyk_verified": False,
                "attempts": krawczyk["attempts"],
            }
        inertia = _certify_symmetric_inertia_arb(krawczyk["bordered_hessian"])
        box = krawczyk["box"]
        formula_inclusions = [
            box[index].contains(formula[index]) for index in range(len(box))
        ]
        selected_real_indices = [index - 2 for index in model.phase_indices]
        max_residual = max((value.abs_upper() for value in residual), key=float)
        digits = max(30, int(precision * 0.30103) - 8)
        return {
            "precision_bits": int(precision),
            "krawczyk_verified": True,
            "existence_verified": bool(krawczyk["existence_verified"]),
            "uniqueness_verified": bool(krawczyk["uniqueness_verified"]),
            "selected_attempt": krawczyk["selected_attempt"],
            "weighted_infinity_contraction_upper_ball": krawczyk[
                "weighted_infinity_contraction_upper"
            ].str(digits),
            "newton_last_correction_ball": correction.str(digits),
            "refined_residual_abs_upper_ball": max_residual.str(digits),
            "phase_chart_regular": all(
                not box[index].contains(0) for index in selected_real_indices
            ),
            "selected_real_amplitude_balls": [
                box[index].str(digits) for index in selected_real_indices
            ],
            "exact_translated_formula_inside_unique_krawczyk_box": all(
                formula_inclusions
            ),
            "formula_failed_component_count": formula_inclusions.count(False),
            "bordered_inertia": {
                "verified": bool(inertia["verified"]),
                "positive": int(inertia["positive"]),
                "negative": int(inertia["negative"]),
                "zero_or_unresolved": int(inertia["zero_or_unresolved"]),
                "minimum_signed_gershgorin_margin_ball": inertia[
                    "minimum_signed_gershgorin_margin"
                ].str(digits),
            },
        }
    finally:
        ctx.prec = old_precision


def build_certificate(precisions: Sequence[int] = (256, 512)) -> dict[str, object]:
    """Build the proof-grade payload at both required Arb precisions."""

    precisions = _validated_proof_precisions(precisions)
    runs = [_run(precision) for precision in precisions]
    proved = all(
        run.get("krawczyk_verified")
        and run.get("existence_verified")
        and run.get("uniqueness_verified")
        and run.get("phase_chart_regular")
        and run.get("exact_translated_formula_inside_unique_krawczyk_box")
        and run.get("bordered_inertia", {}).get("verified")
        and run["bordered_inertia"].get("positive") == 4
        and run["bordered_inertia"].get("negative") == 64
        and run["bordered_inertia"].get("zero_or_unresolved") == 0
        for run in runs
    )
    return {
        "schema_version": 1,
        "truth_label": (
            "proof_grade_inviscid_six_branch_full_64d_strict_local_maximum"
            if proved
            else "inviscid_six_branch_interval_certificate_did_not_close"
        ),
        "problem": {
            "domain": "[0,2*pi)^3",
            "dimension": 64,
            "independent_mode_pairs": 16,
            "viscosity": "0",
            "certified_enstrophy": "100",
            "objective": "A=<omega dot S omega>",
            "phase_waves": [list(wave) for wave in PHASE_WAVES],
            "phase_polarizations": list(PHASE_POLARIZATIONS),
            "phase_indices": list(_model().phase_indices),
            "translation_to_phase_slice": "(pi/2,0,0)",
            "exact_cubic_tensor_sha256": _exact_tensor_sha256(_model()),
        },
        "exact_formula": {
            "unit_enstrophy_six_amplitudes_a_b_c_d_e_f": (
                "(sqrt(69)/23,-sqrt(690)/552,sqrt(345)/276,"
                "sqrt(690)/138,sqrt(69)/276,sqrt(138)/69)"
            ),
            "unit_enstrophy_stretching": "8*sqrt(138)/207",
            "E_100_lagrange_multiplier": "40*sqrt(138)/69",
        },
        "scaling_consequence": (
            "Positive scaling maps every E=100 constrained neighborhood to an "
            "E=E0 constrained neighborhood and scales A by a positive factor. "
            "Thus the certified E=100 strict local maximum gives the same "
            "strict local-maximality statement for every E0>0."
        ),
        "runs": runs,
        "checker_source": _checker_source_record("certify_inviscid_six_local.py"),
        "additional_sources": {
            "six_amplitude_formula": _source_record(
                "src/extreme_flows/reduced_static.py"
            ),
            "translation_action": _source_record(
                "src/extreme_flows/static_symmetry.py"
            ),
        },
        "dependency": _python_flint_dependency_record(),
        "claims": {
            "exact_translated_formula_is_inside_unique_kkt_box": proved,
            "strict_local_maximum_modulo_translations_at_E_100": proved,
            "strict_local_maximum_modulo_translations_for_every_positive_E": proved,
            "full_64d_inviscid_global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/proofs/inviscid_six_full64_local_certificate.json",
    )
    args = parser.parse_args()
    payload = build_certificate()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["claims"]["strict_local_maximum_modulo_translations_at_E_100"]:
        raise RuntimeError("inviscid six-branch interval certificate did not close")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
