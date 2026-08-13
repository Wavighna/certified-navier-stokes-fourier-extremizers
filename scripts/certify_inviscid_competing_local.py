"""Certify the exact inviscid competing branch as a full-space local maximum.

At fixed enstrophy the two-amplitude formula has inviscid amplitudes
``a=sqrt(2 E)/3`` and ``b=sqrt(E)/12``.  A quarter-period translation puts
the exact point in a regular phase slice. This script certifies its strict
full-64D local maximality, not globality.
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
from extreme_flows.reduced_competing import embed_competing
from extreme_flows.static_symmetry import IDENTITY_GENERATOR, static_symmetry_matrix


TARGET_ENSTROPHY = Fraction(100)
PHASE_WAVES = ((0, 1, -1), (0, 1, 1), (1, -1, 0))
PHASE_POLARIZATIONS = (1, 1, 1)
QUARTER_TRANSLATION = (IDENTITY_GENERATOR[0], (0, 1, 0))


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
    """Return the translated exact inviscid two-amplitude KKT point at E=100."""

    from flint import arb

    a = 10 * arb(2).sqrt() / 3
    b = arb(5) / 6
    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    raw_controls = [
        _arb_fraction(primary[index]) * a + _arb_fraction(secondary[index]) * b
        for index in range(64)
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
    # A=(4/9)E^(3/2), so 3A/(2E)=20/3 at E=100.
    return controls + [arb(20) / 3, arb(0), arb(0), arb(0)]


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
    """Build the proof-grade competing-endpoint certificate."""

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
            "proof_grade_inviscid_competing_branch_full_64d_strict_local_maximum"
            if proved
            else "inviscid_competing_branch_interval_certificate_did_not_close"
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
            "translation_to_phase_slice": "(0,pi/2,0)",
            "exact_cubic_tensor_sha256": _exact_tensor_sha256(_model()),
        },
        "exact_formula": {
            "unit_enstrophy_amplitudes_a_b": "(sqrt(2)/3,1/12)",
            "unit_enstrophy_stretching": "4/9",
            "E_100_lagrange_multiplier": "20/3",
        },
        "scaling_consequence": (
            "Positive scaling maps each fixed-enstrophy sphere to every other "
            "and scales the cubic A positively, so the E=100 strict local "
            "statement holds for every E0>0."
        ),
        "runs": runs,
        "checker_source": _checker_source_record("certify_inviscid_competing_local.py"),
        "additional_sources": {
            "competing_formula": _source_record(
                "src/extreme_flows/reduced_competing.py"
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
        default="artifacts/proofs/inviscid_competing_full64_local_certificate.json",
    )
    args = parser.parse_args()
    payload = build_certificate()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["claims"]["strict_local_maximum_modulo_translations_at_E_100"]:
        raise RuntimeError("inviscid competing-branch interval certificate did not close")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
