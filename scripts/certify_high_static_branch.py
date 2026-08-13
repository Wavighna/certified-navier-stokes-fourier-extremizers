"""Certify the numerically continued competing static KKT branch.

This is deliberately a *local* computer-assisted proof.  It establishes a
strict local maximizer of the fixed 64-real-control variational problem at
``nu=1/10, E=100`` (hence ``eta=E/nu**2=10_000``), if the interval checks
close.  It makes no global assertion.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import root

from extreme_flows.certify import (
    StaticLowModePolynomial,
    _arb_fraction,
    _certify_symmetric_inertia_arb,
    _checker_source_record,
    _exact_tensor_sha256,
    _krawczyk_static_box,
    _python_flint_dependency_record,
    _refine_static_root_arb,
    _static_arb_components,
    _validated_proof_precisions,
    phase_fix_controls,
)
from extreme_flows.reduced_competing import embed_competing


HIGH_PHASE_WAVES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
HIGH_PHASE_POLARIZATIONS = (0, 1, 0)
HIGH_VISCOSITY = Fraction(1, 10)
TARGET_ENSTROPHY = 100.0


def _model() -> StaticLowModePolynomial:
    return StaticLowModePolynomial(
        viscosity=HIGH_VISCOSITY,
        phase_waves=HIGH_PHASE_WAVES,
        phase_polarizations=HIGH_PHASE_POLARIZATIONS,
    )


def _source_record(relative_path: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    data = path.read_bytes()
    return {"path": relative_path.replace("\\", "/"), "size_bytes": len(data), "sha256": sha256(data).hexdigest()}


def _exact_formula_arb_point():
    """Enclose the closed-form two-amplitude KKT point with Arb balls."""

    from flint import arb

    viscosity = _arb_fraction(HIGH_VISCOSITY)
    enstrophy = arb(100)
    b = ((enstrophy + viscosity * viscosity).sqrt() - viscosity) / 12
    a = ((enstrophy - 48 * b * b) / 3).sqrt()
    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    controls = [
        _arb_fraction(coefficient_a) * a + _arb_fraction(coefficient_b) * b
        for coefficient_a, coefficient_b in zip(primary, secondary, strict=True)
    ]
    return controls + [8 * b - 2 * viscosity, arb(0), arb(0), arb(0)]


def make_candidate(source: str | Path, output: str | Path) -> dict[str, object]:
    """Scale the eta=10,000 continuation state and polish its 68D KKT system."""

    source_path = Path(source)
    raw = source_path.read_bytes()
    archive = json.loads(raw)
    step = archive["steps"][-1]
    if float(step["eta"]) != 10_000.0:
        raise ValueError("source archive must terminate at eta=10,000")
    # The continuation uses nu=.01, E=1.  Scaling (u,nu) by 10 preserves eta
    # and takes it to the target convention nu=.1, E=100.
    controls = 10.0 * np.asarray(step["terminal_controls"], dtype=float)
    model = _model()
    controls = phase_fix_controls(model, controls)
    if abs(float(model.evaluate(controls)["enstrophy"]) - TARGET_ENSTROPHY) > 2e-6:
        raise ArithmeticError("eta-preserving scale did not produce E=100")

    gradient = np.asarray(model.evaluate(controls)["gradient_rate"])
    constraints = model.constraint_jacobian(controls)
    multipliers = np.linalg.lstsq(constraints.T, gradient, rcond=None)[0]
    initial = np.concatenate((controls, multipliers))
    polished = root(
        lambda point: model.kkt_system(point, TARGET_ENSTROPHY),
        initial,
        jac=model.kkt_jacobian,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 5000},
    )
    point = np.asarray(polished.x, dtype=float)
    residual = model.kkt_system(point, TARGET_ENSTROPHY)
    evaluated = model.evaluate(point[: model.dimension])
    tangent = null_space(model.constraint_jacobian(point[: model.dimension]))
    hessian = np.asarray(evaluated["hessian_rate"]) - point[-4] * np.asarray(
        evaluated["hessian_enstrophy"]
    )
    eigenvalues = np.linalg.eigvalsh(tangent.T @ hessian @ tangent)
    phase_real = [point[index - 2] for index in model.phase_indices]
    phase_matrix = np.asarray(HIGH_PHASE_WAVES, dtype=float) * np.asarray(
        phase_real
    )[:, None]
    payload: dict[str, object] = {
        "schema_version": 1,
        "truth_label": "numerical_high_static_branch_kkt_candidate_only",
        "source": {
            "path": str(source_path),
            "sha256": sha256(raw).hexdigest(),
            "source_eta": float(step["eta"]),
            "source_rate": float(step["rate"]),
            "source_ansatz_rate": float(step["ansatz_rate"]),
            "source_rate_ratio_to_ansatz": float(step["rate_ratio_to_ansatz"]),
            "eta_preserving_scale": 10.0,
        },
        "model": {
            "dimension": model.dimension,
            "viscosity": "1/10",
            "target_enstrophy": TARGET_ENSTROPHY,
            "eta": 10_000.0,
            "phase_waves": [list(wave) for wave in HIGH_PHASE_WAVES],
            "phase_polarizations": list(HIGH_PHASE_POLARIZATIONS),
            "phase_indices": list(model.phase_indices),
            "exact_cubic_tensor_sha256": _exact_tensor_sha256(model),
        },
        "candidate": {
            "controls": point[: model.dimension].tolist(),
            "multipliers": point[model.dimension :].tolist(),
            "enstrophy": float(evaluated["enstrophy"]),
            "palinstrophy": float(evaluated["palinstrophy"]),
            "stretching": float(evaluated["stretching"]),
            "rate": float(evaluated["rate"]),
            "rate_ratio_to_six_amplitude_ansatz": float(evaluated["rate"])
            / (1000.0 * float(step["ansatz_rate"])),
            "max_abs_kkt_residual": float(np.max(np.abs(residual))),
            "root_success": bool(polished.success),
            "root_message": str(polished.message),
            "tangent_hessian_min_eigenvalue": float(eigenvalues[0]),
            "tangent_hessian_max_eigenvalue": float(eigenvalues[-1]),
            "phase_chart": {
                "selected_real_amplitudes": [float(value) for value in phase_real],
                "translation_jacobian_rank": int(np.linalg.matrix_rank(phase_matrix)),
            },
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _run_arb(candidate: dict[str, object], precision: int) -> dict[str, object]:
    from flint import ctx

    model = _model()
    data = candidate["candidate"]
    initial = [float(value) for value in data["controls"] + data["multipliers"]]
    old_precision = ctx.prec
    try:
        ctx.prec = precision
        center, residual, correction = _refine_static_root_arb(model, initial)
        krawczyk = _krawczyk_static_box(model, center)
        if not krawczyk["verified"]:
            return {"precision_bits": precision, "krawczyk_verified": False,
                    "attempts": krawczyk["attempts"]}
        inertia = _certify_symmetric_inertia_arb(krawczyk["bordered_hessian"])
        digits = max(30, int(precision * 0.30103) - 8)
        box = krawczyk["box"]
        formula_point = _exact_formula_arb_point()
        formula_inclusions = [
            box[index].contains(formula_point[index]) for index in range(len(box))
        ]
        selected = [index - 2 for index in model.phase_indices]
        max_residual = max((value.abs_upper() for value in residual), key=float)
        return {
            "precision_bits": precision,
            "krawczyk_verified": True,
            "existence_verified": bool(krawczyk["existence_verified"]),
            "uniqueness_verified": bool(krawczyk["uniqueness_verified"]),
            "selected_attempt": krawczyk["selected_attempt"],
            "weighted_infinity_contraction_upper_ball": krawczyk[
                "weighted_infinity_contraction_upper"
            ].str(digits),
            "newton_last_correction_ball": correction.str(digits),
            "refined_residual_abs_upper_ball": max_residual.str(digits),
            "phase_chart_regular": all(not box[index].contains(0) for index in selected),
            "exact_two_amplitude_formula_inside_unique_krawczyk_box": all(
                formula_inclusions
            ),
            "formula_failed_component_count": formula_inclusions.count(False),
            "selected_real_amplitude_balls": [box[index].str(digits) for index in selected],
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


def certify(candidate_path: str | Path, output: str | Path, precisions: Sequence[int]) -> dict[str, object]:
    precisions = _validated_proof_precisions(precisions)
    raw = Path(candidate_path).read_bytes()
    candidate = json.loads(raw)
    runs = [_run_arb(candidate, precision) for precision in precisions]
    proved = all(
        run.get("krawczyk_verified")
        and run.get("existence_verified")
        and run.get("uniqueness_verified")
        and run.get("phase_chart_regular")
        and run.get("exact_two_amplitude_formula_inside_unique_krawczyk_box")
        and run.get("bordered_inertia", {}).get("verified")
        and run["bordered_inertia"].get("positive") == 4
        and run["bordered_inertia"].get("negative") == 64
        and run["bordered_inertia"].get("zero_or_unresolved") == 0
        for run in runs
    )
    payload = {
        "schema_version": 1,
        "truth_label": (
            "proved_strict_local_static_maximum_high_branch_modulo_translations"
            if proved else "high_branch_interval_certificate_did_not_close"
        ),
        "candidate_sha256": sha256(raw).hexdigest(),
        "parameter_point": {"viscosity": "1/10", "enstrophy": "100", "eta": "10000"},
        "statement": (
            "A unique KKT root in the stated box is a strict local maximum, modulo "
            "translations, in the 64-real-control low-mode class. This is not a global claim."
        ),
        "arb": {"dependency": _python_flint_dependency_record(), "runs": runs},
        "checker_source": _checker_source_record("certify_high_static_branch.py"),
        "formula_dependencies": {
            "exact_formula_embedding": _source_record(
                "src/extreme_flows/reduced_competing.py"
            ),
            "symbolic_full_kkt_reduction": _source_record(
                "scripts/certify_reduced_competing.py"
            ),
            "exact_symmetry_reduction": _source_record(
                "src/extreme_flows/static_symmetry.py"
            ),
        },
        "claims": {
            "unique_kkt_root_in_reported_box": proved,
            "strict_local_maximum_modulo_translations": proved,
            "exact_two_amplitude_formula_is_the_certified_full_kkt_root": proved,
            "strictly_beats_six_amplitude_ansatz": bool(
                candidate["candidate"]["rate_ratio_to_six_amplitude_ansatz"] > 1.0
            ),
            "global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="artifacts/raw/full_static_high_branch_3000_10000.json")
    parser.add_argument("--candidate", default="artifacts/proofs/high_branch_eta10000_candidate.json")
    parser.add_argument("--output", default="artifacts/proofs/high_branch_eta10000_arb_certificate.json")
    parser.add_argument("--precisions", type=int, nargs="+", default=(256, 512))
    parser.add_argument("--reuse-candidate", action="store_true")
    args = parser.parse_args()
    candidate = (json.loads(Path(args.candidate).read_text(encoding="utf-8")) if args.reuse_candidate
                 else make_candidate(args.source, args.candidate))
    proof = certify(args.candidate, args.output, args.precisions)
    print(f"candidate rate: {candidate['candidate']['rate']:.12g}")
    print(f"certificate: {proof['truth_label']}")
    if not proof["claims"]["strict_local_maximum_modulo_translations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
