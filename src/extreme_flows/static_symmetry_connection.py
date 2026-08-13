"""Connect the global six-amplitude extremizer to the full KKT certificate.

The finite symmetry group fixes exactly the six-amplitude ansatz.  By symmetric
criticality, a constrained critical point of the restricted invariant
functional is a full constrained critical point.  This checker constructs the
interval reduced root and proves it lies in the unique full 68-variable
Krawczyk box, including zero translation-gauge multipliers.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Sequence

from .certify import (
    ADAPTIVE_PHASE_WAVES,
    StaticLowModePolynomial,
    _krawczyk_static_box,
    _refine_static_root_arb,
)
from .reduced_static import embed_reduced_static
from .reduced_static_global import (
    _INTERIOR_POLYNOMIAL,
    _arb_fraction,
    _arb_interval,
    _interior_quantities,
    _positive_root_interval,
)
from .static_symmetry import reduced_ansatz_is_exact_fixed_space


def _ball(value) -> dict[str, object]:
    return {
        "ball": str(value),
        "lower": str(value.lower()),
        "upper": str(value.upper()),
        "finite": bool(value.is_finite()),
    }


def _dependency() -> dict[str, object]:
    distribution = importlib.metadata.distribution("python-flint")
    root = Path(distribution.locate_file(""))
    record = root / "python_flint-0.9.0.dist-info" / "RECORD"
    return {
        "package": "python-flint",
        "version": distribution.version,
        "python": sys.version,
        "platform": platform.platform(),
        "record_sha256": sha256(record.read_bytes()).hexdigest(),
    }


def _source_record(script_name: str) -> dict[str, object]:
    module = Path(__file__).resolve()
    root = module.parents[2]
    script = root / "scripts" / script_name
    dependencies = {
        "module": module,
        "script": script,
        "full_krawczyk": root / "src" / "extreme_flows" / "certify.py",
        "reduced_global": root / "src" / "extreme_flows" / "reduced_static_global.py",
        "symmetry": root / "src" / "extreme_flows" / "static_symmetry.py",
    }
    output = {}
    for name, path in dependencies.items():
        data = path.read_bytes()
        output[name] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        }
    return output


def _reduced_full_point(*, bits: int):
    """Return an interval KKT point with the candidate's sign representative."""

    from flint import arb

    root_interval, root_metadata = _positive_root_interval(
        _INTERIOR_POLYNOMIAL, bits=bits
    )
    multiplier = _arb_interval(*root_interval)
    a_squared, d_squared, f_squared, rate = _interior_quantities(multiplier)
    a = a_squared.sqrt()
    d = d_squared.sqrt()
    f = f_squared.sqrt()
    b = -d * f / (multiplier + _arb_fraction(Fraction(2, 25)))
    c = _arb_fraction(Fraction(2, 3)) * a * d / (
        multiplier + _arb_fraction(Fraction(3, 50))
    )
    e = _arb_fraction(Fraction(1, 3)) * a * f / (
        multiplier + _arb_fraction(Fraction(3, 50))
    )
    amplitudes = (a, b, c, d, e, f)
    controls = embed_reduced_static(amplitudes)
    # The adaptive phase chart fixes three imaginary coordinate components to
    # zero.  The symmetry representative has all three exactly zero, hence the
    # associated phase multipliers are zero as well.
    point = [arb(value) for value in controls] + [multiplier, arb(0), arb(0), arb(0)]
    return point, amplitudes, rate, root_interval, root_metadata


def evaluate_static_symmetry_connection(
    candidate_path: str | Path, *, precision: int, bits: int = 100
) -> dict[str, object]:
    """Run one precision of the reduced-root/full-Krawczyk inclusion check."""

    from flint import ctx

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires the pinned python-flint==0.9.0")
    raw = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if tuple(tuple(wave) for wave in raw["model"]["phase_waves"]) != ADAPTIVE_PHASE_WAVES:
        raise ValueError("candidate does not use the adaptive full KKT phase chart")
    previous_precision = ctx.prec
    try:
        ctx.prec = int(precision)
        model = StaticLowModePolynomial(phase_waves=ADAPTIVE_PHASE_WAVES)
        seed = raw["candidate"]
        center, _, _ = _refine_static_root_arb(
            model,
            [float(value) for value in seed["controls"] + seed["multipliers"]],
        )
        krawczyk = _krawczyk_static_box(model, center)
        if not krawczyk["verified"]:
            return {"precision_bits": precision, "full_krawczyk_verified": False}
        point, amplitudes, rate, root_interval, root_metadata = _reduced_full_point(
            bits=bits
        )
        box = krawczyk["box"]
        inclusions = [box[index].contains(point[index]) for index in range(68)]
        ratios = [
            abs(point[index] - box[index].mid()).abs_upper() / box[index].rad()
            for index in range(68)
        ]
        # The point is symmetry fixed by construction.  The static functional
        # and enstrophy constraint are invariant under the finite group; group
        # averaging turns any full variation into a fixed variation, so the
        # restricted KKT equations imply the full KKT equations.  Inclusion in
        # the full box identifies this exact root with its unique Krawczyk root.
        return {
            "precision_bits": precision,
            "full_krawczyk_verified": True,
            "exact_fixed_space_dimension_is_six": reduced_ansatz_is_exact_fixed_space(),
            "reduced_root_inside_full_krawczyk_box": all(inclusions),
            "failed_component_count": inclusions.count(False),
            "maximum_center_distance_over_full_box_radius": float(max(ratios)),
            "reduced_multiplier_interval": [str(root_interval[0]), str(root_interval[1])],
            "reduced_multiplier_sturm": root_metadata,
            "reduced_amplitude_balls": [_ball(value) for value in amplitudes],
            "reduced_rate": _ball(rate),
            "full_box_radius_power_of_two": krawczyk["selected_attempt"]["radius_power_of_two"],
            "full_krawczyk_contraction_upper": _ball(
                krawczyk["weighted_infinity_contraction_upper"]
            ),
        }
    finally:
        ctx.prec = previous_precision


def build_static_symmetry_connection_certificate(
    candidate_path: str | Path,
    *,
    precisions: Sequence[int] = (256, 512),
    bits: int = 100,
) -> dict[str, object]:
    """Build the proof-grade identity certificate for the two KKT roots."""

    precision_set = tuple(int(value) for value in precisions)
    if len(set(precision_set)) < 2 or not {256, 512}.issubset(precision_set):
        raise ValueError("proof-grade builders require distinct 256-bit and 512-bit runs")
    source = Path(candidate_path)
    runs = [
        evaluate_static_symmetry_connection(source, precision=precision, bits=bits)
        for precision in precision_set
    ]
    certified = all(
        run.get("full_krawczyk_verified")
        and run.get("exact_fixed_space_dimension_is_six")
        and run.get("reduced_root_inside_full_krawczyk_box")
        for run in runs
    )
    return {
        "schema_version": 1,
        "truth_label": "proof_grade_full_kkt_symmetry_identification" if certified else "full_kkt_symmetry_identification_incomplete",
        "source": {
            "path": str(source).replace("\\", "/"),
            "sha256": sha256(source.read_bytes()).hexdigest(),
        },
        "proof_method": {
            "fixed_space": "exact rational fixed-point row reduction gives the six-amplitude common fixed space",
            "symmetric_criticality": "finite-group averaging and invariance promote restricted E-constrained stationarity to full E-constrained stationarity",
            "identification": "the interval reduced KKT point lies in the full Krawczyk box, whose contraction proof gives a unique full KKT root",
        },
        "arb": {"dependency": _dependency(), "runs": runs},
        "checker_source": _source_record("certify_static_symmetry_connection.py"),
        "claims": {
            "reduced_global_ansatz_root_is_the_full_certified_kkt_root": certified,
            "full_local_root_is_exactly_symmetry_fixed": certified,
            "full_64d_global_maximum": False,
            "navier_stokes_trajectory_or_regularity_statement": False,
        },
        "limitations": {
            "full_64d_local_maximum_only": True,
            "globality_only_inside_six_amplitude_fixed_space": True,
        },
    }


__all__ = [
    "build_static_symmetry_connection_certificate",
    "evaluate_static_symmetry_connection",
]
