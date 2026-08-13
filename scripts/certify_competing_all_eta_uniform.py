"""Prove full-class local maximality of the competing branch for all eta>0.

The low-s cover uses an algebraically rescaled bordered Hessian.  This removes
the apparent 1/s singularity before interval evaluation; it is congruent to
the ordinary bordered Hessian for every s>0.  The existing high-s cover is
recomputed directly.  The covers overlap and exhaust 0<s<1.
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
try:
    from scripts.certify_competing_high_eta_uniform import (
        _branch_bordered_hessian as _ordinary_bordered_hessian,
        _interval as _high_interval,
    )
except ModuleNotFoundError:
    from certify_competing_high_eta_uniform import (
        _branch_bordered_hessian as _ordinary_bordered_hessian,
        _interval as _high_interval,
    )


ROOT = Path(__file__).resolve().parents[1]
LOW_INTERVAL_COUNT = 188
LOW_INTERVAL_WIDTH = Fraction(1, 200)
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


def _low_interval(index: int) -> tuple[Fraction, Fraction]:
    if not 0 <= index < LOW_INTERVAL_COUNT:
        raise IndexError(index)
    return index * LOW_INTERVAL_WIDTH, (index + 1) * LOW_INTERVAL_WIDTH


def _branch_tensors(model: StaticLowModePolynomial):
    """Return the exact a/b Hessian coefficients and the weak shell set."""

    primary = embed_competing(Fraction(1), Fraction(0))
    secondary = embed_competing(Fraction(0), Fraction(1))
    weak = frozenset(
        index
        for index in range(model.dimension)
        if sum(component * component for component in model.pairs[index // 4]) == 1
    )
    dimension = model.dimension
    tensor_a = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    tensor_b = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    for indices, coefficient in model.third_derivative_terms.items():
        for i, j, k in set(permutations(indices)):
            tensor_a[i][j] += coefficient * primary[k]
            tensor_b[i][j] += coefficient * secondary[k]
    if any(tensor_a[i][j] for i in weak for j in weak):
        raise AssertionError("a-Hessian has unexpected weak-shell self-coupling")
    return primary, secondary, weak, tensor_a, tensor_b


def _rescaled_low_bordered_hessian(
    model: StaticLowModePolynomial,
    lower: Fraction,
    upper: Fraction,
    data,
):
    """Build the cancellation-exact rescaled bordered Hessian on one s box.

    Let W be the |k|^2=1 control coordinates and D(s) scale W by 1/s.
    For s>0 this matrix is s D(s)^T B(s) D(s), where B is the ordinary
    symmetric bordered Hessian. Exact cancellation of P=E on W yields the
    finite formula below, which extends continuously to s=0.
    """

    from flint import arb, arb_mat

    primary, secondary, weak, tensor_a, tensor_b = data
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    s = arb(_arb_fraction(center), _arb_fraction(radius))
    a = (arb(3) - s * s).sqrt() / 3
    dimension = model.dimension
    size = dimension + 4
    bordered = [[arb(0) for _ in range(size)] for _ in range(size)]
    strong = frozenset(range(dimension)) - weak
    for i in range(dimension):
        for j in range(dimension):
            if i in weak and j in weak:
                value = _arb_fraction(tensor_b[i][j]) / 12
            elif i in strong and j in strong:
                value = (
                    s * a * _arb_fraction(tensor_a[i][j])
                    + s * s * _arb_fraction(tensor_b[i][j]) / 12
                )
            else:
                value = (
                    a * _arb_fraction(tensor_a[i][j])
                    + s * _arb_fraction(tensor_b[i][j]) / 12
                )
            bordered[i][j] = value
        e_diagonal = _arb_fraction(model.enstrophy_diagonal_exact[i])
        p_diagonal = _arb_fraction(model.palinstrophy_diagonal_exact[i])
        if i in weak:
            bordered[i][i] -= 2 * e_diagonal / 3
        else:
            bordered[i][i] += (
                e_diagonal
                - p_diagonal
                + s * s * (p_diagonal - 5 * e_diagonal / 3)
            )
        constraint = e_diagonal * (
            _arb_fraction(primary[i]) * a + _arb_fraction(secondary[i]) * s / 12
        )
        if i not in weak:
            constraint *= s
        bordered[i][dimension] = constraint
        bordered[dimension][i] = constraint
    for row, index in enumerate(model.phase_indices, start=dimension + 1):
        bordered[index][row] = arb(1)
        bordered[row][index] = arb(1)
    return arb_mat(bordered)


def _inertia_summary(matrix) -> tuple[bool, object, dict[str, int]]:
    inertia = _certify_symmetric_inertia_arb(matrix)
    valid = bool(
        inertia["verified"]
        and inertia["positive"] == 4
        and inertia["negative"] == 64
        and inertia["zero_or_unresolved"] == 0
    )
    return valid, inertia["minimum_signed_gershgorin_margin"], {
        "positive": int(inertia["positive"]),
        "negative": int(inertia["negative"]),
        "zero_or_unresolved": int(inertia["zero_or_unresolved"]),
    }


def _run(precision: int) -> dict[str, object]:
    from flint import ctx

    model = StaticLowModePolynomial(
        phase_waves=PHASE_WAVES, phase_polarizations=PHASE_POLARIZATIONS
    )
    data = _branch_tensors(model)
    old_precision = ctx.prec
    try:
        ctx.prec = precision
        low_failures = []
        high_failures = []
        margins = []
        for index in range(LOW_INTERVAL_COUNT):
            lower, upper = _low_interval(index)
            valid, margin, counts = _inertia_summary(
                _rescaled_low_bordered_hessian(model, lower, upper, data)
            )
            if valid:
                margins.append(margin)
            else:
                low_failures.append(
                    {"index": index, "lower_s": str(lower), "upper_s": str(upper), **counts}
                )
        for index in range(238):
            lower, upper = _high_interval(index)
            valid, margin, counts = _inertia_summary(
                _ordinary_bordered_hessian(model, lower, upper)
            )
            if valid:
                margins.append(margin)
            else:
                high_failures.append(
                    {"index": index, "lower_s": str(lower), "upper_s": str(upper), **counts}
                )
        return {
            "precision_bits": precision,
            "rescaled_low_cover": {
                "interval_count": LOW_INTERVAL_COUNT,
                "range": ["0", str(LOW_INTERVAL_COUNT * LOW_INTERVAL_WIDTH)],
                "all_intervals_certified": not low_failures,
                "failures": low_failures,
            },
            "ordinary_high_cover": {
                "interval_count": 238,
                "range": ["9049/10000", "1"],
                "all_intervals_certified": not high_failures,
                "failures": high_failures,
            },
            "all_intervals_certified": not low_failures and not high_failures,
            "minimum_signed_gershgorin_margin": str(min(margins, key=float)) if margins else None,
        }
    finally:
        ctx.prec = old_precision


def build_payload(precisions: Sequence[int] = (256, 512)) -> dict[str, object]:
    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires python-flint==0.9.0")
    precisions = _validated_proof_precisions(precisions)
    runs = [_run(precision) for precision in precisions]
    certified = all(run["all_intervals_certified"] for run in runs)
    return {
        "schema_version": 1,
        "truth_label": (
            "proved_uniform_competing_full_space_local_maximum_all_positive_eta"
            if certified
            else "uniform_all_eta_competing_local_maximum_certificate_incomplete"
        ),
        "statement": (
            "For every E,nu>0, the exact competing two-amplitude KKT branch is a "
            "strict local maximum modulo translations in the 64-real-control class. "
            "This is not a full-space global-maximality or PDE regularity statement."
        ),
        "normalization": {
            "unit_enstrophy_parameterization": "a=sqrt(3-s^2)/3, b=s/12, nu=(1-s^2)/(2s)",
            "eta_relation": "eta=4s^2/(1-s^2)^2 maps 0<s<1 bijectively to eta>0",
            "rescaled_low_cover_congruence": "N(s)=s D(s)^T B(s) D(s), D=s^-1 on |k|^2=1 controls",
            "low_cover_upper_endpoint": str(LOW_INTERVAL_COUNT * LOW_INTERVAL_WIDTH),
            "overlap": "47/50 > 9049/10000, so the two covers exhaust 0<s<1",
            "phase_chart_regular": "a^2=(3-s^2)/9>=2/9 for 0<=s<=1",
        },
        "arb": {"dependency": _python_flint_dependency_record(), "runs": runs},
        "checker_source": _record("scripts/certify_competing_all_eta_uniform.py"),
        "high_cover_source": _record("scripts/certify_competing_high_eta_uniform.py"),
        "formula_source": _record("src/extreme_flows/reduced_competing.py"),
        "polynomial_source": _record("src/extreme_flows/certify.py"),
        "claims": {
            "strict_full_space_local_maximum_modulo_translations_for_all_positive_eta": certified,
            "full_space_global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/proofs/competing_all_eta_uniform.json")
    parser.add_argument("--precisions", nargs="+", type=int, default=(256, 512))
    args = parser.parse_args()
    payload = build_payload(args.precisions)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["truth_label"])
    for run in payload["arb"]["runs"]:
        print(run["precision_bits"], run["all_intervals_certified"])
    if not payload["claims"]["strict_full_space_local_maximum_modulo_translations_for_all_positive_eta"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
