"""Exact resultant and interval certificate for the two symmetry-branch crossing."""

from __future__ import annotations

import json
import importlib.metadata
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import platform
import sys


# The nonconstant factor after eliminating b between equal eta and equal rate.
# Coefficients are descending in mu. There is exactly one positive root by
# Descartes' rule, since only the leading coefficient is positive.
CROSSING_POLYNOMIAL = (
    14283, -616032, -75884328, -2607893632, -48929993328,
    -584799255552, -4735225442048, -26610597679104,
    -103880151184128, -275978473299968, -474888508827648,
    -475951787311104, -210195161092096,
)


def _record(relative_path: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    data = path.read_bytes()
    return {
        "path": relative_path.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _dependency() -> dict[str, object]:
    return {
        "python_flint": importlib.metadata.version("python-flint"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _eval(values):
    """Return cleared equal-eta/equal-rate equations and exact Jacobian."""

    mu, b = values
    n_eta = 69 * mu**4 + 1696 * mu**3 + 14936 * mu**2 + 54656 * mu + 68368
    n_rate = 23 * mu**5 + 516 * mu**4 + 3840 * mu**3 + 8864 * mu**2 - 8432 * mu - 33728
    h = 144 * b**2 + 24 * b
    r = 768 * b**3 - 192 * b**2 - 48 * b
    f = n_eta - 32 * (mu + 8) ** 2 * h
    g = n_rate - 16 * (mu + 8) ** 2 * r
    df_mu = 276 * mu**3 + 5088 * mu**2 + 29872 * mu + 54656 - 64 * (mu + 8) * h
    df_b = -32 * (mu + 8) ** 2 * (288 * b + 24)
    dg_mu = 115 * mu**4 + 2064 * mu**3 + 11520 * mu**2 + 17728 * mu - 8432 - 32 * (mu + 8) * r
    dg_b = -16 * (mu + 8) ** 2 * (2304 * b**2 - 384 * b - 48)
    return (f, g), ((df_mu, df_b), (dg_mu, dg_b))


def _eta(mu):
    return (69 * mu**4 + 1696 * mu**3 + 14936 * mu**2 + 54656 * mu + 68368) / (32 * (mu + 8) ** 2)


def _rate(mu):
    return (23 * mu**5 + 516 * mu**4 + 3840 * mu**3 + 8864 * mu**2 - 8432 * mu - 33728) / (16 * (mu + 8) ** 2)


def _six_quartic(mu, eta):
    return (
        69 * mu**4 + 1696 * mu**3 + (14936 - 32 * eta) * mu**2
        + (54656 - 512 * eta) * mu + 68368 - 2048 * eta
    )


def _six_quartic_derivative(mu, eta):
    return 276 * mu**3 + 5088 * mu**2 + 2 * (14936 - 32 * eta) * mu + 54656 - 512 * eta


def ordering_anchor(eta: int, seed: str, precision: int):
    """Interval-certify the sign of competing rate minus six-branch rate."""

    from flint import arb, ctx

    old = ctx.prec
    try:
        ctx.prec = precision
        eta_ball = arb(eta)
        center = arb(seed)
        for _ in range(6):
            center = (center - _six_quartic(center, eta_ball) / _six_quartic_derivative(center, eta_ball)).mid()
        radius = (arb(2) ** -80 * max(abs(center), arb(1))).upper()
        box = arb(center, radius)
        newton = center - _six_quartic(center, eta_ball) / _six_quartic_derivative(box, eta_ball)
        if not box.contains_interior(newton) or _six_quartic_derivative(box, eta_ball).contains(0):
            return {"eta": eta, "verified": False}
        b = ((eta_ball + 1).sqrt() - 1) / 12
        competing_rate = 768 * b**3 - 192 * b**2 - 48 * b
        difference = competing_rate - _rate(box)
        return {
            "eta": eta,
            "verified": True,
            "six_branch_mu_box": str(box),
            "competing_minus_six_rate_box": str(difference),
            "competing_branch_higher": difference.lower() > 0,
            "six_amplitude_branch_higher": difference.upper() < 0,
        }
    finally:
        ctx.prec = old


def certify(precision: int) -> dict[str, object]:
    import numpy as np
    from flint import arb, arb_mat, ctx

    old = ctx.prec
    try:
        ctx.prec = precision
        # High-precision numerical seed, then all validation is Arb.
        point = [arb("109.6633012720475658941218200561192176673"), arb("13.85942463981629734253682530570015801013")]
        for _ in range(6):
            value, jac = _eval(point)
            correction = arb_mat(jac).solve(arb_mat([[value[0]], [value[1]]]))
            point = [(point[i] - correction[i, 0]).mid() for i in range(2)]
        value, jac = _eval(point)
        matrix = arb_mat(jac)
        inverse = matrix.inv()
        preconditioner = arb_mat([[inverse[i, j].mid() for j in range(2)] for i in range(2)])
        center = arb_mat([[x] for x in point])
        newton = center - preconditioner * arb_mat([[value[0]], [value[1]]])
        identity = arb_mat([[arb(1), arb(0)], [arb(0), arb(1)]])
        attempts = []
        selected = None
        for exponent in (80, 60, 50, 40, 30):
            radii = [(arb(2) ** (-exponent) * max(abs(x), arb(1))).upper() for x in point]
            box = [arb(point[i], radii[i]) for i in range(2)]
            _, jac_box = _eval(box)
            defect = identity - preconditioner * arb_mat(jac_box)
            delta = arb_mat([[box[i] - point[i]] for i in range(2)])
            image = newton + defect * delta
            inclusion = [box[i].contains_interior(image[i, 0]) for i in range(2)]
            weights = [abs(box[i] - point[i]).abs_upper() for i in range(2)]
            rows = [
                sum((defect[i, j].abs_upper() * weights[j] for j in range(2)), arb(0)) / weights[i]
                for i in range(2)
            ]
            attempt = {
                "radius_power_of_two": -exponent,
                "strict_inclusion": all(inclusion),
                "contraction_upper": str(max(rows, key=float)),
                "contraction_strict": all(row < 1 for row in rows),
            }
            attempts.append(attempt)
            if attempt["strict_inclusion"] and attempt["contraction_strict"]:
                selected = (box, attempt)
                break
        if selected is None:
            return {"precision_bits": precision, "verified": False, "attempts": attempts}
        box, attempt = selected
        mu_box, b_box = box
        eta_box = _eta(mu_box)
        rate_box = _rate(mu_box)
        # Exact positive-root count / monotonicity claims are separate symbolic
        # facts, recorded alongside this two-variable existence proof.
        return {
            "precision_bits": precision,
            "verified": True,
            "attempts": attempts,
            "selected_attempt": attempt,
            "mu_box": str(mu_box),
            "b_box": str(b_box),
            "eta_box": str(eta_box),
            "common_rate_box": str(rate_box),
            "jacobian_determinant": str(matrix.det()),
        }
    finally:
        ctx.prec = old


def build_payload() -> dict[str, object]:
    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires python-flint==0.9.0")
    runs = [certify(bits) for bits in (256, 512)]
    anchors = {
        "eta_10000": [ordering_anchor(10_000, "63.83423736359512", bits) for bits in (256, 512)],
        "eta_40000": [ordering_anchor(40_000, "131.9229862057184", bits) for bits in (256, 512)],
    }
    ordering_proved = (
        all(run["verified"] and run["competing_branch_higher"] for run in anchors["eta_10000"])
        and all(run["verified"] and run["six_amplitude_branch_higher"] for run in anchors["eta_40000"])
    )
    return {
        "schema_version": 1,
        "truth_label": "proved_unique_high_eta_symmetry_branch_crossover",
        "dependency": _dependency(),
        "checker_source": {
            "crossover_checker": _record("scripts/certify_branch_crossover.py"),
            "six_branch_parameterization": _record(
                "src/extreme_flows/reduced_static_parameterized_global.py"
            ),
            "competing_branch_formula": _record(
                "src/extreme_flows/reduced_competing.py"
            ),
        },
        "resultant": {
            "variable": "mu",
            "coefficients_descending": list(CROSSING_POLYNOMIAL),
            "descarte_positive_root_count": 1,
            "existence": "P(0)<0 and leading coefficient is positive",
            "eta_monotonicity": "eta'(mu)=(69mu^4+1952mu^3+20352mu^2+92160mu+150256)/(16(mu+8)^3)>0",
        },
        "arb_runs": runs,
        "ordering_anchors": anchors,
        "claims": {
            "unique_positive_resultant_root": True,
            "unique_crossing_on_the_high_eta_six_amplitude_branch": all(run["verified"] for run in runs),
            "ordering_on_either_side": ordering_proved,
            "full_64d_global_ordering": False,
        },
        "limitations": {
            "The crossing compares the two symmetry-reduced KKT branches, not the global 64D problem.": True,
            "Ordering is proved only along the two symmetry-reduced KKT branches.": True,
        },
    }


def main() -> None:
    payload = build_payload()
    output = Path("artifacts/proofs/symmetry_branch_crossover.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for run in payload["arb_runs"]:
        print(run["precision_bits"], run["verified"], run.get("eta_box"))
    if not payload["claims"]["unique_crossing_on_the_high_eta_six_amplitude_branch"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
