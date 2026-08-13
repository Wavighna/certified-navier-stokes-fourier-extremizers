"""Exact symbolic continuation proof for uniform competing-branch stability.

This complements the Arb interval cover.  The exact ``A4 x C2`` generalized
Hessian factorization has no nontranslation zero on ``0<s<1``.  Since the
operator is similar to a real symmetric matrix, an exact Sturm count at one
sample fixes the inertia throughout the connected parameter interval.  The
remaining tangent direction in the two-dimensional trivial-even sector has an
explicit negative curvature.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import sympy as sp

try:  # package import under pytest / installed-package execution
    from scripts.derive_competing_hessian_blocks import derive_block_polynomials
except ModuleNotFoundError:  # direct ``python scripts/...py``
    from derive_competing_hessian_blocks import derive_block_polynomials


ROOT = Path(__file__).resolve().parents[1]
S, T = sp.symbols("s t")
SAMPLE = sp.Rational(1, 2)
EXPECTED_SAMPLE_INERTIA = {
    "positive": 1,
    "negative": 60,
    "zero": 3,
}


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    data = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _expression(text: str) -> sp.Expr:
    return sp.cancel(sp.sympify(text, locals={"s": S, "t": T}))


def _remove_zero_factor(expression: sp.Expr) -> tuple[sp.Expr, int]:
    """Divide the exact characteristic factor by its complete power of ``t``."""

    expression = sp.cancel(expression)
    multiplicity = 0
    while sp.simplify(expression.subs(T, 0)) == 0:
        expression = sp.cancel(expression / T)
        multiplicity += 1
    return expression, multiplicity


def _open_unit_root_count(expression: sp.Expr) -> int:
    """Count exact real roots in ``0<s<1`` of a rational expression."""

    numerator, denominator = sp.fraction(sp.cancel(expression))
    def strip_endpoints(polynomial: sp.Poly) -> sp.Poly:
        """Remove roots at 0 or 1; the continuation interval is open."""

        for endpoint_factor in (sp.Poly(S, S), sp.Poly(S - 1, S)):
            while polynomial.eval(0 if endpoint_factor.as_expr() == S else 1) == 0:
                polynomial = polynomial.exquo(endpoint_factor)
        return polynomial

    numerator_polynomial = strip_endpoints(sp.Poly(numerator, S))
    denominator_polynomial = strip_endpoints(sp.Poly(denominator, S))
    numerator_roots = numerator_polynomial.count_roots(0, 1)
    denominator_roots = denominator_polynomial.count_roots(0, 1)
    if denominator_roots:
        raise ArithmeticError("a characteristic factor has a pole inside 0<s<1")
    return int(numerator_roots)


def _sample_root_counts(expression: sp.Expr) -> dict[str, int]:
    """Use exact Sturm counts at ``s=1/2``, retaining multiplicities."""

    specialization = sp.cancel(expression.subs(S, SAMPLE))
    numerator, denominator = sp.fraction(specialization)
    if denominator.has(T):
        raise ArithmeticError("unexpected t-dependent characteristic denominator")
    _, factors = sp.factor_list(numerator, T)
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for factor, multiplicity in factors:
        multiplicity = int(multiplicity)
        polynomial = sp.Poly(factor, T)
        zero = 0
        while polynomial.eval(0) == 0:
            polynomial = polynomial.exquo(sp.Poly(T, T))
            zero += 1
        # SymPy includes a boundary root in each closed-endpoint count.
        negative = int(polynomial.count_roots(-sp.oo, 0))
        positive = int(polynomial.count_roots(0, sp.oo))
        if negative + positive + zero != sp.Poly(factor, T).degree():
            raise ArithmeticError("sample spectrum has a nonreal or uncounted root")
        counts["negative"] += multiplicity * negative
        counts["positive"] += multiplicity * positive
        counts["zero"] += multiplicity * zero
    return counts


def _trivial_even_tangent_curvature() -> sp.Expr:
    """Return the exact curvature on the E-tangent trivial-even direction.

    In amplitudes ``(a,b)`` take ``w=(16b,-a)``, so
    ``3 a w_a+48 b w_b=0``.  Substitution of the branch formulas gives the
    displayed expression below.
    """

    return 32 * (S * S - 3) * (S * S + 1) / (3 * S)


def build_payload(derived: dict[str, Any] | None = None) -> dict[str, object]:
    """Build the exact symbolic continuation certificate.

    ``derived`` may be supplied by the block-factor verifier so a one-command
    verification does not recompute its generic symbolic determinant.
    """

    if sp.__version__ != "1.14.0":
        raise RuntimeError("symbolic continuation certificate requires sympy==1.14.0")
    if derived is None:
        derived = derive_block_polynomials()
    blocks = derived["blocks"]
    geometry = derived["constraint_geometry"]
    if not geometry["nontrivial_sectors_are_enstrophy_tangent"]:
        raise ArithmeticError("a nontrivial symmetry sector is not E-tangent")
    if not geometry["all_translation_basis_tangents_in_odd_three_dimensional_sector"]:
        raise ArithmeticError("translation tangent left the odd three-dimensional sector")
    if geometry["branch_translation_gram_determinant"] != "8*(3+s^2)^3/729":
        raise ArithmeticError("translation Gram determinant changed")
    zero_free: dict[str, dict[str, object]] = {}
    sample_blocks: dict[str, dict[str, int]] = {}
    total = {"positive": 0, "negative": 0, "zero": 0}
    for name, block in blocks.items():
        characteristic = _expression(str(block["characteristic_polynomial"]))
        reduced, translation_multiplicity = _remove_zero_factor(characteristic)
        root_count = _open_unit_root_count(reduced.subs(T, 0))
        if root_count:
            raise ArithmeticError(f"nontranslation zero crossing in {name}")
        zero_free[name] = {
            "translation_zero_multiplicity": translation_multiplicity,
            "nontranslation_t_zero_root_count_on_open_unit_s_interval": root_count,
        }
        sample = _sample_root_counts(characteristic)
        sample_blocks[name] = sample
        for sign in total:
            total[sign] += sample[sign]
    if total != EXPECTED_SAMPLE_INERTIA:
        raise ArithmeticError(f"unexpected exact sample inertia {total}")
    tangent_curvature = sp.factor(_trivial_even_tangent_curvature())
    # For 0<s<1: 32/(3s)>0, s^2-3<0, and s^2+1>0.
    tangent_negative = bool(
        sp.Poly(sp.factor(sp.together(tangent_curvature * 3 * S / 32)), S)
        == sp.Poly((S * S - 3) * (S * S + 1), S)
    )
    if not tangent_negative:
        raise ArithmeticError("trivial-even tangent curvature formula changed")
    return {
        "schema_version": 1,
        "truth_label": "exact_symbolic_continuation_certificate_for_uniform_competing_branch_local_maximality",
        "statement": (
            "For every E,nu>0, the competing branch is a strict local maximum "
            "modulo translations in the fixed 64-real-control Fourier class. "
            "This exact symbolic continuation proof is finite dimensional and "
            "does not claim full-space global maximality or any PDE statement."
        ),
        "parameter_interval": {
            "s_range": "0<s<1",
            "eta_relation": "eta=4s^2/(1-s^2)^2",
            "generalized_operator": "D_E^(-1)[Hessian(R)-lambda Hessian(E)]",
            "self_adjoint_similarity": "D_E^(-1)H is similar to D_E^(-1/2) H D_E^(-1/2), which is real symmetric",
        },
        "zero_crossing_certificate": zero_free,
        "exact_sample": {
            "s": str(SAMPLE),
            "sector_root_counts_with_multiplicity": sample_blocks,
            "full_generalized_hessian_inertia": total,
        },
        "tangent_trivial_even_sector": {
            "direction": "w=(16b,-a)",
            "curvature": str(tangent_curvature),
            "sign_on_open_unit_interval": "strictly negative",
        },
        "constraint_and_orbit_geometry": geometry,
        "logic": {
            "only_three_translation_zeros": (
                "The odd three-dimensional sector contributes exactly t^3; "
                "all other t=0 factors have no root on 0<s<1."
            ),
            "continuation": (
                "The generalized spectrum is real and continuous. Since no "
                "nontranslation eigenvalue crosses zero, the exact sample inertia "
                "holds throughout the connected open interval."
            ),
            "constraint_restriction": (
                "All nontrivial isotypic sectors are E-tangent. The remaining "
                "trivial-even tangent line has the displayed negative curvature. "
                "The odd three-dimensional sector carries precisely the three "
                "linearly independent translation tangents."
            ),
        },
        "checker_source": _record("scripts/certify_competing_all_eta_symbolic.py"),
        "factor_source": _record("scripts/derive_competing_hessian_blocks.py"),
        "polynomial_source": _record("src/extreme_flows/certify.py"),
        "symmetry_source": _record("src/extreme_flows/static_symmetry.py"),
        "formula_source": _record("src/extreme_flows/reduced_competing.py"),
        "claims": {
            "strict_full_space_local_maximum_modulo_translations_for_all_positive_eta": True,
            "full_space_global_maximum": False,
            "navier_stokes_regularity_or_blowup_statement": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/proofs/competing_all_eta_symbolic_continuation.json",
    )
    args = parser.parse_args()
    payload = build_payload()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["truth_label"])
    print(payload["exact_sample"]["full_generalized_hessian_inertia"])
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
