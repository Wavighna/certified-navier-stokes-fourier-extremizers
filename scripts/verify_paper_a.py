"""One-command proof verification manifest for the static-extremizer paper.

This rebuilds the three proof payloads in memory at 256 and 512 bits.  It is a
reproducibility verifier, not a replacement for reading the mathematical proof
or independently auditing the checker implementation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import importlib.metadata

from extreme_flows.reduced_static_global import build_global_static_certificate
from extreme_flows.certify import build_adaptive_static_certificate
from extreme_flows.static_symmetry_connection import (
    build_static_symmetry_connection_certificate,
)
from extreme_flows.reduced_static_parameterized import (
    UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD,
    high_eta_interior_branch_theorem,
)
from extreme_flows.reduced_static_parameterized_global import (
    high_eta_global_ansatz_theorem,
    six_amplitude_inviscid_extremizer,
    symmetry_branch_crossover_asymptotics,
    symmetry_branch_large_eta_efficiency,
)
from extreme_flows.shellwise_bounds import (
    shell112_sos_certificate,
    shell123_sos_certificate,
    shell224_sos_certificate,
    shell334_sos_certificate,
)
try:  # package import under pytest / installed-package execution
    from scripts.certify_high_static_branch import certify as certify_high_static_branch
    from scripts.certify_inviscid_six_local import build_certificate as build_inviscid_six_local_certificate
    from scripts.certify_inviscid_competing_local import build_certificate as build_inviscid_competing_local_certificate
    from scripts.certify_six_high_eta_uniform import build_payload as build_uniform_six_payload
    from scripts.certify_reduced_competing import exact_full_kkt_reduction
    from scripts.certify_competing_symmetry_group import build_payload as build_competing_group_payload
    from scripts.certify_competing_high_eta_uniform import build_payload as build_uniform_competing_payload
    from scripts.certify_competing_all_eta_uniform import build_payload as build_all_eta_competing_payload
    from scripts.certify_branch_crossover import build_payload as build_crossover_payload
    from scripts.derive_competing_hessian_blocks import build_payload as build_hessian_block_payload
    from scripts.certify_competing_all_eta_symbolic import build_payload as build_symbolic_continuation_payload
    from scripts.audit_competing_standalone import audit as standalone_competing_audit
    from scripts.audit_reduced_static_standalone import audit as standalone_static_audit
except ModuleNotFoundError:  # direct ``python scripts/verify_paper_a.py``
    from certify_high_static_branch import certify as certify_high_static_branch
    from certify_inviscid_six_local import build_certificate as build_inviscid_six_local_certificate
    from certify_inviscid_competing_local import build_certificate as build_inviscid_competing_local_certificate
    from certify_six_high_eta_uniform import build_payload as build_uniform_six_payload
    from certify_reduced_competing import exact_full_kkt_reduction
    from certify_competing_symmetry_group import build_payload as build_competing_group_payload
    from certify_competing_high_eta_uniform import build_payload as build_uniform_competing_payload
    from certify_competing_all_eta_uniform import build_payload as build_all_eta_competing_payload
    from certify_branch_crossover import build_payload as build_crossover_payload
    from derive_competing_hessian_blocks import build_payload as build_hessian_block_payload
    from certify_competing_all_eta_symbolic import build_payload as build_symbolic_continuation_payload
    from audit_competing_standalone import audit as standalone_competing_audit
    from audit_reduced_static_standalone import audit as standalone_static_audit


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts" / "proofs" / "static_adaptive_kkt_candidate.json"
HIGH_BRANCH_CANDIDATE = ROOT / "artifacts" / "proofs" / "high_branch_eta10000_candidate.json"


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def verify() -> dict[str, object]:
    """Recompute the Paper A certificate claims from current source code."""

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("Paper A verification requires python-flint==0.9.0")
    global_payload = build_global_static_certificate(precisions=(256, 512))
    local_payload = build_adaptive_static_certificate(
        CANDIDATE, precisions=(256, 512)
    )
    connection_payload = build_static_symmetry_connection_certificate(
        CANDIDATE, precisions=(256, 512)
    )
    parameterized_payload = high_eta_interior_branch_theorem(
        UNIQUE_ADMISSIBLE_BRANCH_ETA_THRESHOLD + Fraction(1, 1_000)
    )
    parameterized_global_payload = high_eta_global_ansatz_theorem()
    asymptotic_payload = symmetry_branch_large_eta_efficiency()
    crossover_asymptotics_payload = symmetry_branch_crossover_asymptotics()
    inviscid_six_payload = six_amplitude_inviscid_extremizer()
    high_branch_payload = certify_high_static_branch(
        HIGH_BRANCH_CANDIDATE,
        ROOT / "artifacts" / "proofs" / "high_branch_eta10000_arb_certificate.json",
        (256, 512),
    )
    inviscid_six_local_payload = build_inviscid_six_local_certificate((256, 512))
    inviscid_competing_local_payload = build_inviscid_competing_local_certificate((256, 512))
    uniform_six_payload = build_uniform_six_payload(count=2048, precisions=(256, 512))
    shell112_payload = shell112_sos_certificate()
    shell123_payload = shell123_sos_certificate()
    shell224_payload = shell224_sos_certificate()
    shell334_payload = shell334_sos_certificate()
    competing_formula_payload = exact_full_kkt_reduction()
    competing_group_payload = build_competing_group_payload()
    uniform_competing_payload = build_uniform_competing_payload()
    all_eta_competing_payload = build_all_eta_competing_payload()
    crossover_payload = build_crossover_payload()
    hessian_block_payload = build_hessian_block_payload()
    symbolic_continuation_payload = build_symbolic_continuation_payload(
        hessian_block_payload["derivation"]
    )
    standalone_payload = standalone_competing_audit()
    standalone_static_payload = standalone_static_audit()
    claims = {
        "global_static_maximum_inside_six_amplitude_ansatz": bool(
            global_payload["claims"]["global_maximum_within_six_amplitude_static_ansatz"]
            and standalone_static_payload["all_coefficients_match"]
        ),
        "strict_local_full_64d_maximum_modulo_translations": bool(
            local_payload["claims"]["strict_local_maximum_modulo_translations_in_adaptive_chart"]
        ),
        "reduced_global_root_identified_with_full_local_root": bool(
            connection_payload["claims"]["reduced_global_ansatz_root_is_the_full_certified_kkt_root"]
        ),
        "unique_admissible_positive_interior_branch_for_eta_gt_1867_over_4": bool(
            parameterized_payload["threshold_met"]
            and parameterized_payload["descartes_positive_root_count"] == 1
            and parameterized_payload["root_exceeds_sqrt_28"]
            and parameterized_payload["positive_a2_d2_f2"]
        ),
        "global_six_amplitude_ansatz_maximum_for_eta_gt_1867_over_4": bool(
            parameterized_global_payload["claims"][
                "global_static_maximum_within_six_amplitude_ansatz_for_eta_gt_1867_over_4"
            ]
        ),
        "higher_competing_full_64d_local_maximum_at_eta_10000": bool(
            high_branch_payload["claims"]["strict_local_maximum_modulo_translations"]
            and high_branch_payload["claims"]["strictly_beats_six_amplitude_ansatz"]
        ),
        "higher_branch_exact_two_amplitude_full_kkt_formula_for_all_positive_E_nu": bool(
            competing_formula_payload["all_full_kkt_remainders_vanish"]
            and standalone_payload["all_coefficients_match"]
        ),
        "competing_ansatz_lattice_symmetry_group_is_A4_times_C2": bool(
            competing_group_payload["claims"]["generated_group_is_A4_times_C2"]
        ),
        "uniform_high_eta_competing_full_64d_local_maximum_modulo_translations": bool(
            uniform_competing_payload["claims"][
                "strict_full_space_local_maximum_modulo_translations_for_all_eta_ge_100"
            ]
        ),
        "uniform_all_eta_competing_full_64d_local_maximum_modulo_translations": bool(
            all_eta_competing_payload["claims"][
                "strict_full_space_local_maximum_modulo_translations_for_all_positive_eta"
            ]
            and symbolic_continuation_payload["claims"][
                "strict_full_space_local_maximum_modulo_translations_for_all_positive_eta"
            ]
        ),
        "unique_ordered_crossover_between_the_two_high_eta_symmetry_branches": bool(
            crossover_payload["claims"]["unique_crossing_on_the_high_eta_six_amplitude_branch"]
            and crossover_payload["claims"]["ordering_on_either_side"]
        ),
        "six_amplitude_branch_has_strictly_higher_high_eta_efficiency_than_competing_branch": bool(
            asymptotic_payload["strict_ordering"]["six_amplitude_branch_eventually_higher"]
        ),
        "exact_inviscid_six_amplitude_cubic_global_maximizer_inside_its_fixed_space": bool(
            inviscid_six_payload["claims"]["exact_inviscid_six_amplitude_cubic_maximizer"]
            and inviscid_six_payload["claims"][
                "global_maximum_within_six_amplitude_inviscid_cubic_problem"
            ]
            and not inviscid_six_payload["claims"]["full_64d_inviscid_cubic_global_maximum"]
        ),
        "exact_inviscid_six_amplitude_full_64d_local_maximizer_modulo_translations": bool(
            inviscid_six_local_payload["claims"][
                "strict_local_maximum_modulo_translations_for_every_positive_E"
            ]
        ),
        "exact_inviscid_competing_branch_full_64d_local_maximizer_modulo_translations": bool(
            inviscid_competing_local_payload["claims"][
                "strict_local_maximum_modulo_translations_for_every_positive_E"
            ]
        ),
        "uniform_high_eta_six_branch_full_64d_local_maximum_modulo_translations": bool(
            uniform_six_payload["claims"][
                "strict_full_space_local_maximum_modulo_translations_for_eta_gt_1867_over_4"
            ]
        ),
        "exact_shell112_enstrophy_production_bound": bool(
            shell112_payload["claims"]["shell112_bound_proved"]
            and not shell112_payload["claims"]["shell112_constant_sharp_proved"]
            and not shell112_payload["claims"]["full_64d_inviscid_globality_proved"]
        ),
        "exact_shell123_enstrophy_production_bound": bool(
            shell123_payload["claims"]["shell123_bound_proved"]
            and not shell123_payload["claims"]["shell123_constant_sharp_proved"]
            and not shell123_payload["claims"]["full_64d_inviscid_globality_proved"]
        ),
        "exact_shell224_enstrophy_production_bound": bool(
            shell224_payload["claims"]["shell224_bound_proved"]
            and not shell224_payload["claims"]["shell224_constant_sharp_proved"]
            and not shell224_payload["claims"]["full_64d_inviscid_globality_proved"]
        ),
        "exact_shell334_enstrophy_production_bound": bool(
            shell334_payload["claims"]["shell334_bound_proved"]
            and not shell334_payload["claims"]["shell334_constant_sharp_proved"]
            and not shell334_payload["claims"]["full_64d_inviscid_globality_proved"]
        ),
        "two_distinct_exact_inviscid_full_64d_local_maxima_coexist": bool(
            inviscid_six_local_payload["claims"][
                "strict_local_maximum_modulo_translations_for_every_positive_E"
            ]
            and inviscid_competing_local_payload["claims"][
                "strict_local_maximum_modulo_translations_for_every_positive_E"
            ]
        ),
        "two_term_asymptotics_explain_the_large_scale_of_the_certified_branch_switch": bool(
            crossover_asymptotics_payload["two_term_crossover_predictor"]["formula"]
            == "7056/(23-2*sqrt(138))^2"
            and not crossover_asymptotics_payload["full_64d_global_ordering"]
        ),
        "competing_branch_generalized_hessian_has_exact_A4_times_C2_spectral_blocks": bool(
            hessian_block_payload["claims"]["exact_generalized_hessian_sector_factorization"]
        ),
        "competing_branch_symbolic_spectral_continuation_separately_recovers_uniform_local_maximality": bool(
            symbolic_continuation_payload["claims"][
                "strict_full_space_local_maximum_modulo_translations_for_all_positive_eta"
            ]
        ),
        "full_64d_global_maximum": False,
        "navier_stokes_regularity_or_blowup_statement": False,
    }
    if not all(value for key, value in claims.items() if key not in {
        "full_64d_global_maximum",
        "navier_stokes_regularity_or_blowup_statement",
    }):
        raise AssertionError("a required Paper A proof claim did not verify")
    return {
        "schema_version": 1,
        "truth_label": "paper_a_reproducibility_verification",
        "scope": "recomputes finite-dimensional static proof claims only",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "python_flint": importlib.metadata.version("python-flint"),
        },
        "inputs": {
            "full_static_candidate": _record(CANDIDATE),
            "high_branch_candidate": _record(HIGH_BRANCH_CANDIDATE),
            "inviscid_six_local_certificate": _record(
                ROOT / "artifacts" / "proofs" / "inviscid_six_full64_local_certificate.json"
            ),
            "inviscid_competing_local_certificate": _record(
                ROOT / "artifacts" / "proofs" / "inviscid_competing_full64_local_certificate.json"
            ),
            "uniform_high_eta_six_branch": _record(
                ROOT / "artifacts" / "proofs" / "six_branch_high_eta_uniform.json"
            ),
            "shell112_sos_certificate": _record(
                ROOT / "artifacts" / "proofs" / "shell112_sos_certificate.json"
            ),
            "shell123_sos_certificate": _record(
                ROOT / "artifacts" / "proofs" / "shell123_sos_certificate.json"
            ),
            "shell224_sos_certificate": _record(
                ROOT / "artifacts" / "proofs" / "shell224_sos_certificate.json"
            ),
            "shell334_sos_certificate": _record(
                ROOT / "artifacts" / "proofs" / "shell334_sos_certificate.json"
            ),
            "high_branch_reduced_formula": _record(
                ROOT / "artifacts" / "proofs" / "reduced_competing_formula.json"
            ),
            "competing_symmetry_group": _record(
                ROOT / "artifacts" / "proofs" / "competing_symmetry_group.json"
            ),
            "uniform_high_eta_competing": _record(
                ROOT / "artifacts" / "proofs" / "competing_high_eta_uniform.json"
            ),
            "uniform_all_eta_competing": _record(
                ROOT / "artifacts" / "proofs" / "competing_all_eta_uniform.json"
            ),
            "symmetry_branch_crossover": _record(
                ROOT / "artifacts" / "proofs" / "symmetry_branch_crossover.json"
            ),
            "competing_hessian_spectral_blocks": _record(
                ROOT / "artifacts" / "proofs" / "competing_hessian_spectral_blocks.json"
            ),
            "competing_all_eta_symbolic_continuation": _record(
                ROOT / "artifacts" / "proofs" / "competing_all_eta_symbolic_continuation.json"
            ),
            "standalone_competing_algebra_audit": _record(
                ROOT / "artifacts" / "proofs" / "competing_standalone_algebra_audit.json"
            ),
            "standalone_six_amplitude_algebra_audit": _record(
                ROOT / "artifacts" / "proofs" / "reduced_static_standalone_algebra_audit.json"
            ),
        },
        "checker_sources": {
            "verifier": _record(Path(__file__).resolve()),
            "full_static_certificate": _record(
                ROOT / "src" / "extreme_flows" / "certify.py"
            ),
            "reduced_global_certificate": _record(
                ROOT / "src" / "extreme_flows" / "reduced_static_global.py"
            ),
            "symmetry_connection": _record(
                ROOT / "src" / "extreme_flows" / "static_symmetry_connection.py"
            ),
            "symmetry_definition": _record(
                ROOT / "src" / "extreme_flows" / "static_symmetry.py"
            ),
            "reduced_formula": _record(
                ROOT / "src" / "extreme_flows" / "reduced_static.py"
            ),
            "parameterized_branch": _record(
                ROOT / "src" / "extreme_flows" / "reduced_static_parameterized.py"
            ),
            "parameterized_global_branch": _record(
                ROOT
                / "src"
                / "extreme_flows"
                / "reduced_static_parameterized_global.py"
            ),
            "high_branch_certificate": _record(
                ROOT / "scripts" / "certify_high_static_branch.py"
            ),
            "inviscid_six_local_certificate": _record(
                ROOT / "scripts" / "certify_inviscid_six_local.py"
            ),
            "inviscid_competing_local_certificate": _record(
                ROOT / "scripts" / "certify_inviscid_competing_local.py"
            ),
            "shell112_sos_certificate": _record(
                ROOT / "scripts" / "certify_shell112_sos.py"
            ),
            "shell123_sos_certificate": _record(
                ROOT / "scripts" / "certify_shell123_sos.py"
            ),
            "shell224_sos_certificate": _record(
                ROOT / "scripts" / "certify_shell224_sos.py"
            ),
            "shell334_sos_certificate": _record(
                ROOT / "scripts" / "certify_shell334_sos.py"
            ),
            "shellwise_bound_algebra": _record(
                ROOT / "src" / "extreme_flows" / "shellwise_bounds.py"
            ),
            "high_branch_formula": _record(
                ROOT / "scripts" / "certify_reduced_competing.py"
            ),
            "competing_symmetry_group": _record(
                ROOT / "scripts" / "certify_competing_symmetry_group.py"
            ),
            "uniform_high_eta_competing": _record(
                ROOT / "scripts" / "certify_competing_high_eta_uniform.py"
            ),
            "uniform_all_eta_competing": _record(
                ROOT / "scripts" / "certify_competing_all_eta_uniform.py"
            ),
            "symmetry_branch_crossover": _record(
                ROOT / "scripts" / "certify_branch_crossover.py"
            ),
            "competing_hessian_spectral_blocks": _record(
                ROOT / "scripts" / "derive_competing_hessian_blocks.py"
            ),
            "competing_all_eta_symbolic_continuation": _record(
                ROOT / "scripts" / "certify_competing_all_eta_symbolic.py"
            ),
            "standalone_competing_algebra_audit": _record(
                ROOT / "scripts" / "audit_competing_standalone.py"
            ),
            "standalone_six_amplitude_algebra_audit": _record(
                ROOT / "scripts" / "audit_reduced_static_standalone.py"
            ),
        },
        "precisions": [256, 512],
        "claims": claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="artifacts/proofs/paper_a_verification.json"
    )
    args = parser.parse_args()
    payload = verify()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for claim, value in payload["claims"].items():
        print(f"{claim}={value}")


if __name__ == "__main__":
    main()
