"""Certify the adaptive-chart static low-mode KKT point with Arb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extreme_flows.certify import build_adaptive_static_certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "candidate",
        nargs="?",
        default="artifacts/proofs/static_adaptive_kkt_candidate.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/proofs/static_adaptive_arb_certificate.json",
    )
    parser.add_argument("--precisions", nargs="+", type=int, default=(256, 512))
    args = parser.parse_args()

    payload = build_adaptive_static_certificate(
        args.candidate, precisions=args.precisions
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"truth label: {payload['truth_label']}")
    for run in payload["arb"]["runs"]:
        inertia = run.get("bordered_inertia", {})
        print(
            f"{run['precision_bits']} bits: Krawczyk={run.get('krawczyk_verified')}, "
            f"inertia=({inertia.get('positive')}, {inertia.get('negative')}, "
            f"{inertia.get('zero_or_unresolved')})"
        )
    certified = bool(
        payload["claims"][
            "strict_local_maximum_modulo_translations_in_adaptive_chart"
        ]
    )
    print(
        "strict local maximum modulo translations: "
        + ("CERTIFIED" if certified else "NOT CERTIFIED")
    )
    if not certified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
