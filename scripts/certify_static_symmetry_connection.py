"""Certify identification of the reduced and full static KKT roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extreme_flows.static_symmetry_connection import (
    build_static_symmetry_connection_certificate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "candidate",
        nargs="?",
        default="artifacts/proofs/static_adaptive_kkt_candidate.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/proofs/static_symmetry_connection_certificate.json",
    )
    parser.add_argument("--precisions", nargs="+", type=int, default=(256, 512))
    parser.add_argument("--bits", type=int, default=100)
    args = parser.parse_args()
    payload = build_static_symmetry_connection_certificate(
        args.candidate, precisions=args.precisions, bits=args.bits
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"truth label: {payload['truth_label']}")
    if not payload["claims"]["reduced_global_ansatz_root_is_the_full_certified_kkt_root"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
