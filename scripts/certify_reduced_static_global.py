"""Certify the global static maximum inside the six-amplitude ansatz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extreme_flows.reduced_static_global import build_global_static_certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/proofs/reduced_static_global_arb_certificate.json",
    )
    parser.add_argument("--precisions", nargs="+", type=int, default=(256, 512))
    parser.add_argument("--bits", type=int, default=100)
    args = parser.parse_args()
    payload = build_global_static_certificate(
        precisions=args.precisions, bits=args.bits
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"truth label: {payload['truth_label']}")
    if not payload["claims"]["global_maximum_within_six_amplitude_static_ansatz"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
