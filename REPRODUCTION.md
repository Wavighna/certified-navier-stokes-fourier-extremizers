# Reproduction and audit guide

This repository is a curated, file-by-file release for the static Fourier
variational paper.  It deliberately excludes ZIP archives, temporary caches,
unrelated Euler notes, and exploratory Navier--Stokes search runs.

## Read first

1. `Wuthoo_2026_Certified_Navier_Stokes_Fourier_Extremizers.pdf` states the
   results and their mathematical scope.
2. `manuscript/REVIEWER_GUIDE.md` maps each theorem to its source and
   certificate.
3. `proofs/Wuthoo_2026_Verification_Summary.json` records the archived
   256- and 512-bit verification results.

## Recompute the certificates

Use Python 3.12 with the dependencies in `pyproject.toml`, including
`python-flint==0.9.0`:

```powershell
python -m pip install -e ".[proof]"
python scripts/verify_certificates.py
```

The command rebuilds the finite-dimensional certificates and writes
`proofs/Wuthoo_2026_Verification_Summary.recomputed.json`.

## Repository map

- `manuscript/` contains LaTeX source, bibliography, reviewer guide,
  pre-submission audit, and the paper figure.
- `proofs/` contains the individual JSON certificate payloads and archived
  verification records.
- `src/extreme_flows/` contains the exact Fourier, interval, symmetry, and
  reduced-branch code used by the checker.
- `scripts/` contains the certificate builders and public verification entry
  point.

## Scope

The results are rigorous, computer-assisted statements in a prescribed
64-real-dimensional low-Fourier-mode class.  They are not a proof of global
regularity, finite-time blow-up, or a full 3D Navier--Stokes extremum.
