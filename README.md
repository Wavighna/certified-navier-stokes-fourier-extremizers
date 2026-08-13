# Certified Navier–Stokes Fourier Extremizers

**Avighna Wuthoo**  
Computer-assisted finite-dimensional variational results for a periodic three-dimensional Navier–Stokes Fourier class (2026).

## Start here

1. Read the **[paper](Wuthoo_2026_Certified_Navier_Stokes_Fourier_Extremizers.pdf)**.
2. Read **[the reproduction and audit guide](REPRODUCTION.md)**.
3. Use **[the reviewer guide](manuscript/REVIEWER_GUIDE.md)** to map every theorem to source code and its individual JSON certificate.

## Individual materials

- `manuscript/` — LaTeX source, bibliography, reviewer guide, pre-submission audit, and figure files.
- `proofs/` — individual interval, Krawczyk, Sturm, and algebra-audit JSON records, including the archived manifest and verification summary.
- `src/extreme_flows/` — exact Fourier algebra, symmetry reductions, and interval-certificate implementation.
- `scripts/` — certificate builders and the public verification entry point `verify_certificates.py`.
- `pyproject.toml` — pinned project dependencies, including `python-flint==0.9.0` in the `proof` extra.

There are no ZIP archives, caches, raw exploratory-run files, unrelated Euler plans, or abandoned optimization logs in this public release.

## Scope

The paper proves computer-assisted, **finite-dimensional** statements for a prescribed 16-pair / 64-real-coordinate Fourier model of instantaneous enstrophy production. Its strongest claims concern symmetry-reduced maxima and strict local maximality modulo translations in that fixed model.

It does **not** prove a full 64-dimensional global maximizer, global regularity, finite-time blow-up, turbulence theory, or any resolution of the 3D Navier–Stokes Millennium Problem.

## Citation

Avighna Wuthoo, *Certified Coexistence of Symmetry-Reduced Local Maximizers in a Periodic 3D Navier–Stokes Fourier Class*, 2026.