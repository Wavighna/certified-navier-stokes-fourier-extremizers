# Paper A pre-submission audit

Updated: 2026-08-12. This checklist governs the theorem paper only. It does
not treat the finite-horizon depletion search as evidence for any theorem.

## Claims supported by archived evidence

- Exact six- and two-amplitude fixed-space reductions in the stated 16-pair,
  64-real-coordinate Fourier class.
- Global constrained maximum only in the six-amplitude fixed space, at the
  specified parameter point, with exact face exhaustion and Arb/Sturm checks.
- Strict local maxima modulo translations in the entire fixed 64-coordinate
  class at the stated points, via 68-variable Krawczyk inclusion and certified
  bordered inertia.
- A uniform strict-local-maximality theorem for the exact competing branch
  for every $E,\nu>0$, via exact symmetry-block/Sturm continuation, with an
  independent cancellation-exact low-parameter cover joined to a direct
  high-parameter cover at 256/512-bit precision.
- A uniform strict-local-maximality theorem for the high-$\eta$ six-amplitude
  branch in the full 64-coordinate class for every $\eta>1867/4$, by a
  2,048-box compact interval-continuation certificate at 256/512-bit
  precision. This remains a local, finite-dimensional statement.
- A direct inviscid endpoint theorem: the exact six-amplitude radical field is
  a strict local maximizer of the full 64-coordinate cubic stretching problem
  modulo translations, at every positive enstrophy by scaling.
- A second direct inviscid endpoint theorem: the exact two-amplitude branch is
  another strict full-space local maximizer. Their explicit positive value gap
  certifies coexistence but does not claim a complete local classification.
- Exact rational sum-of-squares bounds for the $(1,1,2)$ and $(2,2,4)$ shell
  components, $A_{112}^2\leq(4/3)E_1^2E_2$ and $A_{224}^2\leq E_2^2E_4$.
  These are neither sharpness claims nor full-cubic/global-optimizer claims.
- An explicit two-amplitude KKT branch and one crossover **between the two
  symmetry branches**, not in the full class.

The authoritative one-command record is
`artifacts/proofs/paper_a_verification.json`. It deliberately reports
`full_64d_global_maximum=false` and
`navier_stokes_regularity_or_blowup_statement=false`.

## Required gates before submission

- [ ] A researcher not involved in constructing the checker reruns the
      verifier from a clean checkout and audits the Fourier conventions,
      symmetry action, KKT equations, and interval-inclusion logic.
- [ ] Search MathSciNet/zbMATH/Web of Science and the relevant arXiv
      categories for the exact two-amplitude formula, symmetry group, and
      fixed-enstrophy Fourier extremizer. Record query dates and exclusions.
      The public-screening baseline is `paper/PRIORITY_SEARCH_LOG.md`; it is
      not a substitute for this gate. `paper/PRIORITY_COMPARISON.md` records
      the current claim-level comparison with the closest papers.
- [ ] Compare the manuscript line by line with Lu--Doering (2008),
      Ayala--Protas (2017), and both Kiriukhin (2026) preprints; narrow any
      priority wording that overlaps them.
- [ ] Build the LaTeX source with a pinned TeX environment and visually review
      every page, bibliography entry, and displayed formula.
- [ ] Freeze a public archive: exact repository revision, Python/FLINT
      dependency hashes, proof JSON files, the one-command verifier, and
      a README with expected runtime and outputs.
- [ ] Have a fluid-dynamics/numerical-analysis colleague read the title,
      abstract, and conclusion specifically for accidental PDE-level claims.

## Explicitly forbidden claims

- A global optimum in the full 64-coordinate class.
- A statement beyond the fixed Fourier cutoff.
- A Navier--Stokes regularity, singularity, or turbulence theorem.
- Exclusive priority based only on a failed search.
- "Independent verification" merely because the same checker was run at two
  precisions. These are separate precision runs, not independent software.

## Current readiness

The manuscript has a reproducible theorem core and is suitable for external
expert circulation as a **submission draft**. A fresh temporary-tree rerun in
the same pinned Python environment passed on 2026-08-12; its conservative
record is `artifacts/proofs/paper_a_clean_tree_audit.json`. That is a packaging
check, not an independent human or software audit, so it does not discharge
the first required gate. It is not submission-ready until each required gate
is documented. The numerical depletion campaign remains a separate prospective
paper and currently has no positive anti-depletion result.
