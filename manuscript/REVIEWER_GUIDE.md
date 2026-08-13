# Reviewer guide: certified static-extremizer paper

This document is an audit map for Paper A. It is deliberately limited to the
finite 16-pair Fourier variational problem. The project makes no claim about
an unrestricted Navier--Stokes extremum, global regularity, singularity
formation, or turbulence.

## Fast reproduction

With Python 3.12 and python-flint==0.9.0 installed, from the repository root:

    .venv\Scripts\python.exe scripts\verify_paper_a.py
    .venv\Scripts\python.exe scripts\build_paper_a_manifest.py

The first command recomputes all theorem-level certificates at 256 and 512
bits. On the reference machine it can take a few minutes while other numerical
jobs are active. The second hashes every file in the declared release set.
`scripts/audit_paper_a_clean_tree.py` additionally reruns this process from a
fresh temporary copy of the release tree; it is a same-environment packaging
check, not an independent audit.

## Claim-to-evidence map

| Claim | Primary construction | Archived certificate | What is proved |
|---|---|---|---|
| Six-amplitude formula and ansatz globality at E=100, nu=.01 | reduced_static.py; certify_reduced_static_global.py; audit_reduced_static_standalone.py | reduced_static_global_arb_certificate.json; reduced_static_standalone_algebra_audit.json | Literal independent Fourier convolution recovers the complete six-amplitude polynomial; global maximum only inside the six-amplitude fixed space. |
| Full-space local identification of that point | static_symmetry.py; static_symmetry_connection.py; certify.py | static_adaptive_arb_certificate.json; static_symmetry_connection_certificate.json | Strict local maximum modulo translations in the 64-real-coordinate class. |
| Parameterized high-eta six-branch theorem | reduced_static_parameterized.py; reduced_static_parameterized_global.py | parameterized_static_branch.json; parameterized_static_global.json | Unique admissible interior branch and six-space globality for eta>1867/4. |
| Uniform high-eta six branch | certify_six_high_eta_uniform.py | six_branch_high_eta_uniform.json | A 2,048-box compact cover at each of 256 and 512 bits uses certified midpoint bordered inertia and a Frobenius/Neumann homotopy bound to prove strict full-space local maximality modulo translations for every eta>1867/4. This is local, not global. |
| Shellwise stretching bounds | shellwise_bounds.py; certify_shell112_sos.py; certify_shell224_sos.py | shell112_sos_certificate.json; shell224_sos_certificate.json | Exact coefficient expansions and rational PSD Gram matrices prove $A_{112}^2\leq(4/3)E_1^2E_2$ and $A_{224}^2\leq E_2^2E_4$. Neither sharpness nor a bound on the full cubic is claimed. |
| Exact competing branch | reduced_competing.py; certify_reduced_competing.py; audit_competing_standalone.py | reduced_competing_formula.json; competing_standalone_algebra_audit.json | Exact two-dimensional fixed space, separately recomputed polynomial identity, full KKT branch, and closed rate formula. |
| Competing symmetry group | static_symmetry.py; certify_competing_symmetry_group.py | competing_symmetry_group.json | Exact 24-element lattice group, internally $A_4\times C_2$; no stability conclusion follows from this fact alone. |
| Competing control sectors | static_symmetry.py; certify_competing_representation.py | competing_representation_sectors.json | Exact isotypic ranks $(2,6,24)$ in each central-$C_2$ parity sector; a block-structure result, not a uniform Hessian result. |
| Competing generalized-Hessian factors | derive_competing_hessian_blocks.py; certify_competing_all_eta_symbolic.py | competing_hessian_spectral_blocks.json; competing_all_eta_symbolic_continuation.json | Exact $A_4\times C_2$ sector characteristic polynomials of $D_E^{-1}H_s$, followed by exact Sturm no-crossing and sample-inertia continuation. The Arb cover is a separate proof route, not an independent software audit. |
| Higher competing local point at eta=10^4 | certify_high_static_branch.py; certify.py | high_branch_eta10000_arb_certificate.json | Strict full-space local maximum modulo translations, not globality. |
| Uniform high-eta competing branch | certify_competing_high_eta_uniform.py; certify.py | competing_high_eta_uniform.json | 238 rational branch intervals at each of 256 and 512 bits prove strict full-space local maximality modulo translations for every eta>=100. |
| Uniform all-eta competing branch | certify_competing_all_eta_uniform.py; certify_competing_all_eta_symbolic.py | competing_all_eta_uniform.json; competing_all_eta_symbolic_continuation.json | The exact spectral continuation proof and a separate 188 cancellation-exact low-s plus 238 direct high-s interval route at each of 256 and 512 bits prove strict full-space local maximality modulo translations for every positive eta. |
| One crossover between branches | certify_branch_crossover.py | symmetry_branch_crossover.json | Ordering only along the two identified symmetry branches. |
| Exact high-eta efficiency gap | reduced_static_parameterized_global.py | paper/main.tex, Proposition `prop:efficiency` | The limits are $8\sqrt{138}/207$ and $4/9$; comparing their squares gives the exact positive gap $16/1863$. |
| Exact inviscid six-amplitude extremizer | reduced_static_parameterized_global.py | paper/main.tex, Proposition `prop:inviscidsix` | The high-eta branch converges to an explicit unit-enstrophy vector with $A=8\sqrt{138}/207$, globally maximizing the pure cubic stretching form only inside the six-amplitude fixed space. |
| Full-space inviscid local maximum | certify_inviscid_six_local.py | inviscid_six_full64_local_certificate.json | The exact radical six-amplitude field, after an exact quarter-period translation to a regular slice, is a strict local maximum of $A$ modulo translations in the complete 64-coordinate class; homogeneity extends the local statement to every positive enstrophy. |
| Competing inviscid local maximum and coexistence | certify_inviscid_competing_local.py | inviscid_competing_full64_local_certificate.json | The exact two-amplitude inviscid branch is a second strict local maximum of $A$ modulo translations in the complete class. Its exact value is $4E^{3/2}/9$, below the six-branch value by $(8\sqrt{138}-92)E^{3/2}/207>0$; this does not exhaust the local landscape. |
| Two-term crossover asymptotics | reduced_static_parameterized_global.py | paper/main.tex following `prop:efficiency` | Exact inversion gives predictor $7056/(23-2\sqrt{138})^2\approx28834.30$; explanatory only, not the crossing proof. |
| Branch-comparison figure | plot_static_branches.py | figures/symmetry_branch_rates.pdf | Numerical rendering of the exact reduced formulas; the displayed crossover is certified separately above. |

artifacts/proofs/paper_a_verification.json is the top-level record joining
these claims. paper_a_release_manifest.json binds the manuscript, scripts,
sources, candidate inputs, and proof objects by SHA-256.

## What to audit independently

1. Reconstruct the 16 Fourier pairs and transverse polarizations; compare the
   explicit convention in paper/main.tex with certify.py.
2. Check the two symmetry actions and their exact fixed-space dimensions.
3. Recompute the cubic convolution and the coefficient identity
   A=24 a^2 b, E=3 a^2+48 b^2, P=3 a^2+96 b^2.
4. Check the 64 KKT remainders modulo the two branch equations.
5. Review the Krawczyk inclusion and bordered-Hessian inertia reasoning.
6. Review the literature comparison before accepting any novelty wording.

Running the same scripts at two precisions is useful redundancy, but is not an
independent implementation audit.

The standalone algebra audits are intentionally small second code paths for
the displayed two- and six-amplitude Fourier fields. They are not independent
reviews or replacements for checking the full KKT and interval arguments.

## Deliberate exclusions

- No 64-coordinate global optimizer is claimed.
- No conclusion is drawn outside 0<|k|^2<=4.
- The finite stability sweep is not a proof of parameter-uniform local
  maximality.
- The trajectory/depletion project is separate and is not evidence for the
  theorems in this paper.
