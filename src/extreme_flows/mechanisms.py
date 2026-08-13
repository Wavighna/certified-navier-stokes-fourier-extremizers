"""Exact scalar mechanism identities for periodic incompressible flow.

The routines in this module keep two independent factorizations of normalized
vortex stretching side by side.  The first is Betchov's classical strain
factorization.  The second uses the strain/Laplacian orthogonality and
cancellation identities discussed by Miller.  Their agreement is a useful
numerical invariant; it is not presented here as a new cancellation theorem.

All integrals are volume-normalized averages on the periodic cube, matching
the conventions in :mod:`extreme_flows.spectral`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .spectral import (
    SpectralGrid,
    curl_hat,
    ifft_velocity,
)

Array = jax.Array
_SPACE_AXES = (0, 1, 2)
_TINY = jnp.finfo(jnp.float64).tiny


def _safe_ratio(numerator: Array, denominator: Array) -> Array:
    """Divide scalar arrays, returning NaN when the denominator is not positive."""

    return jnp.where(denominator > 0.0, numerator / denominator, jnp.nan)


def _comparison_residual(lhs: Array, rhs: Array) -> Array:
    """Symmetric relative residual for two quantities that should agree."""

    scale = jnp.maximum(jnp.abs(lhs) + jnp.abs(rhs), _TINY)
    return jnp.abs(lhs - rhs) / scale


def _strain_laplacian_fields(
    u_hat: Array, grid: SpectralGrid
) -> tuple[Array, Array]:
    """Return ``S`` and ``-Delta S`` in physical space."""

    grad_u_hat = (
        1j
        * u_hat[..., :, None]
        * grid.kvec[..., None, :]
    )
    strain_hat = 0.5 * (
        grad_u_hat + jnp.swapaxes(grad_u_hat, -1, -2)
    )
    strain = jnp.fft.ifftn(strain_hat, axes=_SPACE_AXES).real
    negative_laplacian_strain = jnp.fft.ifftn(
        grid.k2[..., None, None] * strain_hat,
        axes=_SPACE_AXES,
    ).real
    return strain, negative_laplacian_strain


def miller_residual_strain(u_hat: Array, grid: SpectralGrid) -> Array:
    """Return ``R = S - (<S,-Delta S>/||Delta S||_2^2)(-Delta S)``.

    This field-level interface makes the orthogonal residual available for
    downstream analysis.  :func:`mechanism_diagnostics` additionally exposes
    its norm, stretching contribution, and normalized alignment factor.
    """

    strain, negative_laplacian_strain = _strain_laplacian_fields(u_hat, grid)
    palinstrophy = jnp.mean(
        jnp.sum(strain * negative_laplacian_strain, axis=(-1, -2))
    )
    q_laplacian_strain = jnp.mean(
        jnp.sum(
            negative_laplacian_strain * negative_laplacian_strain,
            axis=(-1, -2),
        )
    )
    return strain - _safe_ratio(
        palinstrophy, q_laplacian_strain
    ) * negative_laplacian_strain


def mechanism_diagnostics(
    u_hat: Array, grid: SpectralGrid
) -> dict[str, Array]:
    """Evaluate the Betchov--Miller mechanism coordinates and identities.

    The canonical scalar names follow the notation used in the research notes:

    ``K``
        Kinetic energy, ``0.5 <|u|^2>``.
    ``E``
        Enstrophy, ``0.5 <|omega|^2> = <|S|^2>``.
    ``P``
        Palinstrophy, ``0.5 <|grad omega|^2> = <S,-Delta S>``.
    ``Q``
        ``<|Delta S|^2>``.
    ``A``
        Signed vortex stretching, ``<omega . S omega>``.

    ``chi_betchov`` and ``chi_dual`` reconstruct the signed normalized
    stretching ``chi`` independently.  Degenerate factors are reported as
    NaN rather than regularized by a small denominator, so callers cannot
    mistake an undefined logarithmic budget for a valid one.
    """

    u = ifft_velocity(u_hat)
    strain, negative_laplacian_strain = _strain_laplacian_fields(u_hat, grid)

    omega_hat = curl_hat(u_hat, grid)
    omega = ifft_velocity(omega_hat)
    omega_squared = jnp.sum(omega * omega, axis=-1)
    omega_tensor = omega[..., :, None] * omega[..., None, :]

    grad_omega_hat = (
        1j
        * omega_hat[..., :, None]
        * grid.kvec[..., None, :]
    )
    grad_omega = jnp.fft.ifftn(grad_omega_hat, axes=_SPACE_AXES).real

    kinetic_energy = 0.5 * jnp.mean(jnp.sum(u * u, axis=-1))
    enstrophy = 0.5 * jnp.mean(omega_squared)
    palinstrophy = 0.5 * jnp.mean(
        jnp.sum(grad_omega * grad_omega, axis=(-1, -2))
    )
    q_laplacian_strain = jnp.mean(
        jnp.sum(
            negative_laplacian_strain * negative_laplacian_strain,
            axis=(-1, -2),
        )
    )
    stretching = jnp.mean(
        jnp.einsum("...i,...ij,...j->...", omega, strain, omega)
    )

    strain_squared_mean = jnp.mean(
        jnp.sum(strain * strain, axis=(-1, -2))
    )
    strain_palinstrophy = jnp.mean(
        jnp.sum(strain * negative_laplacian_strain, axis=(-1, -2))
    )
    strain_norm = jnp.sqrt(jnp.sum(strain * strain, axis=(-1, -2)))
    strain_cubed_mean = jnp.mean(strain_norm**3)
    strain_determinant_mean = jnp.mean(jnp.linalg.det(strain))

    normalized_strain_concentration = _safe_ratio(
        strain_cubed_mean, enstrophy**1.5
    )
    strain_topology_factor = _safe_ratio(
        -3.0 * jnp.sqrt(6.0) * strain_determinant_mean,
        strain_cubed_mean,
    )
    spectral_broadening_ratio = _safe_ratio(
        kinetic_energy * palinstrophy, enstrophy * enstrophy
    )

    defect_unclipped = 1.0 - _safe_ratio(
        palinstrophy * palinstrophy,
        enstrophy * q_laplacian_strain,
    )
    # The exact defect lies in [0,1].  Clipping only protects square roots from
    # a last-bit negative value in a degenerate flow; the raw value remains
    # available for auditing.
    defect = jnp.clip(defect_unclipped, 0.0, 1.0)
    residual_strain = strain - _safe_ratio(
        palinstrophy, q_laplacian_strain
    ) * negative_laplacian_strain
    residual_squared_mean = jnp.mean(
        jnp.sum(residual_strain * residual_strain, axis=(-1, -2))
    )

    residual_stretching = jnp.mean(
        jnp.sum(residual_strain * omega_tensor, axis=(-1, -2))
    )
    omega_l4_squared = jnp.sqrt(jnp.mean(omega_squared * omega_squared))
    topology_denominator = (
        jnp.sqrt(enstrophy * defect) * omega_l4_squared
    )
    topology_efficiency = _safe_ratio(
        residual_stretching, topology_denominator
    )
    l4_saturation = _safe_ratio(
        omega_l4_squared,
        2.0 * enstrophy**0.25 * palinstrophy**0.75,
    )

    chi = _safe_ratio(
        stretching, enstrophy**0.75 * palinstrophy**0.75
    )
    chi_betchov = (
        4.0
        / (3.0 * jnp.sqrt(6.0))
        * strain_topology_factor
        * normalized_strain_concentration
        * _safe_ratio(enstrophy, palinstrophy) ** 0.75
    )
    chi_dual = (
        2.0
        * topology_efficiency
        * l4_saturation
        * jnp.sqrt(defect)
    )

    miller_cancellation = jnp.mean(
        jnp.sum(
            negative_laplacian_strain * omega_tensor,
            axis=(-1, -2),
        )
    )
    miller_orthogonality = jnp.mean(
        jnp.sum(
            residual_strain * negative_laplacian_strain,
            axis=(-1, -2),
        )
    )
    cancellation_scale = jnp.sqrt(q_laplacian_strain) * omega_l4_squared
    orthogonality_scale = (
        jnp.sqrt(residual_squared_mean * q_laplacian_strain)
    )

    return {
        # Canonical coordinates and their descriptive aliases.
        "K": kinetic_energy,
        "E": enstrophy,
        "P": palinstrophy,
        "Q": q_laplacian_strain,
        "A": stretching,
        "Theta": strain_topology_factor,
        "I3": normalized_strain_concentration,
        "rho": spectral_broadening_ratio,
        "d": defect,
        "H": topology_efficiency,
        "J4": l4_saturation,
        "energy": kinetic_energy,
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "laplacian_strain_squared_mean": q_laplacian_strain,
        "stretching": stretching,
        "strain_topology_factor": strain_topology_factor,
        "normalized_strain_concentration": normalized_strain_concentration,
        "spectral_broadening_ratio": spectral_broadening_ratio,
        "orthogonality_defect": defect,
        "orthogonality_defect_unclipped": defect_unclipped,
        "residual_topology_efficiency": topology_efficiency,
        "vorticity_l4_saturation": l4_saturation,
        "omega_l4_squared": omega_l4_squared,
        "residual_stretching": residual_stretching,
        "chi": chi,
        "chi_positive": jnp.maximum(chi, 0.0),
        "chi_betchov": chi_betchov,
        "chi_dual": chi_dual,
        # Raw identity terms.
        "strain_squared_mean": strain_squared_mean,
        "strain_palinstrophy": strain_palinstrophy,
        "strain_determinant_mean": strain_determinant_mean,
        "residual_strain_squared_mean": residual_squared_mean,
        "residual_laplacian_coefficient": _safe_ratio(
            palinstrophy, q_laplacian_strain
        ),
        "miller_cancellation": miller_cancellation,
        "miller_orthogonality": miller_orthogonality,
        # Dimensionless audit residuals.
        "strain_enstrophy_relative_residual": _comparison_residual(
            strain_squared_mean, enstrophy
        ),
        "strain_palinstrophy_relative_residual": _comparison_residual(
            strain_palinstrophy, palinstrophy
        ),
        "betchov_relative_residual": _comparison_residual(
            stretching, -4.0 * strain_determinant_mean
        ),
        "miller_cancellation_relative_residual": _safe_ratio(
            jnp.abs(miller_cancellation), cancellation_scale
        ),
        "miller_orthogonality_relative_residual": _safe_ratio(
            jnp.abs(miller_orthogonality), orthogonality_scale
        ),
        "residual_norm_relative_residual": _comparison_residual(
            residual_squared_mean, enstrophy * defect
        ),
        "betchov_factorization_relative_residual": _comparison_residual(
            chi, chi_betchov
        ),
        "dual_factorization_relative_residual": _comparison_residual(
            chi, chi_dual
        ),
    }


IDENTITY_RESIDUAL_NAMES = (
    "strain_enstrophy_relative_residual",
    "strain_palinstrophy_relative_residual",
    "betchov_relative_residual",
    "miller_cancellation_relative_residual",
    "miller_orthogonality_relative_residual",
    "residual_norm_relative_residual",
    "betchov_factorization_relative_residual",
    "dual_factorization_relative_residual",
)


def scalarize_mechanisms(values: Mapping[str, Any]) -> dict[str, float]:
    """Convert a scalar JAX/NumPy mechanism mapping to ordinary floats."""

    return {name: float(value) for name, value in values.items()}


def _positive_finite(value: float, tolerance: float) -> bool:
    return bool(jnp.isfinite(value)) and value > tolerance


def endpoint_mechanism_ledger(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    samples: Iterable[Mapping[str, Any]] | None = None,
    positivity_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Build the two exact endpoint exponent ledgers with sign guards.

    ``samples`` may contain intermediate mechanism snapshots.  When supplied,
    every sample participates in the positivity guard for ``A``, ``Theta``,
    and ``H``.  Without it, validity certifies only the two endpoints.

    If any logarithm would be undefined, ``valid`` is false and all exponent
    and beta entries are ``None``.  Signed endpoint factors and identity
    residuals are still returned for diagnosis.
    """

    first = scalarize_mechanisms(initial)
    last = scalarize_mechanisms(final)
    path = [first]
    if samples is not None:
        path.extend(scalarize_mechanisms(sample) for sample in samples)
    path.append(last)

    reasons: list[str] = []
    for key in ("E", "P", "K", "I3", "rho", "d", "J4"):
        if not _positive_finite(first[key], positivity_tolerance):
            reasons.append(f"initial {key} is not positive")
        if not _positive_finite(last[key], positivity_tolerance):
            reasons.append(f"final {key} is not positive")
    for index, snapshot in enumerate(path):
        for key in ("A", "Theta", "H", "chi"):
            if not _positive_finite(snapshot[key], positivity_tolerance):
                reasons.append(f"sample {index} {key} is not positive")

    if _positive_finite(first.get("E", float("nan")), positivity_tolerance) and _positive_finite(
        last.get("E", float("nan")), positivity_tolerance
    ):
        log_amplification = float(jnp.log(last["E"] / first["E"]))
        if (
            not bool(jnp.isfinite(log_amplification))
            or abs(log_amplification) <= positivity_tolerance
        ):
            reasons.append("enstrophy amplification has zero or invalid logarithm")
    else:
        log_amplification = float("nan")

    signed_factors = {
        key: {"initial": first[key], "final": last[key]}
        for key in (
            "K",
            "E",
            "P",
            "Q",
            "A",
            "Theta",
            "I3",
            "rho",
            "d",
            "H",
            "J4",
            "chi",
            "chi_betchov",
            "chi_dual",
        )
    }
    identity_residuals = {
        name: {"initial": first[name], "final": last[name]}
        for name in IDENTITY_RESIDUAL_NAMES
    }
    maximum_identity_residual = max(
        value
        for endpoints in identity_residuals.values()
        for value in endpoints.values()
    )

    result: dict[str, Any] = {
        "valid": not reasons,
        "guard_scope": "full supplied sample path" if samples is not None else "endpoints only",
        "invalid_reasons": reasons,
        "enstrophy_amplification": (
            last["E"] / first["E"]
            if _positive_finite(first.get("E", float("nan")), positivity_tolerance)
            else None
        ),
        "log_enstrophy_amplification": (
            log_amplification if bool(jnp.isfinite(log_amplification)) else None
        ),
        "signed_factors": signed_factors,
        "identity_residuals": identity_residuals,
        "maximum_identity_residual": maximum_identity_residual,
        "exponents": None,
        "beta": {
            "direct": None,
            "betchov_ledger": None,
            "dual_ledger": None,
        },
        "ledger_agreement": {
            "betchov_minus_direct": None,
            "dual_minus_direct": None,
            "betchov_minus_dual": None,
        },
    }
    if reasons:
        return result

    exponent_keys = ("chi", "K", "Theta", "I3", "rho", "H", "J4", "d")
    exponents = {
        key: float(jnp.log(last[key] / first[key]) / log_amplification)
        for key in exponent_keys
    }
    beta_direct = -exponents["chi"]
    beta_betchov = (
        0.75
        - exponents["Theta"]
        - exponents["I3"]
        - 0.75 * exponents["K"]
        + 0.75 * exponents["rho"]
    )
    beta_dual = (
        -exponents["H"]
        - exponents["J4"]
        - 0.5 * exponents["d"]
    )
    result["exponents"] = exponents
    result["beta"] = {
        "direct": beta_direct,
        "betchov_ledger": beta_betchov,
        "dual_ledger": beta_dual,
    }
    result["ledger_agreement"] = {
        "betchov_minus_direct": beta_betchov - beta_direct,
        "dual_minus_direct": beta_dual - beta_direct,
        "betchov_minus_dual": beta_betchov - beta_dual,
    }
    return result
