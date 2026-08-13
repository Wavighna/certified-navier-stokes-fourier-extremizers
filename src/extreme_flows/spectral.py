"""Dealiased Fourier tools for 3D periodic incompressible flow.

The implementation favors transparent identities over peak performance.  All
spatial averages are normalized by the torus volume, so energy and enstrophy are
reported per unit volume.  Fourier transforms use JAX's default convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

Array = jax.Array
_SPACE_AXES = (0, 1, 2)


@dataclass(frozen=True)
class SpectralGrid:
    """Fourier geometry for a cubic periodic grid."""

    n: int
    length: float = 2.0 * pi
    kvec: Array = field(init=False, repr=False)
    k2: Array = field(init=False, repr=False)
    inv_k2: Array = field(init=False, repr=False)
    dealias: Array = field(init=False, repr=False)
    tail: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 6 or self.n % 2:
            raise ValueError("n must be an even integer >= 6")
        dx = self.length / self.n
        k = 2.0 * pi * jnp.fft.fftfreq(self.n, d=dx)
        kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
        kvec = jnp.stack((kx, ky, kz), axis=-1)
        k2 = jnp.sum(kvec * kvec, axis=-1)
        inv_k2 = jnp.where(k2 > 0.0, 1.0 / k2, 0.0)

        # The mask is expressed in integer FFT mode numbers and implements the
        # standard 2/3 rule for quadratic nonlinearities.
        modes = jnp.fft.fftfreq(self.n) * self.n
        mx, my, mz = jnp.meshgrid(modes, modes, modes, indexing="ij")
        cutoff = self.n / 3.0
        dealias = (
            (jnp.abs(mx) < cutoff)
            & (jnp.abs(my) < cutoff)
            & (jnp.abs(mz) < cutoff)
        )
        shell = jnp.maximum(jnp.maximum(jnp.abs(mx), jnp.abs(my)), jnp.abs(mz))
        tail = dealias & (shell >= max(1.0, 0.7 * cutoff))

        object.__setattr__(self, "kvec", kvec)
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "inv_k2", inv_k2)
        object.__setattr__(self, "dealias", dealias)
        object.__setattr__(self, "tail", tail)


def fft_velocity(u: Array) -> Array:
    """Transform a real vector field to full complex Fourier coefficients."""

    return jnp.fft.fftn(u, axes=_SPACE_AXES)


def ifft_velocity(u_hat: Array) -> Array:
    """Transform Fourier coefficients to a real vector field."""

    return jnp.fft.ifftn(u_hat, axes=_SPACE_AXES).real


def project_velocity(u_hat: Array, grid: SpectralGrid) -> Array:
    """Apply the Leray projector, zero-mean constraint, and 2/3 mask."""

    k_dot_u = jnp.sum(grid.kvec * u_hat, axis=-1)
    projected = u_hat - grid.kvec * (
        k_dot_u * grid.inv_k2
    )[..., jnp.newaxis]
    projected = projected * grid.dealias[..., jnp.newaxis]
    return projected.at[0, 0, 0, :].set(0.0)


def physical_to_state(u: Array, grid: SpectralGrid) -> Array:
    """Convert an arbitrary physical field into an admissible spectral state."""

    return project_velocity(fft_velocity(u), grid)


def curl_hat(u_hat: Array, grid: SpectralGrid) -> Array:
    """Return Fourier coefficients of curl(u)."""

    return jnp.cross(1j * grid.kvec, u_hat)


def gradient_physical(u_hat: Array, grid: SpectralGrid) -> Array:
    """Return grad(u) in physical space with component order [..., i, j]."""

    grad_hat = 1j * u_hat[..., :, jnp.newaxis] * grid.kvec[..., jnp.newaxis, :]
    return jnp.fft.ifftn(grad_hat, axes=_SPACE_AXES).real


def nonlinear_hat(u_hat: Array, grid: SpectralGrid) -> Array:
    """Return the projected, dealiased Fourier transform of (u dot grad)u."""

    u = ifft_velocity(u_hat)
    grad_u = gradient_physical(u_hat, grid)
    advection = jnp.einsum("...j,...ij->...i", u, grad_u)
    return project_velocity(fft_velocity(advection), grid)


def rhs(u_hat: Array, grid: SpectralGrid, viscosity: float) -> Array:
    """Spectral right-hand side for Euler (viscosity=0) or Navier-Stokes."""

    return -nonlinear_hat(u_hat, grid) - viscosity * grid.k2[..., None] * u_hat


def rk4_step(
    u_hat: Array, grid: SpectralGrid, viscosity: float, dt: float
) -> Array:
    """One projected classical fourth-order Runge-Kutta step."""

    f = lambda state: rhs(state, grid, viscosity)
    k1 = f(u_hat)
    k2 = f(project_velocity(u_hat + 0.5 * dt * k1, grid))
    k3 = f(project_velocity(u_hat + 0.5 * dt * k2, grid))
    k4 = f(project_velocity(u_hat + dt * k3, grid))
    updated = u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return project_velocity(updated, grid)


def diagnostics(u_hat: Array, grid: SpectralGrid, viscosity: float) -> dict[str, Array]:
    """Evaluate balance laws and blow-up-relevant quantities."""

    u = ifft_velocity(u_hat)
    grad_u = gradient_physical(u_hat, grid)
    omega_hat = curl_hat(u_hat, grid)
    omega = ifft_velocity(omega_hat)
    grad_omega_hat = (
        1j
        * omega_hat[..., :, jnp.newaxis]
        * grid.kvec[..., jnp.newaxis, :]
    )
    grad_omega = jnp.fft.ifftn(grad_omega_hat, axes=_SPACE_AXES).real
    strain = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))

    energy = 0.5 * jnp.mean(jnp.sum(u * u, axis=-1))
    enstrophy = 0.5 * jnp.mean(jnp.sum(omega * omega, axis=-1))
    palinstrophy = 0.5 * jnp.mean(jnp.sum(grad_omega * grad_omega, axis=(-1, -2)))
    stretching = jnp.mean(jnp.einsum("...i,...ij,...j->...", omega, strain, omega))
    viscous = 2.0 * viscosity * palinstrophy
    enstrophy_rate = stretching - viscous
    stretching_efficiency = stretching / jnp.maximum(
        enstrophy**0.75 * palinstrophy**0.75,
        jnp.finfo(jnp.float64).tiny,
    )
    positive_stretching_efficiency = jnp.maximum(stretching, 0.0) / jnp.maximum(
        enstrophy**0.75 * palinstrophy**0.75,
        jnp.finfo(jnp.float64).tiny,
    )
    depletion_indicator = positive_stretching_efficiency * enstrophy**0.25
    spectral_broadening_ratio = energy * palinstrophy / jnp.maximum(
        enstrophy * enstrophy, jnp.finfo(jnp.float64).tiny
    )
    normalized_viscous_depletion = viscous / jnp.maximum(
        enstrophy * enstrophy, jnp.finfo(jnp.float64).tiny
    )
    stretching_to_viscous_ratio = stretching / jnp.maximum(
        viscous, jnp.finfo(jnp.float64).tiny
    )
    helicity = jnp.mean(jnp.sum(u * omega, axis=-1))
    divergence = jnp.trace(grad_u, axis1=-2, axis2=-1)

    mode_energy = jnp.sum(jnp.abs(u_hat) ** 2, axis=-1)
    mode_enstrophy = jnp.sum(jnp.abs(omega_hat) ** 2, axis=-1)
    mode_palinstrophy = grid.k2 * mode_enstrophy
    tail_fraction = jnp.sum(jnp.where(grid.tail, mode_energy, 0.0)) / jnp.maximum(
        jnp.sum(mode_energy), jnp.finfo(jnp.float64).tiny
    )
    enstrophy_tail_fraction = jnp.sum(
        jnp.where(grid.tail, mode_enstrophy, 0.0)
    ) / jnp.maximum(jnp.sum(mode_enstrophy), jnp.finfo(jnp.float64).tiny)
    palinstrophy_tail_fraction = jnp.sum(
        jnp.where(grid.tail, mode_palinstrophy, 0.0)
    ) / jnp.maximum(jnp.sum(mode_palinstrophy), jnp.finfo(jnp.float64).tiny)
    max_vorticity = jnp.max(jnp.sqrt(jnp.sum(omega * omega, axis=-1)))

    return {
        "energy": energy,
        "enstrophy": enstrophy,
        "palinstrophy": palinstrophy,
        "stretching": stretching,
        "stretching_efficiency": stretching_efficiency,
        "positive_stretching_efficiency": positive_stretching_efficiency,
        "depletion_indicator": depletion_indicator,
        "spectral_broadening_ratio": spectral_broadening_ratio,
        "normalized_viscous_depletion": normalized_viscous_depletion,
        "stretching_to_viscous_ratio": stretching_to_viscous_ratio,
        "viscous_dissipation": viscous,
        "enstrophy_rate": enstrophy_rate,
        "helicity": helicity,
        "divergence_l2": jnp.sqrt(jnp.mean(divergence * divergence)),
        "spectral_tail_fraction": tail_fraction,
        "spectral_enstrophy_tail_fraction": enstrophy_tail_fraction,
        "spectral_palinstrophy_tail_fraction": palinstrophy_tail_fraction,
        "max_vorticity": max_vorticity,
    }


def strain_alignment_diagnostics(
    u_hat: Array, grid: SpectralGrid
) -> dict[str, Array]:
    """Decompose vortex stretching in the pointwise strain eigenframe.

    Eigenvalues are ordered from most compressive to most extensive.  This is
    intentionally kept out of :func:`diagnostics`, because differentiating an
    eigendecomposition at repeated eigenvalues is fragile and the persistence
    optimizer only needs the cheaper scalar diagnostics.
    """

    grad_u = gradient_physical(u_hat, grid)
    strain = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
    omega = ifft_velocity(curl_hat(u_hat, grid))
    eigenvalues, eigenvectors = jnp.linalg.eigh(strain)
    projections = jnp.einsum("...ji,...j->...i", eigenvectors, omega)
    squared_projections = projections * projections
    contributions = jnp.mean(
        eigenvalues * squared_projections, axis=_SPACE_AXES
    )
    alignment_fractions = jnp.mean(
        squared_projections, axis=_SPACE_AXES
    ) / jnp.maximum(
        jnp.mean(jnp.sum(omega * omega, axis=-1)),
        jnp.finfo(jnp.float64).tiny,
    )
    strain_norm = jnp.sqrt(jnp.sum(strain * strain, axis=(-1, -2)))
    strain_norm_squared_mean = jnp.mean(strain_norm * strain_norm)
    strain_norm_cubed_mean = jnp.mean(strain_norm**3)
    determinant_mean = jnp.mean(jnp.linalg.det(strain))
    stretching = jnp.mean(
        jnp.einsum("...i,...ij,...j->...", omega, strain, omega)
    )
    enstrophy = 0.5 * jnp.mean(jnp.sum(omega * omega, axis=-1))
    normalized_strain_concentration = strain_norm_cubed_mean / jnp.maximum(
        enstrophy**1.5, jnp.finfo(jnp.float64).tiny
    )
    strain_topology_factor = (
        -3.0 * jnp.sqrt(6.0) * determinant_mean
        / jnp.maximum(strain_norm_cubed_mean, jnp.finfo(jnp.float64).tiny)
    )
    betchov_scale = jnp.maximum(
        jnp.abs(stretching) + 4.0 * jnp.abs(determinant_mean),
        jnp.finfo(jnp.float64).tiny,
    )

    return {
        "strain_eigenvalue_mean": jnp.mean(eigenvalues, axis=_SPACE_AXES),
        "strain_eigenvalue_rms": jnp.sqrt(
            jnp.mean(eigenvalues * eigenvalues, axis=_SPACE_AXES)
        ),
        "stretching_contributions": contributions,
        "vorticity_alignment_fractions": alignment_fractions,
        "max_extensive_strain": jnp.max(eigenvalues[..., 2]),
        "strain_norm_squared_mean": strain_norm_squared_mean,
        "normalized_strain_concentration": normalized_strain_concentration,
        "strain_topology_factor": strain_topology_factor,
        "normalized_stretching": stretching
        / jnp.maximum(enstrophy**1.5, jnp.finfo(jnp.float64).tiny),
        "betchov_relative_residual": jnp.abs(stretching + 4.0 * determinant_mean)
        / betchov_scale,
    }


def coordinate_mesh(grid: SpectralGrid) -> tuple[Array, Array, Array]:
    """Return an endpoint-free periodic coordinate mesh."""

    x = jnp.arange(grid.n, dtype=jnp.float64) * (grid.length / grid.n)
    return jnp.meshgrid(x, x, x, indexing="ij")


def taylor_green(grid: SpectralGrid) -> Array:
    """Classical Taylor-Green velocity field as an admissible spectral state."""

    x, y, z = coordinate_mesh(grid)
    u = jnp.stack(
        (
            jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
            -jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
            jnp.zeros_like(x),
        ),
        axis=-1,
    )
    return physical_to_state(u, grid)


def abc_flow(
    grid: SpectralGrid, a: float = 1.0, b: float = 1.0, c: float = 1.0
) -> Array:
    """Arnold-Beltrami-Childress flow satisfying curl(u)=u."""

    x, y, z = coordinate_mesh(grid)
    u = jnp.stack(
        (
            a * jnp.sin(z) + c * jnp.cos(y),
            b * jnp.sin(x) + a * jnp.cos(z),
            c * jnp.sin(y) + b * jnp.cos(x),
        ),
        axis=-1,
    )
    return physical_to_state(u, grid)


def resample_state(
    u_hat: Array, old_grid: SpectralGrid, new_grid: SpectralGrid
) -> Array:
    """Spectrally zero-pad a state onto a finer even grid.

    JAX's forward FFT is unnormalized, so coefficients are multiplied by the
    cube of the grid-size ratio.  The routine is intended for already
    dealiased states without Nyquist content.
    """

    if new_grid.n < old_grid.n:
        raise ValueError("resample_state currently supports refinement only")
    difference = new_grid.n - old_grid.n
    if difference % 2:
        raise ValueError("old and new grid sizes must have the same parity")
    if difference == 0:
        return project_velocity(u_hat, new_grid)
    padding = difference // 2
    shifted = jnp.fft.fftshift(u_hat, axes=_SPACE_AXES)
    padded = jnp.pad(
        shifted,
        ((padding, padding), (padding, padding), (padding, padding), (0, 0)),
    )
    refined = jnp.fft.ifftshift(padded, axes=_SPACE_AXES)
    refined = refined * (new_grid.n / old_grid.n) ** 3
    return project_velocity(refined, new_grid)


def oversampled_max_vorticity(
    u_hat: Array, grid: SpectralGrid, factor: int = 3
) -> Array:
    """Evaluate the band-limited vorticity maximum on a padded grid.

    A native collocation grid can substantially under-sample a sharp but still
    band-limited peak.  Padding changes only where the same trigonometric
    polynomial is evaluated; it does not add physical Fourier content.
    """

    if factor < 1 or int(factor) != factor:
        raise ValueError("factor must be a positive integer")
    if factor == 1:
        refined = u_hat
        fine_grid = grid
    else:
        fine_grid = SpectralGrid(grid.n * factor, length=grid.length)
        refined = resample_state(u_hat, grid, fine_grid)
    omega = ifft_velocity(curl_hat(refined, fine_grid))
    return jnp.max(jnp.sqrt(jnp.sum(omega * omega, axis=-1)))
