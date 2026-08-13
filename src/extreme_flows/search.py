"""Differentiable trajectory objectives for extreme-flow searches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import ceil

import jax
import jax.numpy as jnp

from .spectral import (
    SpectralGrid,
    diagnostics,
    physical_to_state,
    project_velocity,
    rk4_step,
)

Array = jax.Array
OBJECTIVE_VERSION = "persistence-v3-full-window-fixed-amplification-tail-barrier"
DEPLETION_OBJECTIVE_VERSION = "depletion-v1-zeta-softmin-fixed-amplification-tail-barrier"
INSTANTANEOUS_OBJECTIVE_VERSION = "instantaneous-v1-normalized-enstrophy-production"

DIAGNOSTIC_NAMES = (
    "energy",
    "enstrophy",
    "enstrophy_rate",
    "stretching",
    "viscous_dissipation",
    "spectral_velocity_tail_fraction",
    "max_vorticity",
    "palinstrophy",
    "stretching_efficiency",
    "spectral_enstrophy_tail_fraction",
    "spectral_palinstrophy_tail_fraction",
    "depletion_indicator",
    "spectral_broadening_ratio",
    "normalized_viscous_depletion",
    "stretching_to_viscous_ratio",
)


@dataclass(frozen=True)
class SearchConfig:
    """Parameters defining one differentiable search problem."""

    viscosity: float = 0.01
    target_enstrophy: float = 100.0
    dt: float = 0.002
    steps: int = 50
    delta: float = 0.1
    softmin_temperature: float = 1.0e-4
    amplification_weight: float = 2.0e-4
    target_amplification: float = 2.0
    amplification_penalty: float = 1.0e-2
    amplification_temperature: float = 2.0e-2
    tail_weight: float = 1.0e-1
    tail_threshold: float = 1.0e-1
    tail_temperature: float = 1.0e-2
    skip_time: float = 0.0
    initial_max_mode: float | None = 3.0


def spectral_enstrophy(u_hat: Array, grid: SpectralGrid) -> Array:
    """Compute normalized enstrophy directly from Fourier coefficients."""

    omega_hat = jnp.cross(1j * grid.kvec, u_hat)
    return 0.5 * jnp.sum(jnp.abs(omega_hat) ** 2) / (grid.n**6)


def constrained_state(theta: Array, grid: SpectralGrid, config: SearchConfig) -> Array:
    """Map unconstrained real parameters to fixed-enstrophy initial data."""

    state = physical_to_state(theta, grid)
    if config.initial_max_mode is not None:
        low_pass = grid.k2 <= config.initial_max_mode**2
        state = project_velocity(state * low_pass[..., None], grid)
    enstrophy = spectral_enstrophy(state, grid)
    scale = jnp.sqrt(config.target_enstrophy / jnp.maximum(enstrophy, 1.0e-30))
    return state * scale


def _diagnostic_vector(
    state: Array, grid: SpectralGrid, viscosity: float
) -> Array:
    d = diagnostics(state, grid, viscosity)
    return jnp.stack(
        (
            d["energy"],
            d["enstrophy"],
            d["enstrophy_rate"],
            d["stretching"],
            d["viscous_dissipation"],
            d["spectral_tail_fraction"],
            d["max_vorticity"],
            d["palinstrophy"],
            d["stretching_efficiency"],
            d["spectral_enstrophy_tail_fraction"],
            d["spectral_palinstrophy_tail_fraction"],
            d["depletion_indicator"],
            d["spectral_broadening_ratio"],
            d["normalized_viscous_depletion"],
            d["stretching_to_viscous_ratio"],
        )
    )


def trajectory_diagnostics(
    state0: Array, grid: SpectralGrid, config: SearchConfig
) -> tuple[Array, Array]:
    """Integrate one trajectory and return final state plus scalar diagnostics.

    Diagnostic columns are energy, enstrophy, enstrophy rate, stretching,
    viscous dissipation, velocity-energy tail fraction, maximum vorticity,
    palinstrophy, stretching efficiency, enstrophy-tail fraction, and
    palinstrophy-tail fraction, and the four mechanism coordinates zeta, rho,
    z, and sigma.  Row zero corresponds to the initial state.
    """

    def advance(state: Array, _: None) -> tuple[Array, Array]:
        next_state = rk4_step(state, grid, config.viscosity, config.dt)
        return next_state, _diagnostic_vector(next_state, grid, config.viscosity)

    final_state, evolved = jax.lax.scan(advance, state0, xs=None, length=config.steps)
    initial = _diagnostic_vector(state0, grid, config.viscosity)[None, :]
    return final_state, jnp.concatenate((initial, evolved), axis=0)


def soft_min(values: Array, temperature: float) -> Array:
    """Stable conservative smooth approximation of min(values)."""

    return -temperature * jax.scipy.special.logsumexp(-values / temperature)


def soft_max(values: Array, temperature: float) -> Array:
    """Stable conservative smooth approximation of max(values)."""

    return temperature * jax.scipy.special.logsumexp(values / temperature)


def instantaneous_objective(theta: Array, grid: SpectralGrid, config: SearchConfig) -> Array:
    """Benchmark objective: normalized instantaneous enstrophy production."""

    state = constrained_state(theta, grid, config)
    d = diagnostics(state, grid, config.viscosity)
    return d["enstrophy_rate"] / config.target_enstrophy**3


def persistence_objective(theta: Array, grid: SpectralGrid, config: SearchConfig) -> Array:
    """Reward the weakest superquadratic-growth margin along a trajectory.

    The primary term is a soft minimum of R/E^(2+delta).  A small endpoint
    amplification reward prevents the optimizer from obtaining a deceptively
    good ratio only by shrinking E, and a spectral-tail penalty rejects states
    that exploit the truncation boundary.
    """

    state0 = constrained_state(theta, grid, config)
    _, history = trajectory_diagnostics(state0, grid, config)
    enstrophy = history[:, 1]
    rate = history[:, 2]
    palinstrophy_tail = history[:, 10]
    start = min(ceil(config.skip_time / config.dt), config.steps)
    margin = rate[start:] / jnp.maximum(enstrophy[start:], 1.0e-30) ** (
        2.0 + config.delta
    )
    bottleneck = soft_min(margin, config.softmin_temperature)
    amplification = jnp.log(jnp.maximum(enstrophy[-1], 1.0e-30) / enstrophy[0])
    shortfall, tail_excess = _constraint_violations(
        amplification, palinstrophy_tail[start:], config
    )
    return (
        bottleneck
        + config.amplification_weight * amplification
        - config.amplification_penalty * shortfall * shortfall
        - config.tail_weight * tail_excess * tail_excess
    )


def _constraint_violations(
    log_amplification: Array, palinstrophy_tail: Array, config: SearchConfig
) -> tuple[Array, Array]:
    """Smooth endpoint-amplification and resolution-barrier violations."""

    target_log_amplification = jnp.log(config.target_amplification)
    shortfall = config.amplification_temperature * jax.nn.softplus(
        (target_log_amplification - log_amplification)
        / config.amplification_temperature
    )
    tail_maximum = jnp.max(palinstrophy_tail)
    tail_excess = config.tail_temperature * jax.nn.softplus(
        (tail_maximum - config.tail_threshold) / config.tail_temperature
    )
    return shortfall, tail_excess


def depletion_objective(theta: Array, grid: SpectralGrid, config: SearchConfig) -> Array:
    """Reward persistence of the borderline regularity variable zeta.

    Here zeta = chi_+ E^(1/4).  A finite a-priori upper bound on zeta along
    every trajectory would imply global regularity, so this objective directly
    searches for the geometric obstruction rather than allowing palinstrophy
    growth alone to carry the enstrophy-production score.
    """

    state0 = constrained_state(theta, grid, config)
    _, history = trajectory_diagnostics(state0, grid, config)
    enstrophy = history[:, 1]
    palinstrophy_tail = history[:, 10]
    depletion_indicator = history[:, 11]
    start = min(ceil(config.skip_time / config.dt), config.steps)
    bottleneck = soft_min(
        depletion_indicator[start:], config.softmin_temperature
    )
    amplification = jnp.log(jnp.maximum(enstrophy[-1], 1.0e-30) / enstrophy[0])
    shortfall, tail_excess = _constraint_violations(
        amplification, palinstrophy_tail[start:], config
    )
    return (
        bottleneck
        + config.amplification_weight * amplification
        - config.amplification_penalty * shortfall * shortfall
        - config.tail_weight * tail_excess * tail_excess
    )


def make_value_and_grad(
    grid: SpectralGrid, config: SearchConfig, *, persistent: bool
):
    """Build a JIT-compiled value/gradient function with constants captured."""

    objective = persistence_objective if persistent else instantaneous_objective
    return jax.jit(jax.value_and_grad(lambda theta: objective(theta, grid, config)))


def adam_ascent(
    theta0: Array,
    value_and_grad,
    *,
    iterations: int,
    learning_rate: float,
    callback=None,
) -> tuple[Array, list[float]]:
    """Small dependency-free Adam maximizer for dense research prototypes."""

    theta = theta0 / jnp.sqrt(jnp.mean(theta0 * theta0))
    first = jnp.zeros_like(theta)
    second = jnp.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    history: list[float] = []

    for iteration in range(1, iterations + 1):
        value, gradient = value_and_grad(theta)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * (gradient * gradient)
        first_hat = first / (1.0 - beta1**iteration)
        second_hat = second / (1.0 - beta2**iteration)
        theta = theta + learning_rate * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
        theta = theta / jnp.sqrt(jnp.mean(theta * theta))
        scalar = float(value)
        history.append(scalar)
        if callback is not None:
            callback(iteration, scalar, theta)

    return theta, history
