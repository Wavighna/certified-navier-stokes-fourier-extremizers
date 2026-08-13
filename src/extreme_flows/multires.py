"""Multi-resolution critical-depletion objectives on the low-mode sphere.

This module deliberately separates the 64-dimensional initial-data control
from the state resolution.  The same point on ``S^63`` is lifted onto every
grid, integrated independently, and scored by the worst critical-exponent
margin over grids and prescribed enstrophy amplification levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .low_modes import CONTROL_DIMENSION, lift_control, normalize_coordinates
from .spectral import SpectralGrid, diagnostics, ifft_velocity, rk4_step

Array = jax.Array
MULTIRES_OBJECTIVE_VERSION = "critical-beta-v3-initial-q-constrained"
CERTIFIED_P3_INITIAL_Q_FLOOR = 0.013828980388103783

MULTIRES_DIAGNOSTIC_NAMES = (
    "energy",
    "enstrophy",
    "enstrophy_rate",
    "stretching",
    "palinstrophy",
    "depletion_indicator",
    "spectral_velocity_tail_fraction",
    "spectral_enstrophy_tail_fraction",
    "spectral_palinstrophy_tail_fraction",
    "cfl",
)

ENERGY = 0
ENSTROPHY = 1
ENSTROPHY_RATE = 2
STRETCHING = 3
PALINSTROPHY = 4
ZETA = 5
VELOCITY_TAIL = 6
ENSTROPHY_TAIL = 7
PALINSTROPHY_TAIL = 8
CFL = 9

PROTECTED_MARGIN_NAMES = (
    "initial_q",
    "enstrophy_rate",
    "enstrophy_monotonicity",
    "velocity_tail",
    "enstrophy_tail",
    "palinstrophy_tail",
    "cfl",
    "endpoint_tolerance",
)


@dataclass(frozen=True)
class MultiResolutionConfig:
    """Complete numerical and penalty specification for a joint search."""

    grid_sizes: tuple[int, ...] = (16, 20)
    viscosity: float = 0.01
    target_enstrophy: float = 100.0
    dt: float = 0.001
    steps: int = 208
    block_size: int = 8
    amplification_levels: tuple[float, ...] = (
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
    )
    target_amplification: float = 2.0
    velocity_tail_limits: tuple[float, ...] = (0.01, 0.005)
    enstrophy_tail_limits: tuple[float, ...] = (0.05, 0.025)
    palinstrophy_tail_limits: tuple[float, ...] = (0.15, 0.08)
    cfl_limit: float = 0.35
    delta: float = 0.1
    initial_q_floor: float = CERTIFIED_P3_INITIAL_Q_FLOOR
    initial_q_penalty: float = 50.0
    hard_margin_tolerance: float = 1.0e-12
    endpoint_tolerance: float = 2.0e-3
    softmin_temperature: float = 2.0e-3
    constraint_temperature: float = 1.0e-2
    # Calibrated on the full N16/P3 horizon: 100 drives the endpoint residual
    # toward zero without collapsing the safeguarded q-boundary step.  The
    # previous value 20 let the Adam warm start trade endpoint feasibility for
    # a better gamma score, leaving too much work for the constrained refine.
    endpoint_penalty: float = 100.0
    tail_penalty: float = 2.0
    monotonicity_penalty: float = 2.0
    cfl_penalty: float = 2.0

    def __post_init__(self) -> None:
        grid_count = len(self.grid_sizes)
        if grid_count == 0:
            raise ValueError("at least one state grid is required")
        if any(n < 8 or n % 2 for n in self.grid_sizes):
            raise ValueError("all grid sizes must be even integers >= 8")
        if self.viscosity < 0.0 or self.target_enstrophy <= 0.0:
            raise ValueError("viscosity must be nonnegative and enstrophy positive")
        if self.dt <= 0.0 or self.steps < 1 or self.block_size < 1:
            raise ValueError("dt, steps, and block_size must be positive")
        if any(level <= 1.0 for level in self.amplification_levels):
            raise ValueError("amplification levels must exceed one")
        if tuple(sorted(self.amplification_levels)) != self.amplification_levels:
            raise ValueError("amplification levels must be increasing")
        if self.target_amplification <= 1.0:
            raise ValueError("target amplification must exceed one")
        for name, values in (
            ("velocity_tail_limits", self.velocity_tail_limits),
            ("enstrophy_tail_limits", self.enstrophy_tail_limits),
            ("palinstrophy_tail_limits", self.palinstrophy_tail_limits),
        ):
            if len(values) != grid_count:
                raise ValueError(f"{name} must contain one value per grid")
            if any(value <= 0.0 for value in values):
                raise ValueError(f"{name} values must be positive")
        if self.cfl_limit <= 0.0:
            raise ValueError("cfl_limit must be positive")
        if self.delta < 0.0:
            raise ValueError("delta must be nonnegative")
        if self.initial_q_floor <= 0.0 or self.initial_q_penalty <= 0.0:
            raise ValueError("initial q floor and penalty must be positive")
        if self.hard_margin_tolerance < 0.0:
            raise ValueError("hard_margin_tolerance must be nonnegative")
        if self.endpoint_tolerance <= 0.0:
            raise ValueError("endpoint_tolerance must be positive")
        if self.softmin_temperature <= 0.0 or self.constraint_temperature <= 0.0:
            raise ValueError("smoothing temperatures must be positive")


def _trajectory_row(
    state: Array, grid: SpectralGrid, config: MultiResolutionConfig
) -> Array:
    values = diagnostics(state, grid, config.viscosity)
    velocity = ifft_velocity(state)
    # sum_i |u_i| gives the standard conservative multidimensional CFL bound.
    cfl = (
        config.dt
        * grid.n
        / grid.length
        * jnp.max(jnp.sum(jnp.abs(velocity), axis=-1))
    )
    return jnp.stack(
        (
            values["energy"],
            values["enstrophy"],
            values["enstrophy_rate"],
            values["stretching"],
            values["palinstrophy"],
            values["depletion_indicator"],
            values["spectral_tail_fraction"],
            values["spectral_enstrophy_tail_fraction"],
            values["spectral_palinstrophy_tail_fraction"],
            cfl,
        )
    )


def low_mode_trajectory(
    coordinates: Array,
    grid: SpectralGrid,
    config: MultiResolutionConfig,
) -> tuple[Array, Array]:
    """Integrate one lifted control with checkpointed blocks of RK4 steps."""

    state0 = lift_control(coordinates, grid, config.target_enstrophy)

    def advance(state: Array, _: None) -> tuple[Array, Array]:
        next_state = rk4_step(state, grid, config.viscosity, config.dt)
        return next_state, _trajectory_row(next_state, grid, config)

    def run_steps(state: Array, count: int) -> tuple[Array, Array]:
        return jax.lax.scan(advance, state, xs=None, length=count)

    complete_blocks, remainder = divmod(config.steps, config.block_size)
    pieces: list[Array] = []
    state = state0
    if complete_blocks:
        # Rematerialising an eight-step block avoids retaining every RK4 stage
        # during reverse mode while preserving every diagnostic observation.
        run_block = jax.checkpoint(
            lambda block_state: run_steps(block_state, config.block_size),
            prevent_cse=False,
        )

        def advance_block(block_state: Array, _: None) -> tuple[Array, Array]:
            return run_block(block_state)

        state, blocked_history = jax.lax.scan(
            advance_block, state, xs=None, length=complete_blocks
        )
        pieces.append(
            blocked_history.reshape(
                complete_blocks * config.block_size, len(MULTIRES_DIAGNOSTIC_NAMES)
            )
        )
    if remainder:
        run_remainder = jax.checkpoint(
            lambda remainder_state: run_steps(remainder_state, remainder),
            prevent_cse=False,
        )
        state, remainder_history = run_remainder(state)
        pieces.append(remainder_history)

    evolved = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    initial = _trajectory_row(state0, grid, config)[None, :]
    return state, jnp.concatenate((initial, evolved), axis=0)


def gamma_at_amplifications(
    enstrophy: Array,
    zeta: Array,
    amplification_levels: Sequence[float] | Array,
    *,
    dt: float,
    baseline_zeta_floor: float | Array | None = None,
) -> tuple[Array, Array, Array]:
    """Interpolate ``gamma`` at first amplification crossings in log coordinates.

    The discrete first-crossing index is piecewise constant.  Gradients flow
    through the bracketing values and the log-linear interpolation.  If a
    level has not been reached, the endpoint is returned and ``reached`` is
    false; the endpoint equality penalty then makes that state infeasible.

    ``baseline_zeta_floor`` changes only the subtracted initial log value.  It
    is used as a bounded extension of the objective outside the certified
    initial-bottleneck feasible set.  Crossing values are never clipped to
    this floor, and feasible states pass a zero floor so their score is exact.
    """

    enstrophy = jnp.asarray(enstrophy, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    levels = jnp.asarray(amplification_levels, dtype=jnp.float64)
    tiny = jnp.finfo(jnp.float64).tiny
    log_relative_enstrophy = jnp.log(jnp.maximum(enstrophy, tiny) / enstrophy[0])
    log_zeta = jnp.log(jnp.maximum(zeta, tiny))
    if baseline_zeta_floor is None:
        baseline_log_zeta = log_zeta[0]
    else:
        baseline_log_zeta = jnp.log(
            jnp.maximum(
                jnp.maximum(zeta[0], jnp.asarray(baseline_zeta_floor)), tiny
            )
        )
    sample_count = enstrophy.shape[0]

    def interpolate_one(level: Array) -> tuple[Array, Array, Array]:
        target = jnp.log(level)
        crossings = log_relative_enstrophy >= target
        reached = jnp.any(crossings)
        first = jnp.argmax(crossings.astype(jnp.int32))
        high = jnp.where(reached, jnp.maximum(first, 1), sample_count - 1)
        low = high - 1
        left_x = log_relative_enstrophy[low]
        right_x = log_relative_enstrophy[high]
        denominator = right_x - left_x
        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1.0e-14, denominator, 1.0
        )
        fraction = jnp.clip((target - left_x) / safe_denominator, 0.0, 1.0)
        interpolated_log_zeta = (
            log_zeta[low] + fraction * (log_zeta[high] - log_zeta[low])
        )
        gamma = (interpolated_log_zeta - baseline_log_zeta) / target
        crossing_time = (low.astype(jnp.float64) + fraction) * dt
        return gamma, crossing_time, reached

    return tuple(jax.vmap(interpolate_one)(levels))  # type: ignore[return-value]


def _smooth_violation(margins: Array, temperature: float) -> Array:
    """A smooth dimensionless approximation to ``max(-margin, 0)``."""

    return temperature * jax.nn.softplus(-margins / temperature)


def multiresolution_raw_evaluation(
    coordinates: Array,
    config: MultiResolutionConfig,
    *,
    softmin_temperature: float | None = None,
) -> dict[str, Array]:
    """Return trajectories, gamma arrays, and trust-constr-ready constraints.

    ``endpoint_equalities`` must equal zero.  Every entry of
    ``inequality_constraints`` and each named margin array must be nonnegative.
    Thus callers can pass these arrays directly to an epigraph/trust-constr
    layer without reconstructing or changing the numerical definition.
    ``gammas`` uses the certified bounded extension only when initial q is
    infeasible; ``unfloored_gammas`` always exposes the literal log ratio.
    """

    temperature = (
        config.softmin_temperature
        if softmin_temperature is None
        else softmin_temperature
    )
    if temperature <= 0.0:
        raise ValueError("softmin_temperature must be positive")
    grids = tuple(SpectralGrid(n) for n in config.grid_sizes)
    outcomes = [low_mode_trajectory(coordinates, grid, config) for grid in grids]
    final_states = tuple(outcome[0] for outcome in outcomes)
    trajectories = jnp.stack([outcome[1] for outcome in outcomes])

    enstrophy = trajectories[:, :, ENSTROPHY]
    rate = trajectories[:, :, ENSTROPHY_RATE]
    initial_q_values = rate[:, 0] / jnp.maximum(
        enstrophy[:, 0], jnp.finfo(jnp.float64).tiny
    ) ** (2.0 + config.delta)
    initial_q_margins = (
        initial_q_values - config.initial_q_floor
    ) / config.initial_q_floor
    # P <= 4E for the controlled |k|^2 <= 4 initial band.  Therefore the
    # certified q floor implies this rigorous lower bound on initial zeta.
    initial_zeta_floor = (
        config.initial_q_floor
        * config.target_enstrophy ** (0.75 + config.delta)
        / 4.0**0.75
    )

    gamma_rows: list[Array] = []
    unfloored_gamma_rows: list[Array] = []
    crossing_time_rows: list[Array] = []
    reached_rows: list[Array] = []
    for grid_index, history in enumerate(trajectories):
        # Only infeasible states receive the bounded baseline extension.
        # Once q(0) >= q_floor, the low-band implication guarantees that the
        # floor is inactive and the score is the exact gamma.
        active_baseline_floor = jnp.where(
            initial_q_margins[grid_index] < -config.hard_margin_tolerance,
            initial_zeta_floor,
            0.0,
        )
        gamma, crossing_times, reached = gamma_at_amplifications(
            history[:, ENSTROPHY],
            history[:, ZETA],
            config.amplification_levels,
            dt=config.dt,
            baseline_zeta_floor=active_baseline_floor,
        )
        unfloored_gamma = gamma_at_amplifications(
            history[:, ENSTROPHY],
            history[:, ZETA],
            config.amplification_levels,
            dt=config.dt,
        )
        gamma_rows.append(gamma)
        unfloored_gamma_rows.append(unfloored_gamma[0])
        crossing_time_rows.append(crossing_times)
        reached_rows.append(reached)
    gammas = jnp.stack(gamma_rows)
    unfloored_gammas = jnp.stack(unfloored_gamma_rows)
    crossing_times = jnp.stack(crossing_time_rows)
    reached = jnp.stack(reached_rows)

    duration = config.steps * config.dt
    rate_scale = config.target_enstrophy / duration
    increment_scale = config.target_enstrophy * config.dt / duration
    rate_margins = rate / rate_scale
    monotonicity_margins = jnp.diff(enstrophy, axis=1) / increment_scale

    velocity_limits = jnp.asarray(config.velocity_tail_limits)[:, None]
    enstrophy_limits = jnp.asarray(config.enstrophy_tail_limits)[:, None]
    palinstrophy_limits = jnp.asarray(config.palinstrophy_tail_limits)[:, None]
    velocity_tail_margins = (
        velocity_limits - trajectories[:, :, VELOCITY_TAIL]
    ) / velocity_limits
    enstrophy_tail_margins = (
        enstrophy_limits - trajectories[:, :, ENSTROPHY_TAIL]
    ) / enstrophy_limits
    palinstrophy_tail_margins = (
        palinstrophy_limits - trajectories[:, :, PALINSTROPHY_TAIL]
    ) / palinstrophy_limits
    cfl_margins = (config.cfl_limit - trajectories[:, :, CFL]) / config.cfl_limit
    endpoint_equalities = jnp.log(
        jnp.maximum(enstrophy[:, -1], jnp.finfo(jnp.float64).tiny)
        / enstrophy[:, 0]
    ) - jnp.log(config.target_amplification)

    inequality_constraints = jnp.concatenate(
        tuple(
            values.reshape(-1)
            for values in (
                initial_q_margins,
                rate_margins,
                monotonicity_margins,
                velocity_tail_margins,
                enstrophy_tail_margins,
                palinstrophy_tail_margins,
                cfl_margins,
            )
        )
    )

    constraint_temperature = config.constraint_temperature
    initial_q_loss = jnp.sum(
        _smooth_violation(initial_q_margins, constraint_temperature) ** 2
    )
    # A full-trajectory hard limit must not be diluted by averaging hundreds
    # of feasible samples around one bad point.  Penalise the worst smooth
    # violation on each grid, then sum over grids and constraint families.
    rate_worst = jnp.max(
        _smooth_violation(rate_margins, constraint_temperature), axis=1
    )
    increment_worst = jnp.max(
        _smooth_violation(monotonicity_margins, constraint_temperature), axis=1
    )
    monotonicity_loss = jnp.sum(rate_worst**2 + increment_worst**2)
    tail_loss = jnp.sum(
        jnp.max(
            _smooth_violation(velocity_tail_margins, constraint_temperature),
            axis=1,
        )
        ** 2
        + jnp.max(
            _smooth_violation(enstrophy_tail_margins, constraint_temperature),
            axis=1,
        )
        ** 2
        + jnp.max(
            _smooth_violation(palinstrophy_tail_margins, constraint_temperature),
            axis=1,
        )
        ** 2
    )
    cfl_loss = jnp.sum(
        jnp.max(_smooth_violation(cfl_margins, constraint_temperature), axis=1)
        ** 2
    )
    endpoint_loss = jnp.sum(endpoint_equalities**2)
    penalty = (
        config.initial_q_penalty * initial_q_loss
        + config.endpoint_penalty * endpoint_loss
        + config.tail_penalty * tail_loss
        + config.monotonicity_penalty * monotonicity_loss
        + config.cfl_penalty * cfl_loss
    )
    bottleneck = -temperature * jax.scipy.special.logsumexp(
        -gammas.reshape(-1) / temperature
    )
    objective = bottleneck - penalty

    inequality_group_margins = jnp.stack(
        (
            initial_q_margins,
            jnp.min(rate_margins, axis=1),
            jnp.min(monotonicity_margins, axis=1),
            jnp.min(velocity_tail_margins, axis=1),
            jnp.min(enstrophy_tail_margins, axis=1),
            jnp.min(palinstrophy_tail_margins, axis=1),
            jnp.min(cfl_margins, axis=1),
        ),
        axis=1,
    )
    endpoint_tolerance_margins = (
        config.endpoint_tolerance - jnp.abs(endpoint_equalities)
    ) / config.endpoint_tolerance
    protected_group_margins = jnp.concatenate(
        (inequality_group_margins, endpoint_tolerance_margins[:, None]), axis=1
    )
    hard_infeasibility = (
        jnp.sum(
            jnp.maximum(
                -inequality_group_margins - config.hard_margin_tolerance, 0.0
            )
            ** 2
        )
        + jnp.sum(endpoint_equalities**2)
        + 1.0e-2 * jnp.sum(jnp.logical_not(reached))
    )

    return {
        "objective": objective,
        "bottleneck_gamma": bottleneck,
        "penalty": penalty,
        "gammas": gammas,
        "betas": 0.25 - gammas,
        "unfloored_gammas": unfloored_gammas,
        "crossing_times": crossing_times,
        "reached": reached,
        "trajectories": trajectories,
        "final_states": final_states,
        "endpoint_equalities": endpoint_equalities,
        "inequality_constraints": inequality_constraints,
        "initial_q_values": initial_q_values,
        "initial_q_margins": initial_q_margins,
        "initial_zeta_floor": jnp.asarray(initial_zeta_floor),
        "initial_q_loss": initial_q_loss,
        "inequality_group_margins": inequality_group_margins,
        "protected_group_margins": protected_group_margins,
        "hard_infeasibility": hard_infeasibility,
        "rate_margins": rate_margins,
        "monotonicity_margins": monotonicity_margins,
        "velocity_tail_margins": velocity_tail_margins,
        "enstrophy_tail_margins": enstrophy_tail_margins,
        "palinstrophy_tail_margins": palinstrophy_tail_margins,
        "cfl_margins": cfl_margins,
    }


def joint_critical_beta_objective(
    coordinates: Array,
    config: MultiResolutionConfig,
    *,
    softmin_temperature: float | None = None,
) -> Array:
    """Penalised smooth worst-grid/worst-amplification gamma objective."""

    return multiresolution_raw_evaluation(
        coordinates, config, softmin_temperature=softmin_temperature
    )["objective"]


def epigraph_constraint_values(
    coordinates: Array,
    epigraph_gamma: Array,
    config: MultiResolutionConfig,
) -> tuple[Array, Array]:
    """Return equality and nonnegative inequality values for trust-constr.

    The first block of inequalities is ``gamma - epigraph_gamma``.  Maximising
    the scalar epigraph subject to these and the returned trajectory margins
    implements the exact nonsmooth worst-case objective after an Adam warm
    start.  The sphere equality is omitted because callers may either add
    ``dot(coordinates, coordinates)-1`` or use a 63-dimensional chart.
    """

    raw = multiresolution_raw_evaluation(coordinates, config)
    gamma_margins = raw["gammas"].reshape(-1) - epigraph_gamma
    inequalities = jnp.concatenate(
        (gamma_margins, raw["inequality_constraints"])
    )
    return raw["endpoint_equalities"], inequalities


@lru_cache(maxsize=32)
def make_joint_value_and_grad(
    config: MultiResolutionConfig,
    softmin_temperature: float | None = None,
) -> Callable[[Array], tuple[Array, Array]]:
    """Build and cache a compiled joint value/gradient evaluator."""

    return jax.jit(
        jax.value_and_grad(
            lambda coordinates: joint_critical_beta_objective(
                coordinates,
                config,
                softmin_temperature=softmin_temperature,
            )
        )
    )


class _OptimizerState(NamedTuple):
    objective: Array
    hard_infeasibility: Array
    protected_margins: Array
    all_reached: Array
    minimum_initial_q_margin: Array


def _optimizer_state(
    coordinates: Array,
    config: MultiResolutionConfig,
    softmin_temperature: float,
) -> _OptimizerState:
    raw = multiresolution_raw_evaluation(
        coordinates, config, softmin_temperature=softmin_temperature
    )
    return _OptimizerState(
        raw["objective"],
        raw["hard_infeasibility"],
        # Preserve every (grid, constraint-family) margin independently.  A
        # reduction over grids would allow an already-feasible family on one
        # grid to be sacrificed whenever the same family remained infeasible
        # on another grid.
        raw["protected_group_margins"].reshape(-1),
        jnp.all(raw["reached"]),
        jnp.min(raw["initial_q_margins"]),
    )


@lru_cache(maxsize=32)
def make_optimizer_value_and_grad(
    config: MultiResolutionConfig,
    softmin_temperature: float,
):
    """Compile an objective gradient carrying compact feasibility telemetry."""

    def objective_with_aux(coordinates: Array) -> tuple[Array, _OptimizerState]:
        state = _optimizer_state(coordinates, config, softmin_temperature)
        return state.objective, state

    return jax.jit(jax.value_and_grad(objective_with_aux, has_aux=True))


@lru_cache(maxsize=32)
def make_optimizer_state_evaluator(
    config: MultiResolutionConfig,
    softmin_temperature: float,
):
    """Compile the post-step value and feasibility evaluator used by line search."""

    return jax.jit(
        lambda coordinates: _optimizer_state(
            coordinates, config, softmin_temperature
        )
    )


def initial_q_margin(coordinates: Array, config: MultiResolutionConfig) -> Array:
    """Evaluate the normalized certified q margin from the initial state only."""

    grid = SpectralGrid(min(config.grid_sizes))
    state = lift_control(coordinates, grid, config.target_enstrophy)
    values = diagnostics(state, grid, config.viscosity)
    q_delta = values["enstrophy_rate"] / jnp.maximum(
        values["enstrophy"], jnp.finfo(jnp.float64).tiny
    ) ** (2.0 + config.delta)
    return (q_delta - config.initial_q_floor) / config.initial_q_floor


@lru_cache(maxsize=16)
def make_initial_q_margin_value_and_grad(config: MultiResolutionConfig):
    """Compile the cheap initial-state active-constraint gradient."""

    return jax.jit(jax.value_and_grad(lambda coordinates: initial_q_margin(
        coordinates, config
    )))


@lru_cache(maxsize=16)
def make_raw_evaluator(
    config: MultiResolutionConfig,
    softmin_temperature: float | None = None,
) -> Callable[[Array], dict[str, Array]]:
    """Build and cache a compiled full evaluator for reporting/constraints."""

    return jax.jit(
        lambda coordinates: multiresolution_raw_evaluation(
            coordinates, config, softmin_temperature=softmin_temperature
        )
    )


def summarize_multiresolution(
    coordinates: Array,
    config: MultiResolutionConfig,
    *,
    softmin_temperature: float | None = None,
) -> dict[str, object]:
    """Return JSON-friendly exact scores and hard constraint summaries."""

    raw = make_raw_evaluator(config, softmin_temperature)(coordinates)
    histories = np.asarray(raw["trajectories"])
    gammas = np.asarray(raw["gammas"])
    unfloored_gammas = np.asarray(raw["unfloored_gammas"])
    initial_q_values = np.asarray(raw["initial_q_values"])
    initial_q_margins = np.asarray(raw["initial_q_margins"])
    reached = np.asarray(raw["reached"])
    endpoint = np.asarray(raw["endpoint_equalities"])
    reports: list[dict[str, object]] = []
    for index, n in enumerate(config.grid_sizes):
        history = histories[index]
        reports.append(
            {
                "n": n,
                "gammas": gammas[index].tolist(),
                "betas": (0.25 - gammas[index]).tolist(),
                "unfloored_gammas": unfloored_gammas[index].tolist(),
                "initial_q_delta": float(initial_q_values[index]),
                "initial_q_floor": config.initial_q_floor,
                "initial_q_margin": float(initial_q_margins[index]),
                "initial_zeta": float(history[0, ZETA]),
                "initial_zeta_floor": float(raw["initial_zeta_floor"]),
                "initial_bottleneck_feasible": bool(
                    initial_q_margins[index] >= -config.hard_margin_tolerance
                ),
                "crossing_times": np.asarray(raw["crossing_times"])[index].tolist(),
                "reached": reached[index].tolist(),
                "endpoint_log_amplification_residual": float(endpoint[index]),
                "minimum_enstrophy_rate": float(np.min(history[:, ENSTROPHY_RATE])),
                "minimum_enstrophy_increment": float(
                    np.min(np.diff(history[:, ENSTROPHY]))
                ),
                "maximum_velocity_tail": float(np.max(history[:, VELOCITY_TAIL])),
                "maximum_enstrophy_tail": float(np.max(history[:, ENSTROPHY_TAIL])),
                "maximum_palinstrophy_tail": float(
                    np.max(history[:, PALINSTROPHY_TAIL])
                ),
                "maximum_cfl": float(np.max(history[:, CFL])),
                "hard_feasible": bool(
                    np.all(reached[index])
                    and initial_q_margins[index]
                    >= -config.hard_margin_tolerance
                    and abs(endpoint[index]) <= config.endpoint_tolerance
                    and np.min(history[:, ENSTROPHY_RATE]) >= 0.0
                    and np.min(np.diff(history[:, ENSTROPHY])) >= 0.0
                    and np.max(history[:, VELOCITY_TAIL])
                    <= config.velocity_tail_limits[index]
                    and np.max(history[:, ENSTROPHY_TAIL])
                    <= config.enstrophy_tail_limits[index]
                    and np.max(history[:, PALINSTROPHY_TAIL])
                    <= config.palinstrophy_tail_limits[index]
                    and np.max(history[:, CFL]) < config.cfl_limit
                ),
            }
        )
    return {
        "objective_version": MULTIRES_OBJECTIVE_VERSION,
        "objective": float(raw["objective"]),
        "bottleneck_gamma": float(raw["bottleneck_gamma"]),
        "minimum_scoring_gamma": float(np.min(gammas)),
        "minimum_exact_gamma": (
            float(np.min(unfloored_gammas))
            if np.all(initial_q_margins >= -config.hard_margin_tolerance)
            else None
        ),
        "minimum_initial_q": float(np.min(initial_q_values)),
        "minimum_initial_q_margin": float(np.min(initial_q_margins)),
        "initial_q_floor": config.initial_q_floor,
        "initial_zeta_floor": float(raw["initial_zeta_floor"]),
        "hard_infeasibility": float(raw["hard_infeasibility"]),
        "penalty": float(raw["penalty"]),
        "amplification_levels": list(config.amplification_levels),
        "all_hard_feasible": all(bool(report["hard_feasible"]) for report in reports),
        "grids": reports,
    }


class AdamStepTelemetry(NamedTuple):
    """Post-update state recorded for one safeguarded Adam iteration."""

    accepted: bool
    accepted_step: float
    backtracks: int
    hard_infeasibility: float
    minimum_initial_q_margin: float
    minimum_protected_margin: float
    all_reached: bool
    q_direction_adjusted: bool
    q_restoration: bool
    q_directional_derivative_before: float
    q_directional_derivative_after: float
    q_projection_norm: float
    q_projection_relative_norm: float


class AdamStageTermination(NamedTuple):
    """How one fixed-temperature Adam stage ended.

    A rejection-patience exit is a numerical convergence control: no
    acceptable retracted step was found for the stated number of consecutive
    full line searches.  It is not a stationarity or KKT certificate.
    """

    stage_index: int
    temperature: float
    requested_iterations: int
    executed_iterations: int
    consecutive_rejections: int
    reason: str


class RiemannianAdamResult(NamedTuple):
    coordinates: Array
    objective_history: list[float]
    temperature_history: list[float]
    telemetry_history: list[AdamStepTelemetry]
    stage_terminations: list[AdamStageTermination]


class TrustConstrResult(NamedTuple):
    """Compact serialisable result of exact epigraph refinement."""

    coordinates: np.ndarray
    epigraph_gamma: float
    success: bool
    status: int
    message: str
    iterations: int
    optimality: float
    constraint_violation: float
    solver_optimality: float
    solver_constraint_violation: float
    normalization_displacement: float


def trust_constr_epigraph_refine(
    coordinates0: Array,
    config: MultiResolutionConfig,
    *,
    max_iterations: int = 25,
    gtol: float = 1.0e-5,
    callback: Callable[[int, np.ndarray, float, object], None] | None = None,
    verbose: int = 0,
) -> TrustConstrResult:
    """Refine an Adam result with SciPy's exact constrained epigraph solve.

    The nonlinear constraint vector contains the sphere equality, one endpoint
    equality per state grid, every ``gamma - eta`` epigraph margin, and every
    sampled hard trajectory margin.  A forward-mode JAX Jacobian is used
    because there are only 65 variables and thousands of constraint outputs.
    ``max_iterations`` intentionally bounds this expensive proof-search stage.
    """

    if max_iterations < 1 or gtol <= 0.0:
        raise ValueError("max_iterations and gtol must be positive")
    from scipy.optimize import BFGS, NonlinearConstraint, minimize

    point = np.asarray(normalize_coordinates(coordinates0), dtype=np.float64)
    initial_raw = make_raw_evaluator(config)(jnp.asarray(point))
    initial_epigraph = float(jnp.min(initial_raw["gammas"]))
    initial = np.concatenate((point, np.asarray([initial_epigraph])))

    def constraint_map(parameters: Array) -> Array:
        coordinates = parameters[:CONTROL_DIMENSION]
        epigraph = parameters[CONTROL_DIMENSION]
        equalities, inequalities = epigraph_constraint_values(
            coordinates, epigraph, config
        )
        sphere = jnp.vdot(coordinates, coordinates) - 1.0
        return jnp.concatenate((sphere[None], equalities, inequalities))

    compiled_constraints = jax.jit(constraint_map)
    compiled_jacobian = jax.jit(jax.jacfwd(constraint_map))
    initial_constraints = np.asarray(compiled_constraints(jnp.asarray(initial)))
    equality_count = 1 + len(config.grid_sizes)
    lower = np.zeros_like(initial_constraints)
    upper = np.full_like(initial_constraints, np.inf)
    upper[:equality_count] = 0.0

    def objective(parameters: np.ndarray) -> float:
        return -float(parameters[-1])

    def objective_jacobian(parameters: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(parameters)
        gradient[-1] = -1.0
        return gradient

    constraint = NonlinearConstraint(
        lambda parameters: np.asarray(
            compiled_constraints(jnp.asarray(parameters)), dtype=np.float64
        ),
        lower,
        upper,
        jac=lambda parameters: np.asarray(
            compiled_jacobian(jnp.asarray(parameters)), dtype=np.float64
        ),
        keep_feasible=False,
    )
    callback_iteration = 0

    def scipy_callback(parameters: np.ndarray, state: object) -> bool:
        nonlocal callback_iteration
        callback_iteration += 1
        if callback is not None:
            callback(
                callback_iteration,
                np.asarray(parameters[:-1]),
                float(parameters[-1]),
                state,
            )
        return False

    result = minimize(
        objective,
        initial,
        method="trust-constr",
        jac=objective_jacobian,
        hess=BFGS(),
        constraints=(constraint,),
        callback=scipy_callback,
        options={
            "maxiter": max_iterations,
            "gtol": gtol,
            "xtol": 1.0e-10,
            "barrier_tol": 1.0e-10,
            "verbose": verbose,
        },
    )
    solver_coordinates = np.asarray(result.x[:-1], dtype=np.float64)
    refined_coordinates = np.asarray(
        normalize_coordinates(solver_coordinates), dtype=np.float64
    )
    normalization_displacement = float(
        np.linalg.norm(refined_coordinates - solver_coordinates)
    )
    returned_parameters = np.concatenate(
        (refined_coordinates, np.asarray([result.x[-1]], dtype=np.float64))
    )
    returned_constraints = np.asarray(
        compiled_constraints(jnp.asarray(returned_parameters)), dtype=np.float64
    )
    equality_violation = float(
        np.max(np.abs(returned_constraints[:equality_count]), initial=0.0)
    )
    inequality_violation = float(
        np.max(
            np.maximum(-returned_constraints[equality_count:], 0.0),
            initial=0.0,
        )
    )
    returned_constraint_violation = max(
        equality_violation, inequality_violation
    )

    # SciPy reports optimality at its unnormalised solver point.  Re-evaluate
    # the Lagrangian gradient at the coordinates actually returned, using the
    # solver's final nonlinear-constraint multipliers when available.
    returned_optimality = float(result.optimality)
    multiplier_blocks = getattr(result, "v", None)
    if isinstance(multiplier_blocks, (list, tuple)) and multiplier_blocks:
        multipliers = np.asarray(multiplier_blocks[0], dtype=np.float64).reshape(-1)
        if multipliers.size == returned_constraints.size:
            returned_jacobian = np.asarray(
                compiled_jacobian(jnp.asarray(returned_parameters)),
                dtype=np.float64,
            )
            lagrangian_gradient = (
                objective_jacobian(returned_parameters)
                + returned_jacobian.T @ multipliers
            )
            returned_optimality = float(
                np.linalg.norm(lagrangian_gradient, ord=np.inf)
            )
    return TrustConstrResult(
        refined_coordinates,
        float(result.x[-1]),
        bool(result.success),
        int(result.status),
        str(result.message),
        int(result.nit),
        returned_optimality,
        returned_constraint_violation,
        float(result.optimality),
        float(result.constr_violation),
        normalization_displacement,
    )


def deterministic_tangent_perturbations(
    base: Array,
    *,
    count: int = 5,
    scale: float = 0.05,
    seed: int = 1729,
) -> list[np.ndarray]:
    """Generate reproducible unit-sphere perturbations of a control."""

    if count < 0 or scale <= 0.0:
        raise ValueError("count must be nonnegative and scale positive")
    base_array = np.asarray(normalize_coordinates(base))
    generator = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(count):
        direction = generator.standard_normal(CONTROL_DIMENSION)
        direction -= np.dot(direction, base_array) * base_array
        direction /= np.linalg.norm(direction)
        perturbed = base_array + scale * direction
        perturbed /= np.linalg.norm(perturbed)
        result.append(perturbed)
    return result


def tangent_gradient_check(
    coordinates: Array,
    config: MultiResolutionConfig,
    *,
    directions: int = 5,
    epsilon: float = 1.0e-5,
    seed: int = 2718,
    softmin_temperature: float | None = None,
) -> list[dict[str, float]]:
    """Compare AD and central finite differences along sphere tangents."""

    if directions < 1 or epsilon <= 0.0:
        raise ValueError("directions and epsilon must be positive")
    point = normalize_coordinates(coordinates)
    point_host = np.asarray(point)
    value_and_grad = make_joint_value_and_grad(config, softmin_temperature)
    _, gradient = value_and_grad(point)
    tangent_gradient = gradient - jnp.vdot(point, gradient) * point
    generator = np.random.default_rng(seed)
    reports: list[dict[str, float]] = []
    for _ in range(directions):
        direction = generator.standard_normal(CONTROL_DIMENSION)
        direction -= np.dot(direction, point_host) * point_host
        direction /= np.linalg.norm(direction)
        plus = normalize_coordinates(point + epsilon * direction)
        minus = normalize_coordinates(point - epsilon * direction)
        plus_value = value_and_grad(plus)[0]
        minus_value = value_and_grad(minus)[0]
        finite_difference = float((plus_value - minus_value) / (2.0 * epsilon))
        automatic = float(jnp.vdot(tangent_gradient, jnp.asarray(direction)))
        relative_error = abs(finite_difference - automatic) / max(
            1.0e-8, abs(finite_difference), abs(automatic)
        )
        reports.append(
            {
                "automatic": automatic,
                "finite_difference": finite_difference,
                "relative_error": relative_error,
            }
        )
    return reports


def multistage_riemannian_adam(
    coordinates0: Array,
    config: MultiResolutionConfig,
    *,
    stages: Sequence[tuple[int, float]] = (
        (100, 2.0e-3),
        (75, 5.0e-4),
        (75, 2.0e-4),
    ),
    learning_rate: float = 0.005,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 9,
    stage_rejection_patience: int | None = 20,
    feasibility_tolerance: float = 1.0e-9,
    q_active_tolerance: float = 2.0e-2,
    q_restoration_cosine: float = 0.1,
    q_boundary_cosine: float = 1.0e-3,
    q_boundary_buffer: float = 1.0e-6,
    callback: Callable[
        [int, float, float, Array, AdamStepTelemetry], None
    ]
    | None = None,
) -> RiemannianAdamResult:
    """Maximise the joint objective with safeguarded Riemannian Adam.

    Trial steps are retracted to the sphere and backtracked while protecting
    every currently feasible initial-q/rate/tail/CFL/endpoint group and the
    reached-amplification condition.  Among those protected trials, a step is
    accepted when either the penalised objective or aggregate infeasibility
    improves.  The saved value and callback telemetry are always recomputed at
    the post-update coordinates.
    """

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < backtrack_factor < 1.0 or max_backtracks < 0:
        raise ValueError("backtrack_factor must lie in (0,1) and count be nonnegative")
    if stage_rejection_patience is not None and stage_rejection_patience < 1:
        raise ValueError("stage_rejection_patience must be positive or None")
    if feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    if q_active_tolerance < 0.0:
        raise ValueError("q_active_tolerance must be nonnegative")
    if not 0.0 < q_restoration_cosine <= 1.0:
        raise ValueError("q_restoration_cosine must lie in (0,1]")
    if not 0.0 < q_boundary_cosine <= 1.0:
        raise ValueError("q_boundary_cosine must lie in (0,1]")
    if q_boundary_buffer < 0.0:
        raise ValueError("q_boundary_buffer must be nonnegative")
    if not stages or any(iterations < 1 or temperature <= 0.0 for iterations, temperature in stages):
        raise ValueError("each stage must have positive iterations and temperature")
    coordinates = normalize_coordinates(coordinates0)
    first = jnp.zeros_like(coordinates)
    second = jnp.zeros_like(coordinates)
    beta1, beta2 = 0.9, 0.999
    objective_history: list[float] = []
    temperature_history: list[float] = []
    telemetry_history: list[AdamStepTelemetry] = []
    stage_terminations: list[AdamStageTermination] = []
    iteration = 0
    q_value_and_grad = make_initial_q_margin_value_and_grad(config)

    for stage_index, (stage_iterations, temperature) in enumerate(stages):
        value_and_grad = make_optimizer_value_and_grad(config, float(temperature))
        state_evaluator = make_optimizer_state_evaluator(config, float(temperature))
        consecutive_rejections = 0
        executed_iterations = 0
        termination_reason = "completed_requested_iterations"
        for _ in range(stage_iterations):
            iteration += 1
            executed_iterations += 1
            (value, current_state), gradient = value_and_grad(coordinates)
            tangent_gradient = gradient - jnp.vdot(coordinates, gradient) * coordinates
            proposed_first = beta1 * first + (1.0 - beta1) * tangent_gradient
            proposed_second = beta2 * second + (1.0 - beta2) * tangent_gradient**2
            first_hat = proposed_first / (1.0 - beta1**iteration)
            second_hat = proposed_second / (1.0 - beta2**iteration)
            direction = first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
            direction -= jnp.vdot(coordinates, direction) * coordinates

            # First-order active-set handling for the certified initial
            # bottleneck.  A merely tangent direction can leave the feasible
            # side through negative second-order curvature after retraction,
            # so near the boundary we request a small, scale-aware *inward*
            # derivative.  Below the floor, minimally add a stronger restoring
            # component that aims for a positive normalized-margin buffer.
            q_margin_value, q_gradient = q_value_and_grad(coordinates)
            q_gradient -= jnp.vdot(coordinates, q_gradient) * coordinates
            q_gradient_norm_squared = jnp.vdot(q_gradient, q_gradient)
            direction_norm = jnp.linalg.norm(direction)
            q_derivative_before = jnp.vdot(q_gradient, direction)
            correction = jnp.zeros_like(direction)
            q_direction_adjusted = False
            q_restoration = False
            q_margin_scalar = float(q_margin_value)
            gradient_norm_squared_scalar = float(q_gradient_norm_squared)
            derivative_before_scalar = float(q_derivative_before)
            if gradient_norm_squared_scalar > 1.0e-24:
                if q_margin_scalar < -feasibility_tolerance:
                    q_restoration = True
                    target_derivative = max(
                        q_restoration_cosine
                        * np.sqrt(gradient_norm_squared_scalar)
                        * float(direction_norm),
                        (q_boundary_buffer - q_margin_scalar) / learning_rate,
                    )
                    if derivative_before_scalar < target_derivative:
                        correction = (
                            (target_derivative - q_derivative_before)
                            / q_gradient_norm_squared
                            * q_gradient
                        )
                        q_direction_adjusted = True
                elif q_margin_scalar <= q_active_tolerance:
                    target_derivative = max(
                        q_boundary_cosine
                        * np.sqrt(gradient_norm_squared_scalar)
                        * float(direction_norm),
                        max(q_boundary_buffer - q_margin_scalar, 0.0)
                        / learning_rate,
                    )
                    if derivative_before_scalar < target_derivative:
                        correction = (
                            (target_derivative - q_derivative_before)
                            / q_gradient_norm_squared
                            * q_gradient
                        )
                        q_direction_adjusted = True
            direction = direction + correction
            q_derivative_after = jnp.vdot(q_gradient, direction)
            correction_norm = float(jnp.linalg.norm(correction))
            correction_relative_norm = correction_norm / max(
                float(direction_norm), 1.0e-30
            )

            current_value = float(value)
            current_hard = float(current_state.hard_infeasibility)
            current_margins = np.asarray(current_state.protected_margins)
            current_q_margin = float(current_state.minimum_initial_q_margin)
            accepted = False
            accepted_step = 0.0
            used_backtracks = max_backtracks + 1
            post_state = current_state
            trial_coordinates = coordinates
            for backtracks in range(max_backtracks + 1):
                step = learning_rate * backtrack_factor**backtracks
                candidate = normalize_coordinates(coordinates + step * direction)
                candidate_state = state_evaluator(candidate)
                candidate_value = float(candidate_state.objective)
                candidate_hard = float(candidate_state.hard_infeasibility)
                candidate_margins = np.asarray(
                    candidate_state.protected_margins
                )
                finite = bool(
                    np.isfinite(candidate_value)
                    and np.isfinite(candidate_hard)
                    and np.isfinite(candidate_margins).all()
                )
                preserves_feasible_groups = bool(
                    np.all(
                        np.logical_or(
                            current_margins < -feasibility_tolerance,
                            candidate_margins >= -feasibility_tolerance,
                        )
                    )
                )
                preserves_reached = bool(
                    not bool(current_state.all_reached)
                    or bool(candidate_state.all_reached)
                )
                candidate_q_margin = float(
                    candidate_state.minimum_initial_q_margin
                )
                q_not_worse = bool(
                    candidate_q_margin >= -feasibility_tolerance
                    if current_q_margin >= -feasibility_tolerance
                    else candidate_q_margin
                    >= current_q_margin - feasibility_tolerance
                )
                feasibility_improved = bool(
                    candidate_hard
                    < current_hard
                    - feasibility_tolerance * max(1.0, abs(current_hard))
                )
                objective_not_worse = bool(
                    candidate_value >= current_value - 1.0e-10
                )
                if (
                    finite
                    and preserves_feasible_groups
                    and preserves_reached
                    and q_not_worse
                    and (objective_not_worse or feasibility_improved)
                ):
                    accepted = True
                    accepted_step = step
                    used_backtracks = backtracks
                    trial_coordinates = candidate
                    post_state = candidate_state
                    break

            if accepted:
                coordinates = trial_coordinates
                first = proposed_first
                second = proposed_second
                # Vector-transport the first moment to the new tangent plane.
                first -= jnp.vdot(coordinates, first) * coordinates
            else:
                # Do not retain a momentum update that points outside the
                # conservative feasibility cone.
                first = 0.5 * first

            scalar = float(post_state.objective)
            if not np.isfinite(scalar):
                raise FloatingPointError(f"non-finite objective at iteration {iteration}")
            telemetry = AdamStepTelemetry(
                accepted,
                float(accepted_step),
                int(used_backtracks),
                float(post_state.hard_infeasibility),
                float(post_state.minimum_initial_q_margin),
                float(jnp.min(post_state.protected_margins)),
                bool(post_state.all_reached),
                q_direction_adjusted,
                q_restoration,
                derivative_before_scalar,
                float(q_derivative_after),
                correction_norm,
                correction_relative_norm,
            )
            objective_history.append(scalar)
            temperature_history.append(float(temperature))
            telemetry_history.append(telemetry)
            if callback is not None:
                callback(
                    iteration,
                    scalar,
                    float(temperature),
                    coordinates,
                    telemetry,
                )

            if accepted:
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
                if (
                    stage_rejection_patience is not None
                    and consecutive_rejections >= stage_rejection_patience
                ):
                    termination_reason = "rejection_patience_exhausted"
                    break
        stage_terminations.append(
            AdamStageTermination(
                stage_index=stage_index,
                temperature=float(temperature),
                requested_iterations=stage_iterations,
                executed_iterations=executed_iterations,
                consecutive_rejections=consecutive_rejections,
                reason=termination_reason,
            )
        )

    return RiemannianAdamResult(
        coordinates,
        objective_history,
        temperature_history,
        telemetry_history,
        stage_terminations,
    )


__all__ = [
    "AdamStepTelemetry",
    "AdamStageTermination",
    "CERTIFIED_P3_INITIAL_Q_FLOOR",
    "MULTIRES_DIAGNOSTIC_NAMES",
    "MULTIRES_OBJECTIVE_VERSION",
    "MultiResolutionConfig",
    "PROTECTED_MARGIN_NAMES",
    "RiemannianAdamResult",
    "TrustConstrResult",
    "deterministic_tangent_perturbations",
    "epigraph_constraint_values",
    "gamma_at_amplifications",
    "initial_q_margin",
    "joint_critical_beta_objective",
    "low_mode_trajectory",
    "make_joint_value_and_grad",
    "make_initial_q_margin_value_and_grad",
    "make_optimizer_state_evaluator",
    "make_optimizer_value_and_grad",
    "make_raw_evaluator",
    "multiresolution_raw_evaluation",
    "multistage_riemannian_adam",
    "summarize_multiresolution",
    "tangent_gradient_check",
    "trust_constr_epigraph_refine",
]
