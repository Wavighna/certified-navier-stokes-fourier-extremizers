"""Tools for reproducible searches of extreme incompressible flows."""

from .low_modes import (
    CANONICAL_WAVEVECTORS,
    CONTROL_DIMENSION,
    LowModeControl,
    lift_control,
    load_low_mode_candidate,
)
from .mechanisms import endpoint_mechanism_ledger, mechanism_diagnostics
from .multires import (
    AdamStepTelemetry,
    CERTIFIED_P3_INITIAL_Q_FLOOR,
    MultiResolutionConfig,
    epigraph_constraint_values,
    gamma_at_amplifications,
    initial_q_margin,
    joint_critical_beta_objective,
    multiresolution_raw_evaluation,
    multistage_riemannian_adam,
    summarize_multiresolution,
    trust_constr_epigraph_refine,
)
from .spectral import SpectralGrid, diagnostics, project_velocity, rhs

__all__ = [
    "AdamStepTelemetry",
    "CANONICAL_WAVEVECTORS",
    "CONTROL_DIMENSION",
    "CERTIFIED_P3_INITIAL_Q_FLOOR",
    "LowModeControl",
    "MultiResolutionConfig",
    "SpectralGrid",
    "diagnostics",
    "endpoint_mechanism_ledger",
    "epigraph_constraint_values",
    "gamma_at_amplifications",
    "initial_q_margin",
    "joint_critical_beta_objective",
    "lift_control",
    "load_low_mode_candidate",
    "mechanism_diagnostics",
    "multiresolution_raw_evaluation",
    "multistage_riemannian_adam",
    "project_velocity",
    "rhs",
    "summarize_multiresolution",
    "trust_constr_epigraph_refine",
]
