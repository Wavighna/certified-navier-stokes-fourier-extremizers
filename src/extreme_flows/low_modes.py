"""Resolution-independent low-mode controls for periodic vector fields.

The control space contains the sixteen conjugate Fourier pairs with
``0 < |k|^2 <= 4``.  Each representative has two complex, divergence-free
polarizations, hence the real control dimension is 16 * 2 * 2 = 64.

The coordinates are enstrophy-orthonormal: if ``x`` is on the Euclidean unit
sphere, :func:`lift_control` has exactly ``target_enstrophy`` independently of
the collocation grid.  Fourier coefficients in this module use the continuum
convention ``u(x) = sum_k c_k exp(i k dot x)``; lifting multiplies them by
``N^3`` for JAX's unnormalised forward FFT convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .spectral import SpectralGrid

Array = jax.Array
CONTROL_DIMENSION = 64
MAX_WAVENUMBER_SQUARED = 4


def _is_canonical_representative(k: Sequence[int]) -> bool:
    """Choose one vector from ``{k, -k}`` by its first nonzero component."""

    return next(component for component in k if component != 0) > 0


def _build_wavevectors() -> np.ndarray:
    vectors = [
        k
        for k in product(range(-2, 3), repeat=3)
        if 0 < sum(component * component for component in k)
        <= MAX_WAVENUMBER_SQUARED
        and _is_canonical_representative(k)
    ]
    vectors.sort(key=lambda k: (sum(component * component for component in k), k))
    result = np.asarray(vectors, dtype=np.int32)
    if result.shape != (16, 3):  # pragma: no cover - guards a mathematical invariant
        raise RuntimeError(f"expected 16 Fourier pairs, found {result.shape[0]}")
    return result


def _build_polarizations(wavevectors: np.ndarray) -> np.ndarray:
    """Construct a deterministic right-handed orthonormal transverse frame."""

    frames: list[np.ndarray] = []
    coordinate_axes = np.eye(3, dtype=np.float64)
    for integer_k in wavevectors:
        k = integer_k.astype(np.float64)
        khat = k / np.linalg.norm(k)
        # The least-aligned coordinate axis maximises numerical separation.
        seed = coordinate_axes[int(np.argmin(np.abs(k)))]
        first = np.cross(khat, seed)
        first /= np.linalg.norm(first)
        second = np.cross(khat, first)
        second /= np.linalg.norm(second)
        frames.append(np.stack((first, second)))
    return np.stack(frames)


CANONICAL_WAVEVECTORS = _build_wavevectors()
POLARIZATIONS = _build_polarizations(CANONICAL_WAVEVECTORS)
WAVENUMBER_SQUARED = np.sum(CANONICAL_WAVEVECTORS**2, axis=1).astype(np.float64)


def normalize_coordinates(coordinates: Array) -> Array:
    """Return a nonzero 64-vector retracted onto ``S^63``."""

    coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
    if coordinates.shape != (CONTROL_DIMENSION,):
        raise ValueError(
            f"low-mode coordinates must have shape ({CONTROL_DIMENSION},), "
            f"got {coordinates.shape}"
        )
    norm = jnp.linalg.norm(coordinates)
    return coordinates / jnp.maximum(norm, jnp.finfo(jnp.float64).tiny)


def mode_coefficients(
    coordinates: Array, target_enstrophy: float = 100.0
) -> Array:
    """Decode sphere coordinates into the 16 positive-mode coefficients."""

    if target_enstrophy <= 0.0:
        raise ValueError("target_enstrophy must be positive")
    unit = normalize_coordinates(coordinates).reshape(16, 2, 2)
    complex_amplitudes = unit[..., 0] + 1j * unit[..., 1]
    weights = jnp.sqrt(target_enstrophy) / jnp.sqrt(
        jnp.asarray(WAVENUMBER_SQUARED)
    )
    return weights[:, None] * jnp.einsum(
        "pa,paj->pj", complex_amplitudes, jnp.asarray(POLARIZATIONS)
    )


def lift_control(
    coordinates: Array,
    grid: SpectralGrid,
    target_enstrophy: float = 100.0,
) -> Array:
    """Lift a grid-independent control to an admissible spectral state."""

    if grid.n < 8:
        raise ValueError("the |k|^2 <= 4 basis requires an even grid N >= 8")
    coefficients = mode_coefficients(coordinates, target_enstrophy)
    fft_coefficients = coefficients * float(grid.n**3)
    positive = CANONICAL_WAVEVECTORS % grid.n
    negative = (-CANONICAL_WAVEVECTORS) % grid.n
    state = jnp.zeros((grid.n, grid.n, grid.n, 3), dtype=jnp.complex128)
    positive_index = tuple(jnp.asarray(positive[:, axis]) for axis in range(3))
    negative_index = tuple(jnp.asarray(negative[:, axis]) for axis in range(3))
    state = state.at[positive_index].set(fft_coefficients)
    state = state.at[negative_index].set(jnp.conj(fft_coefficients))
    return state


def coordinates_from_coefficients(
    coefficients: np.ndarray,
    *,
    target_enstrophy: float | None = None,
) -> tuple[np.ndarray, float]:
    """Encode the 16 positive coefficients in the orthonormal basis."""

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.shape != (16, 3):
        raise ValueError(f"coefficients must have shape (16, 3), got {coefficients.shape}")
    transverse = np.einsum("paj,pj->pa", POLARIZATIONS, coefficients)
    represented_enstrophy = float(
        np.sum(WAVENUMBER_SQUARED[:, None] * np.abs(transverse) ** 2)
    )
    if represented_enstrophy <= 0.0:
        raise ValueError("candidate has zero represented enstrophy")
    if target_enstrophy is None:
        target_enstrophy = represented_enstrophy
    if target_enstrophy <= 0.0:
        raise ValueError("target_enstrophy must be positive")
    scaled = (
        np.sqrt(WAVENUMBER_SQUARED)[:, None]
        * transverse
        / np.sqrt(target_enstrophy)
    )
    packed = np.stack((scaled.real, scaled.imag), axis=-1).reshape(-1)
    packed /= np.linalg.norm(packed)
    return packed, float(target_enstrophy)


def coordinates_from_state(
    state: np.ndarray,
    *,
    target_enstrophy: float | None = None,
) -> tuple[np.ndarray, float]:
    """Extract the controlled modes from a full unnormalised FFT state."""

    state = np.asarray(state)
    if state.ndim != 4 or state.shape[-1] != 3 or len(set(state.shape[:3])) != 1:
        raise ValueError("state must have shape (N, N, N, 3)")
    n = state.shape[0]
    positive = CANONICAL_WAVEVECTORS % n
    coefficients = np.stack(
        [state[tuple(index)] for index in positive], axis=0
    ) / float(n**3)
    return coordinates_from_coefficients(
        coefficients, target_enstrophy=target_enstrophy
    )


def coefficients_from_modes(modes: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Read continuum Fourier coefficients from an exported ``modes`` list."""

    table: dict[tuple[int, int, int], np.ndarray] = {}
    for mode in modes:
        k = tuple(int(component) for component in mode["k"])
        value = np.asarray(mode["real"], dtype=np.float64) + 1j * np.asarray(
            mode["imag"], dtype=np.float64
        )
        if len(k) != 3 or value.shape != (3,):
            raise ValueError("each mode must contain a 3-vector k, real, and imag")
        table[k] = value

    coefficients: list[np.ndarray] = []
    for integer_k in CANONICAL_WAVEVECTORS:
        k = tuple(int(component) for component in integer_k)
        minus_k = tuple(-component for component in k)
        if k in table:
            coefficients.append(table[k])
        elif minus_k in table:
            coefficients.append(np.conj(table[minus_k]))
        else:
            coefficients.append(np.zeros(3, dtype=np.complex128))
    return np.stack(coefficients)


@dataclass(frozen=True)
class LowModeControl:
    """Host-side, serialisable representation of a point on ``S^63``."""

    coordinates: np.ndarray
    target_enstrophy: float = 100.0

    def __post_init__(self) -> None:
        values = np.asarray(self.coordinates, dtype=np.float64).copy()
        if values.shape != (CONTROL_DIMENSION,):
            raise ValueError(
                f"coordinates must have shape ({CONTROL_DIMENSION},), got {values.shape}"
            )
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("coordinates must be finite and nonzero")
        if self.target_enstrophy <= 0.0:
            raise ValueError("target_enstrophy must be positive")
        values /= norm
        values.setflags(write=False)
        object.__setattr__(self, "coordinates", values)

    @classmethod
    def from_modes(
        cls,
        modes: Sequence[Mapping[str, Any]],
        *,
        target_enstrophy: float | None = None,
    ) -> "LowModeControl":
        coordinates, inferred_target = coordinates_from_coefficients(
            coefficients_from_modes(modes), target_enstrophy=target_enstrophy
        )
        return cls(coordinates, inferred_target)

    @classmethod
    def from_state(
        cls,
        state: np.ndarray,
        *,
        target_enstrophy: float | None = None,
    ) -> "LowModeControl":
        coordinates, inferred_target = coordinates_from_state(
            state, target_enstrophy=target_enstrophy
        )
        return cls(coordinates, inferred_target)

    @classmethod
    def from_candidate(cls, path: str | Path) -> "LowModeControl":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        target = document.get("target_enstrophy")
        if target is None:
            target = document.get("initial_diagnostics", {}).get("enstrophy")
        return cls.from_modes(document["modes"], target_enstrophy=target)

    def lift(self, grid: SpectralGrid) -> Array:
        return lift_control(self.coordinates, grid, self.target_enstrophy)

    def to_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        coefficients = np.asarray(
            mode_coefficients(self.coordinates, self.target_enstrophy)
        )
        modes: list[dict[str, Any]] = []
        for k, coefficient in zip(CANONICAL_WAVEVECTORS, coefficients, strict=True):
            for signed_k, signed_coefficient in (
                (k, coefficient), (-k, np.conj(coefficient))
            ):
                modes.append(
                    {
                        "k": [int(component) for component in signed_k],
                        "real": [float(value) for value in signed_coefficient.real],
                        "imag": [float(value) for value in signed_coefficient.imag],
                    }
                )
        result: dict[str, Any] = {
            "format_version": 2,
            "control_type": "enstrophy-orthonormal-low-modes-s63",
            "control_dimension": CONTROL_DIMENSION,
            "target_enstrophy": self.target_enstrophy,
            "control_norm": float(np.linalg.norm(self.coordinates)),
            "wavevector_cutoff_squared": MAX_WAVENUMBER_SQUARED,
            "canonical_pair_rule": "first nonzero component is positive",
            "coefficient_convention": "u(x)=sum_k c_k exp(i k dot x)",
            "reality_condition": "c[-k]=conjugate(c[k])",
            "coordinates": self.coordinates.tolist(),
            "canonical_wavevectors": CANONICAL_WAVEVECTORS.tolist(),
            "polarizations": POLARIZATIONS.tolist(),
            "modes": modes,
        }
        if metadata is not None:
            result["metadata"] = dict(metadata)
        return result

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(metadata=metadata), indent=2) + "\n",
            encoding="utf-8",
        )


def load_low_mode_candidate(path: str | Path) -> LowModeControl:
    """Load either the legacy mode-list schema or the v2 control schema."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "coordinates" in document:
        return LowModeControl(
            np.asarray(document["coordinates"], dtype=np.float64),
            float(document.get("target_enstrophy", 100.0)),
        )
    return LowModeControl.from_candidate(path)


__all__ = [
    "CANONICAL_WAVEVECTORS",
    "CONTROL_DIMENSION",
    "LowModeControl",
    "MAX_WAVENUMBER_SQUARED",
    "POLARIZATIONS",
    "WAVENUMBER_SQUARED",
    "coefficients_from_modes",
    "coordinates_from_coefficients",
    "coordinates_from_state",
    "lift_control",
    "load_low_mode_candidate",
    "mode_coefficients",
    "normalize_coordinates",
]
