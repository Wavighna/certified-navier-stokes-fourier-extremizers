"""Independent exact/interval certificates for the low-mode research data.

This module deliberately does not import JAX or :mod:`extreme_flows.spectral`.
The onset calculation starts with the literal decimal strings in an exported
candidate, performs all Fourier algebra over ``fractions.Fraction``, and uses
Arb only to enclose the remaining algebraic real numbers.

The static 64-variable utilities expose the exact sparse polynomial underlying
the proposed KKT proof.  The numerical optimizer and its Hessian inertia are
candidate-generation tools, not interval certificates; their output says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from hashlib import sha256
from itertools import combinations_with_replacement, permutations
import importlib.metadata
import json
from math import atan2, cos, sin, sqrt
import platform
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np

Wave = tuple[int, int, int]
Q = Fraction


def _q(value: int | str | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class QComplex:
    """A Gaussian rational, represented without floating-point arithmetic."""

    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __add__(self, other: QComplex | int | Fraction) -> QComplex:
        other = as_qcomplex(other)
        return QComplex(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self) -> QComplex:
        return QComplex(-self.real, -self.imag)

    def __sub__(self, other: QComplex | int | Fraction) -> QComplex:
        return self + (-as_qcomplex(other))

    def __rsub__(self, other: QComplex | int | Fraction) -> QComplex:
        return as_qcomplex(other) - self

    def __mul__(self, other: QComplex | int | Fraction) -> QComplex:
        other = as_qcomplex(other)
        return QComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: int | Fraction) -> QComplex:
        divisor = _q(other)
        return QComplex(self.real / divisor, self.imag / divisor)

    def conjugate(self) -> QComplex:
        return QComplex(self.real, -self.imag)

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0

    def to_complex(self) -> complex:
        return complex(float(self.real), float(self.imag))


def as_qcomplex(value: QComplex | int | Fraction) -> QComplex:
    if isinstance(value, QComplex):
        return value
    return QComplex(_q(value), Fraction(0))


QVector = tuple[QComplex, QComplex, QComplex]
SparseQField = dict[Wave, QVector]


def _zero_vector() -> QVector:
    return (QComplex(), QComplex(), QComplex())


def _vector_add(a: QVector, b: QVector) -> QVector:
    return tuple(a[i] + b[i] for i in range(3))  # type: ignore[return-value]


def _vector_scale(a: QVector, scalar: QComplex | Fraction | int) -> QVector:
    return tuple(scalar * a[i] for i in range(3))  # type: ignore[return-value]


def _vector_is_zero(a: QVector) -> bool:
    return all(component.is_zero() for component in a)


def wave_square(k: Wave) -> int:
    return k[0] * k[0] + k[1] * k[1] + k[2] * k[2]


def negate_wave(k: Wave) -> Wave:
    return (-k[0], -k[1], -k[2])


def is_positive_half_wave(k: Wave) -> bool:
    """Choose one member of a conjugate pair by first nonzero coordinate."""

    for component in k:
        if component:
            return component > 0
    raise ValueError("the zero mode has no positive-half representative")


def leray_project(k: Wave, vector: QVector) -> QVector:
    """Apply the exact rational Leray projector at a nonzero integer mode."""

    k2 = wave_square(k)
    if k2 == 0:
        raise ValueError("Leray projection is undefined at the zero mode")
    dot = sum((k[j] * vector[j] for j in range(3)), QComplex())
    return tuple(
        vector[j] - Fraction(k[j], k2) * dot for j in range(3)
    )  # type: ignore[return-value]


def field_add(*fields: Mapping[Wave, QVector]) -> SparseQField:
    keys: set[Wave] = set()
    for field in fields:
        keys.update(field)
    result: SparseQField = {}
    for k in keys:
        value = _zero_vector()
        for field in fields:
            value = _vector_add(value, field.get(k, _zero_vector()))
        if not _vector_is_zero(value):
            result[k] = value
    return result


def field_scale(field: Mapping[Wave, QVector], scalar: Fraction) -> SparseQField:
    return {
        k: _vector_scale(value, scalar)
        for k, value in field.items()
        if not _vector_is_zero(value)
    }


def bilinear_navier_stokes(
    left: Mapping[Wave, QVector], right: Mapping[Wave, QVector]
) -> SparseQField:
    """Return ``-P[(left . grad) right]`` in exact Fourier arithmetic."""

    accumulated: SparseQField = {}
    minus_i = QComplex(Fraction(0), Fraction(-1))
    for p, left_p in left.items():
        for q, right_q in right.items():
            k = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
            if k == (0, 0, 0):
                continue
            directional = sum(
                (q[j] * left_p[j] for j in range(3)), QComplex()
            )
            contribution = _vector_scale(right_q, minus_i * directional)
            accumulated[k] = _vector_add(
                accumulated.get(k, _zero_vector()), contribution
            )
    result: SparseQField = {}
    for k, value in accumulated.items():
        projected = leray_project(k, value)
        if not _vector_is_zero(projected):
            result[k] = projected
    return result


def viscous_field(
    field: Mapping[Wave, QVector], viscosity: Fraction
) -> SparseQField:
    return {
        k: _vector_scale(value, -viscosity * wave_square(k))
        for k, value in field.items()
    }


def weighted_real_inner(
    left: Mapping[Wave, QVector],
    right: Mapping[Wave, QVector],
    wave_power: int,
) -> Fraction:
    """Return ``Re sum |k|^(2p) conj(left_k).right_k`` exactly."""

    total = QComplex()
    for k in left.keys() & right.keys():
        dot = sum(
            (left[k][j].conjugate() * right[k][j] for j in range(3)),
            QComplex(),
        )
        total += (wave_square(k) ** wave_power) * dot
    if total.imag != 0:
        raise ArithmeticError("a real Fourier inner product acquired an imaginary part")
    return total.real


def enstrophy(field: Mapping[Wave, QVector]) -> Fraction:
    return Fraction(1, 2) * weighted_real_inner(field, field, 1)


def palinstrophy(field: Mapping[Wave, QVector]) -> Fraction:
    return Fraction(1, 2) * weighted_real_inner(field, field, 2)


@dataclass(frozen=True)
class ExactDatum:
    """The deterministic rational datum obtained from an exported candidate."""

    field: SparseQField
    source_path: str
    source_sha256: str

    def validate(self) -> None:
        if not self.field:
            raise ValueError("candidate has no nonzero Fourier modes")
        for k, value in self.field.items():
            if k == (0, 0, 0):
                raise ValueError("candidate contains a mean mode")
            divergence = sum((k[j] * value[j] for j in range(3)), QComplex())
            if not divergence.is_zero():
                raise ValueError(f"mode {k} is not exactly divergence-free")
            partner = self.field.get(negate_wave(k))
            expected = tuple(component.conjugate() for component in value)
            if partner != expected:
                raise ValueError(f"mode {k} does not have exact conjugate symmetry")


def load_exactified_candidate(path: str | Path) -> ExactDatum:
    """Read literal decimals as rationals, average pairs, project, and conjugate.

    Averaging ``c(k)`` with ``conj(c(-k))`` makes the handling of last-digit
    export asymmetries explicit.  Projection is then exact over the rationals.
    """

    source = Path(path)
    raw_bytes = source.read_bytes()
    payload = json.loads(raw_bytes, parse_float=Fraction)
    modes: dict[Wave, QVector] = {}
    for entry in payload["modes"]:
        k = tuple(int(value) for value in entry["k"])
        if len(k) != 3 or k == (0, 0, 0):
            raise ValueError(f"invalid Fourier mode {k}")
        if k in modes:
            raise ValueError(f"duplicate Fourier mode {k}")
        modes[k] = tuple(
            QComplex(_q(entry["real"][j]), _q(entry["imag"][j]))
            for j in range(3)
        )  # type: ignore[assignment]

    exact: SparseQField = {}
    for k in sorted(modes):
        if not is_positive_half_wave(k):
            continue
        opposite = negate_wave(k)
        if opposite not in modes:
            raise ValueError(f"mode {k} is missing its conjugate partner")
        averaged = tuple(
            (modes[k][j] + modes[opposite][j].conjugate()) / 2
            for j in range(3)
        )  # type: ignore[assignment]
        projected = leray_project(k, averaged)
        exact[k] = projected
        exact[opposite] = tuple(
            component.conjugate() for component in projected
        )  # type: ignore[assignment]

    datum = ExactDatum(
        field=exact,
        source_path=str(source),
        source_sha256=sha256(raw_bytes).hexdigest(),
    )
    datum.validate()
    return datum


@dataclass(frozen=True)
class OnsetAlgebra:
    """Exact coefficients in ``Q(alpha)``, where ``alpha^2=scale_squared``."""

    raw_enstrophy: Fraction
    scale_squared: Fraction
    eprime_constant: Fraction
    eprime_alpha: Fraction
    eprime2_constant: Fraction
    eprime2_alpha: Fraction
    nonlinear_support: int
    second_derivative_support: int


def derive_onset_algebra(
    datum: ExactDatum,
    *,
    viscosity: Fraction = Fraction(1, 100),
    target_enstrophy: Fraction = Fraction(100),
) -> OnsetAlgebra:
    """Derive exact algebraic expressions for ``E'(0)`` and ``E''(0)``."""

    raw = datum.field
    raw_e = enstrophy(raw)
    if raw_e <= 0:
        raise ValueError("initial enstrophy must be positive")
    scale_squared = target_enstrophy / raw_e

    nonlinear = bilinear_navier_stokes(raw, raw)
    viscous = viscous_field(raw, viscosity)

    # u = alpha*r, F(u) = s*N(r,r) + alpha*V(r), alpha^2=s.
    r_n = weighted_real_inner(raw, nonlinear, 1)
    r_v = weighted_real_inner(raw, viscous, 1)
    eprime_constant = scale_squared * r_v
    eprime_alpha = scale_squared * r_n

    # DF(u)F(u) = d0 + alpha*d1.
    d0 = field_scale(
        field_add(
            bilinear_navier_stokes(viscous, raw),
            bilinear_navier_stokes(raw, viscous),
            viscous_field(nonlinear, viscosity),
        ),
        scale_squared,
    )
    d1 = field_add(
        field_scale(
            field_add(
                bilinear_navier_stokes(nonlinear, raw),
                bilinear_navier_stokes(raw, nonlinear),
            ),
            scale_squared,
        ),
        viscous_field(viscous, viscosity),
    )

    nn = weighted_real_inner(nonlinear, nonlinear, 1)
    nv = weighted_real_inner(nonlinear, viscous, 1)
    vv = weighted_real_inner(viscous, viscous, 1)
    r_d0 = weighted_real_inner(raw, d0, 1)
    r_d1 = weighted_real_inner(raw, d1, 1)
    eprime2_constant = (
        scale_squared * scale_squared * nn
        + scale_squared * vv
        + scale_squared * r_d1
    )
    eprime2_alpha = 2 * scale_squared * nv + r_d0

    return OnsetAlgebra(
        raw_enstrophy=raw_e,
        scale_squared=scale_squared,
        eprime_constant=eprime_constant,
        eprime_alpha=eprime_alpha,
        eprime2_constant=eprime2_constant,
        eprime2_alpha=eprime2_alpha,
        nonlinear_support=len(field_add(nonlinear, viscous)),
        second_derivative_support=len(field_add(d0, d1)),
    )


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _arb_fraction(value: Fraction):
    from flint import arb

    return arb(value.numerator) / arb(value.denominator)


def evaluate_onset_arb(
    algebra: OnsetAlgebra,
    *,
    precision: int,
    target_enstrophy: Fraction = Fraction(100),
    delta: Fraction = Fraction(1, 10),
) -> dict[str, object]:
    """Rigourously enclose onset quantities with Arb at one precision."""

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires the pinned python-flint==0.9.0")
    from flint import arb, ctx

    previous_precision = ctx.prec
    try:
        ctx.prec = int(precision)
        alpha = _arb_fraction(algebra.scale_squared).sqrt()
        eprime = _arb_fraction(algebra.eprime_constant) + alpha * _arb_fraction(
            algebra.eprime_alpha
        )
        eprime2 = _arb_fraction(algebra.eprime2_constant) + alpha * _arb_fraction(
            algebra.eprime2_alpha
        )
        e = _arb_fraction(target_enstrophy)
        exponent = arb(2) + _arb_fraction(delta)
        q0 = eprime / (e**exponent)
        qprime = eprime2 / (e**exponent) - exponent * eprime * eprime / (
            e ** (exponent + 1)
        )
        digits = max(30, int(precision * 0.30103) - 8)
        return {
            "precision_bits": int(precision),
            "alpha": alpha,
            "enstrophy_rate": eprime,
            "enstrophy_second_derivative": eprime2,
            "q0": q0,
            "qprime0": qprime,
            "qprime_strictly_positive": bool(qprime > 0),
            "display_digits": digits,
        }
    finally:
        ctx.prec = previous_precision


def _serialize_arb_run(run: Mapping[str, object]) -> dict[str, object]:
    digits = int(run["display_digits"])
    serialized: dict[str, object] = {
        "precision_bits": run["precision_bits"],
        "qprime_strictly_positive": run["qprime_strictly_positive"],
    }
    for key in (
        "alpha",
        "enstrophy_rate",
        "enstrophy_second_derivative",
        "q0",
        "qprime0",
    ):
        serialized[f"{key}_ball"] = run[key].str(digits)  # type: ignore[attr-defined]
    return serialized


def _serialize_exact_field(field: Mapping[Wave, QVector]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for k in sorted(field):
        result.append(
            {
                "k": list(k),
                "real": [fraction_text(component.real) for component in field[k]],
                "imag": [fraction_text(component.imag) for component in field[k]],
            }
        )
    return result


def _python_flint_dependency_record() -> dict[str, object]:
    """Hash the installed distribution metadata and compiled Arb bindings."""

    distribution = importlib.metadata.distribution("python-flint")
    files = distribution.files or ()

    def digest(relative_path: str) -> dict[str, object]:
        path = Path(distribution.locate_file(relative_path))
        data = path.read_bytes()
        return {
            "path": relative_path.replace("\\", "/"),
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        }

    metadata_file = next(str(path) for path in files if str(path).endswith("METADATA"))
    record_file = next(str(path) for path in files if str(path).endswith("RECORD"))
    binary_files = sorted(str(path) for path in files if str(path).endswith((".pyd", ".so")))
    binaries = [digest(path) for path in binary_files]
    binary_manifest = "\n".join(
        f"{entry['path']}:{entry['size_bytes']}:{entry['sha256']}" for entry in binaries
    ).encode("ascii")
    primary = next(
        entry for entry in binaries if entry["path"].endswith("flint/pyflint.pyd")
    )
    return {
        "package": "python-flint",
        "version": distribution.version,
        "python": sys.version,
        "platform": platform.platform(),
        "distribution_metadata": digest(metadata_file),
        "distribution_record": digest(record_file),
        "primary_extension_binary": primary,
        "compiled_binary_count": len(binaries),
        "compiled_binary_manifest_sha256": sha256(binary_manifest).hexdigest(),
    }


def _validated_proof_precisions(precisions: Sequence[int]) -> tuple[int, ...]:
    """Require the two independent precision levels used by proof artifacts."""

    normalized = tuple(int(precision) for precision in precisions)
    required = {256, 512}
    if len(set(normalized)) < 2 or not required.issubset(normalized):
        raise ValueError(
            "proof-grade builders require distinct 256-bit and 512-bit runs"
        )
    return normalized


def _checker_source_record(script_name: str) -> dict[str, object]:
    """Hash this checker module and the entry-point script that invoked it."""

    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[2]
    script_path = repository_root / "scripts" / script_name

    def record(path: Path) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "path": str(path.relative_to(repository_root)).replace("\\", "/"),
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        }

    return {"module": record(module_path), "script": record(script_path)}


def build_onset_certificate(
    candidate_path: str | Path,
    *,
    precisions: Sequence[int] = (256, 512),
    viscosity: Fraction = Fraction(1, 100),
    target_enstrophy: Fraction = Fraction(100),
    delta: Fraction = Fraction(1, 10),
) -> dict[str, object]:
    """Construct the proof payload and rerun the Arb enclosure independently."""

    precisions = _validated_proof_precisions(precisions)
    datum = load_exactified_candidate(candidate_path)
    algebra = derive_onset_algebra(
        datum, viscosity=viscosity, target_enstrophy=target_enstrophy
    )
    runs = [
        evaluate_onset_arb(
            algebra,
            precision=precision,
            target_enstrophy=target_enstrophy,
            delta=delta,
        )
        for precision in precisions
    ]
    nested = all(
        runs[index]["q0"].contains(runs[index + 1]["q0"])
        and runs[index]["qprime0"].contains(runs[index + 1]["qprime0"])
        for index in range(len(runs) - 1)
    )
    positive = all(bool(run["qprime_strictly_positive"]) for run in runs)

    return {
        "schema_version": 1,
        "truth_label": "proof_grade_exact_onset_derivative",
        "source": {
            "path": datum.source_path,
            "sha256": datum.source_sha256,
            "decimal_policy": (
                "literal JSON decimals are rationals; each +/- pair is averaged, "
                "then rationally Leray projected and exactly conjugated"
            ),
        },
        "problem": {
            "domain": "[0,2*pi)^3",
            "viscosity": fraction_text(viscosity),
            "target_enstrophy": fraction_text(target_enstrophy),
            "delta": fraction_text(delta),
            "q_definition": "q=E'/E^(2+delta)",
        },
        "exact_data": {
            "modes": _serialize_exact_field(datum.field),
            "raw_enstrophy": fraction_text(algebra.raw_enstrophy),
            "scale_squared": fraction_text(algebra.scale_squared),
            "scale_definition": "alpha=sqrt(scale_squared)",
            "eprime": {
                "constant": fraction_text(algebra.eprime_constant),
                "alpha_coefficient": fraction_text(algebra.eprime_alpha),
            },
            "eprime2": {
                "constant": fraction_text(algebra.eprime2_constant),
                "alpha_coefficient": fraction_text(algebra.eprime2_alpha),
            },
            "rhs_mode_count": algebra.nonlinear_support,
            "second_derivative_mode_count": algebra.second_derivative_support,
        },
        "arb": {
            "dependency": _python_flint_dependency_record(),
            "runs": [_serialize_arb_run(run) for run in runs],
            "successive_balls_nested": nested,
        },
        "checker_source": _checker_source_record("certify_p3_onset.py"),
        "claims": {
            "qprime_at_zero_strictly_positive": positive,
            "strict_one_sided_local_bottleneck": positive,
            "reason": (
                "q'(0)>0 and smooth local Navier-Stokes evolution imply q(t)>q(0) "
                "on some non-explicit right neighborhood"
            ),
        },
        "limitations": {
            "explicit_positive_tau_certified": False,
            "h6_error_tube_certified": False,
            "finite_horizon_trajectory_certified": False,
            "note": (
                "This proves the exact onset derivative only. A residual-based "
                "PDE error tube is still required for a numerical value of tau."
            ),
        },
    }


def low_mode_pairs(max_wave_square: int = 4) -> tuple[Wave, ...]:
    radius = int(sqrt(max_wave_square))
    modes: list[Wave] = []
    for kx in range(-radius, radius + 1):
        for ky in range(-radius, radius + 1):
            for kz in range(-radius, radius + 1):
                k = (kx, ky, kz)
                if k != (0, 0, 0) and wave_square(k) <= max_wave_square:
                    if is_positive_half_wave(k):
                        modes.append(k)
    return tuple(sorted(modes))


def integer_polarizations(k: Wave) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return deterministic orthogonal integer vectors perpendicular to ``k``."""

    if k == (0, 0, 0):
        raise ValueError("the zero mode has no transverse polarizations")
    axis_index = min(range(3), key=lambda index: (abs(k[index]), index))
    axis = [0, 0, 0]
    axis[axis_index] = 1
    p1 = (
        k[1] * axis[2] - k[2] * axis[1],
        k[2] * axis[0] - k[0] * axis[2],
        k[0] * axis[1] - k[1] * axis[0],
    )
    p2 = (
        k[1] * p1[2] - k[2] * p1[1],
        k[2] * p1[0] - k[0] * p1[2],
        k[0] * p1[1] - k[1] * p1[0],
    )
    if sum(value * value for value in p1) == 0:
        raise ArithmeticError("failed to construct a transverse polarization")
    return p1, p2


def _basis_field(k: Wave, polarization: Sequence[int], imaginary: bool) -> SparseQField:
    factor = QComplex(Fraction(0), Fraction(1)) if imaginary else QComplex(1, 0)
    value = tuple(factor * int(component) for component in polarization)
    opposite = tuple(component.conjugate() for component in value)
    return {k: value, negate_wave(k): opposite}  # type: ignore[dict-item]


def _trilinear_basis_entry(
    tested: SparseQField, left: SparseQField, right: SparseQField
) -> Fraction:
    """Return ``<tested,B(left,right)>_E`` without constructing ``B``.

    Each basis field has only a conjugate pair.  Moreover, the Leray projector
    in ``B`` can be omitted inside this inner product because ``tested`` is
    already transverse.  Testing the eight possible signed triads directly is
    dramatically faster than allocating a sparse convolution for every tensor
    entry, while retaining exact rational arithmetic.
    """

    minus_i = QComplex(Fraction(0), Fraction(-1))
    total = QComplex()
    for tested_wave, tested_value in tested.items():
        for left_wave, left_value in left.items():
            for right_wave, right_value in right.items():
                if tuple(
                    left_wave[index] + right_wave[index] for index in range(3)
                ) != tested_wave:
                    continue
                directional = sum(
                    (right_wave[index] * left_value[index] for index in range(3)),
                    QComplex(),
                )
                coefficient = minus_i * directional
                dot = sum(
                    (
                        tested_value[index].conjugate()
                        * coefficient
                        * right_value[index]
                        for index in range(3)
                    ),
                    QComplex(),
                )
                total += wave_square(tested_wave) * dot
    if total.imag:
        raise ArithmeticError("a conjugate-paired trilinear form was not real")
    return total.real


def _third_derivative_entry(a: SparseQField, b: SparseQField, c: SparseQField) -> Fraction:
    return (
        _trilinear_basis_entry(a, b, c)
        + _trilinear_basis_entry(a, c, b)
        + _trilinear_basis_entry(b, a, c)
        + _trilinear_basis_entry(b, c, a)
        + _trilinear_basis_entry(c, a, b)
        + _trilinear_basis_entry(c, b, a)
    )


@dataclass
class StaticLowModePolynomial:
    """Exact integer-basis polynomial for ``R=A-2*nu*P`` in 64 variables."""

    viscosity: Fraction = Fraction(1, 100)
    pairs: tuple[Wave, ...] = low_mode_pairs(4)
    phase_waves: tuple[Wave, Wave, Wave] = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    phase_polarizations: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if any(wave not in self.pairs for wave in self.phase_waves):
            raise ValueError("every phase-fixing wave must belong to the low-mode basis")
        phase_matrix = np.asarray(self.phase_waves, dtype=float)
        if np.linalg.matrix_rank(phase_matrix) != 3:
            raise ValueError("phase-fixing waves must be linearly independent")
        if len(self.phase_polarizations) != 3 or any(
            polarization not in (0, 1) for polarization in self.phase_polarizations
        ):
            raise ValueError("phase polarizations must be three indices in {0,1}")

    @cached_property
    def basis(self) -> tuple[SparseQField, ...]:
        fields: list[SparseQField] = []
        for k in self.pairs:
            p1, p2 = integer_polarizations(k)
            fields.extend(
                (
                    _basis_field(k, p1, False),
                    _basis_field(k, p2, False),
                    _basis_field(k, p1, True),
                    _basis_field(k, p2, True),
                )
            )
        return tuple(fields)

    @property
    def dimension(self) -> int:
        return 4 * len(self.pairs)

    @cached_property
    def enstrophy_diagonal_exact(self) -> tuple[Fraction, ...]:
        return tuple(weighted_real_inner(field, field, 1) for field in self.basis)

    @cached_property
    def palinstrophy_diagonal_exact(self) -> tuple[Fraction, ...]:
        return tuple(weighted_real_inner(field, field, 2) for field in self.basis)

    @cached_property
    def third_derivative_terms(self) -> dict[tuple[int, int, int], Fraction]:
        terms: dict[tuple[int, int, int], Fraction] = {}
        for i, j, k in combinations_with_replacement(range(self.dimension), 3):
            value = _third_derivative_entry(self.basis[i], self.basis[j], self.basis[k])
            if value:
                terms[(i, j, k)] = value
        return terms

    @cached_property
    def third_derivative_float(self) -> np.ndarray:
        tensor = np.zeros((self.dimension, self.dimension, self.dimension))
        for indices, value in self.third_derivative_terms.items():
            for permuted in set(permutations(indices)):
                tensor[permuted] = float(value)
        return tensor

    @cached_property
    def enstrophy_diagonal(self) -> np.ndarray:
        return np.asarray([float(value) for value in self.enstrophy_diagonal_exact])

    @cached_property
    def palinstrophy_diagonal(self) -> np.ndarray:
        return np.asarray([float(value) for value in self.palinstrophy_diagonal_exact])

    @cached_property
    def phase_indices(self) -> tuple[int, int, int]:
        indices: list[int] = []
        for wave, polarization in zip(
            self.phase_waves, self.phase_polarizations, strict=True
        ):
            pair_index = self.pairs.index(wave)
            indices.append(4 * pair_index + 2 + polarization)
        return tuple(indices)  # type: ignore[return-value]

    def exact_field(self, controls: Sequence[Fraction]) -> SparseQField:
        if len(controls) != self.dimension:
            raise ValueError(f"expected {self.dimension} controls")
        fields = [
            field_scale(self.basis[index], _q(value))
            for index, value in enumerate(controls)
            if value
        ]
        return field_add(*fields) if fields else {}

    def exact_invariants(self, controls: Sequence[Fraction]) -> dict[str, Fraction]:
        field = self.exact_field(controls)
        nonlinear = bilinear_navier_stokes(field, field)
        e = enstrophy(field)
        p = palinstrophy(field)
        a = weighted_real_inner(field, nonlinear, 1)
        return {
            "enstrophy": e,
            "palinstrophy": p,
            "stretching": a,
            "rate": a - 2 * self.viscosity * p,
        }

    def evaluate(self, controls: Sequence[float]) -> dict[str, np.ndarray | float]:
        c = np.asarray(controls, dtype=float)
        if c.shape != (self.dimension,):
            raise ValueError(f"expected a vector of shape ({self.dimension},)")
        hessian_a = np.einsum("ijk,k->ij", self.third_derivative_float, c)
        gradient_a = 0.5 * hessian_a @ c
        a = float(c @ gradient_a / 3.0)
        e = float(0.5 * np.dot(self.enstrophy_diagonal * c, c))
        p = float(0.5 * np.dot(self.palinstrophy_diagonal * c, c))
        nu = float(self.viscosity)
        return {
            "enstrophy": e,
            "palinstrophy": p,
            "stretching": a,
            "rate": a - 2.0 * nu * p,
            "gradient_enstrophy": self.enstrophy_diagonal * c,
            "gradient_rate": gradient_a - 2.0 * nu * self.palinstrophy_diagonal * c,
            "hessian_enstrophy": np.diag(self.enstrophy_diagonal),
            "hessian_rate": hessian_a
            - 2.0 * nu * np.diag(self.palinstrophy_diagonal),
        }

    def constraints(self, controls: Sequence[float], target: float = 100.0) -> np.ndarray:
        c = np.asarray(controls, dtype=float)
        evaluated = self.evaluate(c)
        return np.asarray(
            [float(evaluated["enstrophy"]) - target]
            + [float(c[index]) for index in self.phase_indices]
        )

    def constraint_jacobian(self, controls: Sequence[float]) -> np.ndarray:
        c = np.asarray(controls, dtype=float)
        jacobian = np.zeros((4, self.dimension))
        jacobian[0] = self.enstrophy_diagonal * c
        for row, index in enumerate(self.phase_indices, start=1):
            jacobian[row, index] = 1.0
        return jacobian

    def kkt_system(self, point: Sequence[float], target: float = 100.0) -> np.ndarray:
        x = np.asarray(point, dtype=float)
        if x.shape != (self.dimension + 4,):
            raise ValueError(f"expected {self.dimension + 4} KKT variables")
        c, multipliers = x[: self.dimension], x[self.dimension :]
        evaluated = self.evaluate(c)
        jacobian = self.constraint_jacobian(c)
        stationarity = np.asarray(evaluated["gradient_rate"]) - jacobian.T @ multipliers
        return np.concatenate((stationarity, self.constraints(c, target)))

    def kkt_jacobian(self, point: Sequence[float]) -> np.ndarray:
        x = np.asarray(point, dtype=float)
        c, multipliers = x[: self.dimension], x[self.dimension :]
        evaluated = self.evaluate(c)
        constraint_jacobian = self.constraint_jacobian(c)
        hessian_lagrangian = np.asarray(evaluated["hessian_rate"]) - multipliers[
            0
        ] * np.asarray(evaluated["hessian_enstrophy"])
        result = np.zeros((self.dimension + 4, self.dimension + 4))
        result[: self.dimension, : self.dimension] = hessian_lagrangian
        result[: self.dimension, self.dimension :] = -constraint_jacobian.T
        result[self.dimension :, : self.dimension] = constraint_jacobian
        return result


def exact_controls_from_datum(
    model: StaticLowModePolynomial, datum: ExactDatum
) -> tuple[Fraction, ...]:
    """Encode a rational low-mode datum in the exact integer basis."""

    controls: list[Fraction] = []
    for k in model.pairs:
        coefficient = datum.field.get(k, _zero_vector())
        p1, p2 = integer_polarizations(k)
        norm1 = sum(value * value for value in p1)
        norm2 = sum(value * value for value in p2)
        z1 = sum((p1[j] * coefficient[j] for j in range(3)), QComplex()) / norm1
        z2 = sum((p2[j] * coefficient[j] for j in range(3)), QComplex()) / norm2
        controls.extend((z1.real, z2.real, z1.imag, z2.imag))
    reconstructed = model.exact_field(controls)
    if reconstructed != datum.field:
        raise ArithmeticError("integer-polarization encoding did not round-trip")
    return tuple(controls)


def phase_fix_controls(model: StaticLowModePolynomial, controls: Sequence[float]) -> np.ndarray:
    """Translate a control vector so the three selected amplitudes are real."""

    c = np.asarray(controls, dtype=float).copy()
    target_phases = np.zeros(3)
    for row, (wave, polarization) in enumerate(
        zip(model.phase_waves, model.phase_polarizations, strict=True)
    ):
        pair_index = model.pairs.index(wave)
        base = 4 * pair_index
        amplitude = complex(c[base + polarization], c[base + 2 + polarization])
        if abs(amplitude) < 1e-12:
            raise ValueError(f"cannot phase-fix vanishing selected amplitude at {wave}")
        target_phases[row] = -atan2(amplitude.imag, amplitude.real)
    translation = np.linalg.solve(
        np.asarray(model.phase_waves, dtype=float), target_phases
    )
    for pair_index, wave in enumerate(model.pairs):
        base = 4 * pair_index
        phase = sum(wave[index] * translation[index] for index in range(3))
        rotation = complex(cos(phase), sin(phase))
        z1 = complex(c[base], c[base + 2]) * rotation
        z2 = complex(c[base + 1], c[base + 3]) * rotation
        c[base : base + 4] = (z1.real, z2.real, z1.imag, z2.imag)
    return c


def optimize_static_rate(
    candidate_path: str | Path,
    *,
    max_iterations: int = 600,
    target_enstrophy: float = 100.0,
    phase_waves: tuple[Wave, Wave, Wave] | None = None,
    phase_polarizations: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, object]:
    """Generate, but do not certify, a static low-mode KKT candidate."""

    from scipy.linalg import null_space
    from scipy.optimize import minimize, root

    model = StaticLowModePolynomial(
        phase_waves=phase_waves
        if phase_waves is not None
        else ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        phase_polarizations=phase_polarizations,
    )
    datum = load_exactified_candidate(candidate_path)
    rational_controls = exact_controls_from_datum(model, datum)
    scale = sqrt(target_enstrophy / float(enstrophy(datum.field)))
    initial = phase_fix_controls(
        model, np.asarray([float(value) * scale for value in rational_controls])
    )

    def objective(c: np.ndarray) -> float:
        return -float(model.evaluate(c)["rate"])

    phase_index_set = set(model.phase_indices)
    free_indices = np.asarray(
        [index for index in range(model.dimension) if index not in phase_index_set]
    )

    def lift(free: np.ndarray) -> tuple[np.ndarray, float, float]:
        unscaled = np.zeros(model.dimension)
        unscaled[free_indices] = free
        quadratic = float(np.dot(model.enstrophy_diagonal * unscaled, unscaled))
        if not np.isfinite(quadratic) or quadratic <= 0:
            raise FloatingPointError("invalid normalized-sphere coordinate")
        scale_factor = sqrt(2.0 * target_enstrophy / quadratic)
        return scale_factor * unscaled, scale_factor, quadratic

    def manifold_objective(free: np.ndarray) -> float:
        controls, _, _ = lift(free)
        return objective(controls)

    def manifold_gradient(free: np.ndarray) -> np.ndarray:
        controls, scale_factor, quadratic = lift(free)
        unscaled = np.zeros(model.dimension)
        unscaled[free_indices] = free
        gradient = np.asarray(model.evaluate(controls)["gradient_rate"])
        pullback = scale_factor * (
            gradient
            - model.enstrophy_diagonal
            * unscaled
            * float(np.dot(gradient, unscaled))
            / quadratic
        )
        return -pullback[free_indices]

    minimized = minimize(
        manifold_objective,
        initial[free_indices],
        jac=manifold_gradient,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-15,
            "gtol": 1e-11,
            "maxls": 50,
        },
    )
    candidate, _, _ = lift(np.asarray(minimized.x))
    gradient = np.asarray(model.evaluate(candidate)["gradient_rate"])
    constraint_jacobian = model.constraint_jacobian(candidate)
    multipliers = np.linalg.lstsq(constraint_jacobian.T, gradient, rcond=None)[0]
    kkt_initial = np.concatenate((candidate, multipliers))
    polished = root(
        lambda point: model.kkt_system(point, target_enstrophy),
        kkt_initial,
        jac=model.kkt_jacobian,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 2000},
    )
    polished_point = np.asarray(polished.x)
    if (
        np.all(np.isfinite(polished_point))
        and np.linalg.norm(model.kkt_system(polished_point, target_enstrophy), ord=np.inf)
        < np.linalg.norm(model.kkt_system(kkt_initial, target_enstrophy), ord=np.inf)
    ):
        point = polished_point
    else:
        point = kkt_initial
    controls = point[: model.dimension]
    multipliers = point[model.dimension :]
    evaluated = model.evaluate(controls)
    residual = model.kkt_system(point, target_enstrophy)
    jacobian = model.constraint_jacobian(controls)
    tangent = null_space(jacobian)
    hessian_lagrangian = np.asarray(evaluated["hessian_rate"]) - multipliers[
        0
    ] * np.asarray(evaluated["hessian_enstrophy"])
    tangent_eigenvalues = np.linalg.eigvalsh(tangent.T @ hessian_lagrangian @ tangent)
    bordered = np.block(
        [
            [hessian_lagrangian, jacobian.T],
            [jacobian, np.zeros((4, 4))],
        ]
    )
    bordered_eigenvalues = np.linalg.eigvalsh(bordered)
    inertia_tolerance = 1e-8
    inertia = {
        "positive": int(np.sum(bordered_eigenvalues > inertia_tolerance)),
        "negative": int(np.sum(bordered_eigenvalues < -inertia_tolerance)),
        "near_zero": int(np.sum(np.abs(bordered_eigenvalues) <= inertia_tolerance)),
    }
    scaled_residual = float(
        np.max(np.abs(residual))
        / max(1.0, np.max(np.abs(np.asarray(evaluated["gradient_rate"]))))
    )
    selected_phase_amplitudes = np.asarray(
        [controls[index - 2] for index in model.phase_indices]
    )
    phase_translation_jacobian = selected_phase_amplitudes[:, None] * np.asarray(
        model.phase_waves, dtype=float
    )
    translation_chart_rank = int(
        np.linalg.matrix_rank(phase_translation_jacobian, tol=1e-8)
    )

    return {
        "schema_version": 1,
        "truth_label": "numerical_static_kkt_candidate_only",
        "source_sha256": datum.source_sha256,
        "model": {
            "dimension": model.dimension,
            "independent_mode_pairs": len(model.pairs),
            "mode_cutoff_squared": 4,
            "viscosity": fraction_text(model.viscosity),
            "target_enstrophy": target_enstrophy,
            "phase_indices": list(model.phase_indices),
            "phase_waves": [list(wave) for wave in model.phase_waves],
            "phase_polarizations": list(model.phase_polarizations),
            "basis": [
                {
                    "k": list(k),
                    "polarizations": [list(vector) for vector in integer_polarizations(k)],
                }
                for k in model.pairs
            ],
            "nonzero_exact_cubic_tensor_entries_canonical": len(
                model.third_derivative_terms
            ),
        },
        "optimizer": {
            "manifold_lbfgs_success": bool(minimized.success),
            "manifold_lbfgs_message": str(minimized.message),
            "manifold_lbfgs_iterations": int(minimized.nit),
            "root_success": bool(polished.success),
            "root_message": str(polished.message),
        },
        "candidate": {
            "controls": [float(value) for value in controls],
            "multipliers": [float(value) for value in multipliers],
            "enstrophy": float(evaluated["enstrophy"]),
            "palinstrophy": float(evaluated["palinstrophy"]),
            "stretching": float(evaluated["stretching"]),
            "rate": float(evaluated["rate"]),
            "max_abs_kkt_residual": float(np.max(np.abs(residual))),
            "scaled_kkt_residual": scaled_residual,
            "max_abs_constraint_residual": float(
                np.max(np.abs(model.constraints(controls, target_enstrophy)))
            ),
            "tangent_hessian_min_eigenvalue": float(tangent_eigenvalues[0]),
            "tangent_hessian_max_eigenvalue": float(tangent_eigenvalues[-1]),
            "numeric_bordered_hessian_inertia": inertia,
            "phase_chart": {
                "selected_real_amplitudes": [
                    float(value) for value in selected_phase_amplitudes
                ],
                "translation_jacobian_numeric_rank": translation_chart_rank,
                "translation_jacobian": phase_translation_jacobian.tolist(),
                "regular_chart": translation_chart_rank == 3,
                "explanation": (
                    "At a phase-fixed point, row i is Re(z_i) times the "
                    "selected wave vector."
                ),
            },
        },
        "proof_status": {
            "interval_krawczyk_verified": False,
            "interval_bordered_hessian_inertia_verified": False,
            "strict_local_maximum_proved": False,
            "blockers": [
                "The 68-dimensional KKT root has only a floating-point enclosure candidate.",
                "No interval Krawczyk inclusion has been established.",
                "The reported Hessian inertia is a double-precision diagnostic, not a proof.",
                (
                    "The selected phase chart has numeric rank "
                    f"{translation_chart_rank}, so its KKT Jacobian retains "
                    f"{3 - translation_chart_rank} translation zero mode(s)."
                ),
            ],
        },
    }


ADAPTIVE_PHASE_WAVES: tuple[Wave, Wave, Wave] = (
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 1),
)


def _static_arb_components(
    model: StaticLowModePolynomial,
    point: Sequence[object],
    target_enstrophy: Fraction = Fraction(100),
):
    """Evaluate the exact polynomial KKT system and two bordered matrices.

    ``point`` may contain Arb points or balls.  Every polynomial coefficient is
    converted directly from its exact ``Fraction`` representation.
    """

    from flint import arb, arb_mat

    dimension = model.dimension
    if len(point) != dimension + 4:
        raise ValueError(f"expected {dimension + 4} KKT variables")
    controls = list(point[:dimension])
    multipliers = list(point[dimension:])

    hessian_a = [[arb(0) for _ in range(dimension)] for _ in range(dimension)]
    for indices, exact_value in model.third_derivative_terms.items():
        value = _arb_fraction(exact_value)
        for i, j, k in set(permutations(indices)):
            hessian_a[i][j] += value * controls[k]

    gradient_a = [
        sum((hessian_a[i][j] * controls[j] for j in range(dimension)), arb(0))
        / 2
        for i in range(dimension)
    ]
    viscosity = _arb_fraction(model.viscosity)
    e_diagonal = [_arb_fraction(value) for value in model.enstrophy_diagonal_exact]
    p_diagonal = [_arb_fraction(value) for value in model.palinstrophy_diagonal_exact]
    gradient_rate = [
        gradient_a[i] - 2 * viscosity * p_diagonal[i] * controls[i]
        for i in range(dimension)
    ]
    hessian_rate = [[hessian_a[i][j] for j in range(dimension)] for i in range(dimension)]
    for i in range(dimension):
        hessian_rate[i][i] -= 2 * viscosity * p_diagonal[i]

    constraint_jacobian = [
        [e_diagonal[i] * controls[i] for i in range(dimension)]
    ] + [[arb(0) for _ in range(dimension)] for _ in range(3)]
    for row, index in enumerate(model.phase_indices, start=1):
        constraint_jacobian[row][index] = arb(1)

    stationarity = []
    for i in range(dimension):
        correction = sum(
            (
                multipliers[row] * constraint_jacobian[row][i]
                for row in range(4)
            ),
            arb(0),
        )
        stationarity.append(gradient_rate[i] - correction)
    enstrophy_value = sum(
        (e_diagonal[i] * controls[i] * controls[i] for i in range(dimension)),
        arb(0),
    ) / 2
    constraints = [enstrophy_value - _arb_fraction(target_enstrophy)] + [
        controls[index] for index in model.phase_indices
    ]
    residual = stationarity + constraints

    hessian_lagrangian = [
        [hessian_rate[i][j] for j in range(dimension)] for i in range(dimension)
    ]
    for i in range(dimension):
        hessian_lagrangian[i][i] -= multipliers[0] * e_diagonal[i]

    size = dimension + 4
    kkt_jacobian = [[arb(0) for _ in range(size)] for _ in range(size)]
    bordered_hessian = [[arb(0) for _ in range(size)] for _ in range(size)]
    for i in range(dimension):
        for j in range(dimension):
            kkt_jacobian[i][j] = hessian_lagrangian[i][j]
            bordered_hessian[i][j] = hessian_lagrangian[i][j]
        for row in range(4):
            value = constraint_jacobian[row][i]
            kkt_jacobian[i][dimension + row] = -value
            kkt_jacobian[dimension + row][i] = value
            bordered_hessian[i][dimension + row] = value
            bordered_hessian[dimension + row][i] = value
    return residual, arb_mat(kkt_jacobian), arb_mat(bordered_hessian)


def _arb_vector(values: Sequence[object]):
    from flint import arb_mat

    return arb_mat([[value] for value in values])


def _refine_static_root_arb(
    model: StaticLowModePolynomial,
    initial_point: Sequence[float],
    *,
    iterations: int = 6,
) -> tuple[list[object], list[object], object]:
    """Newton-refine a numerical KKT point, midpointing after every solve."""

    from flint import arb

    point = [arb(repr(float(value))).mid() for value in initial_point]
    last_correction = arb(0)
    for _ in range(iterations):
        residual, jacobian, _ = _static_arb_components(model, point)
        correction = jacobian.solve(_arb_vector(residual))
        last_correction = max(
            (correction[index, 0].abs_upper() for index in range(len(point))),
            key=float,
        )
        point = [
            (point[index] - correction[index, 0]).mid()
            for index in range(len(point))
        ]
    final_residual, _, _ = _static_arb_components(model, point)
    return point, final_residual, last_correction


def _identity_arb(size: int):
    from flint import arb, arb_mat

    entries = [[arb(0) for _ in range(size)] for _ in range(size)]
    for index in range(size):
        entries[index][index] = arb(1)
    return arb_mat(entries)


def _krawczyk_static_box(
    model: StaticLowModePolynomial,
    center: Sequence[object],
    *,
    radius_exponents: Sequence[int] = (30, 40, 50, 60, 80),
) -> dict[str, object]:
    """Find a componentwise box satisfying a rigorous Krawczyk inclusion."""

    from flint import arb, arb_mat

    size = model.dimension + 4
    center_residual, center_jacobian, _ = _static_arb_components(model, center)
    inverse_enclosure = center_jacobian.inv()
    preconditioner = arb_mat(
        [
            [inverse_enclosure[i, j].mid() for j in range(size)]
            for i in range(size)
        ]
    )
    preconditioner_determinant = preconditioner.det()
    preconditioner_nonsingular = not preconditioner_determinant.contains(0)
    center_vector = _arb_vector(center)
    newton_center = center_vector - preconditioner * _arb_vector(center_residual)
    identity = _identity_arb(size)

    attempts: list[dict[str, object]] = []
    for exponent in radius_exponents:
        base_radius = arb(2) ** (-int(exponent))
        radii = []
        box = []
        delta = []
        for value in center:
            magnitude = abs(value)
            scale = magnitude if magnitude > 1 else arb(1)
            radius = (base_radius * scale).upper()
            radii.append(radius)
            box_value = arb(value, radius)
            box.append(box_value)
            # Form this by subtraction so it certainly encloses the exact
            # ``X-center`` represented by ``box_value``, including constructor
            # rounding in its radius.
            delta.append(box_value - value)
        _, interval_jacobian, bordered = _static_arb_components(model, box)
        defect = identity - preconditioner * interval_jacobian
        image = newton_center + defect * _arb_vector(delta)
        inclusions = [
            box[index].contains_interior(image[index, 0]) for index in range(size)
        ]
        ratios = [
            float(image[index, 0].rad() / box[index].rad())
            for index in range(size)
        ]
        # This weighted infinity norm is an interval-certified Lipschitz bound
        # for g(z)=z-YF(z) on X.  The weights enclose X-center componentwise.
        weights = [value.abs_upper() for value in delta]
        contraction_rows = [
            sum(
                (
                    defect[i, j].abs_upper() * weights[j]
                    for j in range(size)
                ),
                arb(0),
            )
            / weights[i]
            for i in range(size)
        ]
        contraction_strict = all(value < 1 for value in contraction_rows)
        contraction_upper = max(contraction_rows, key=float)
        theorem_hypotheses = (
            all(inclusions) and preconditioner_nonsingular and contraction_strict
        )
        attempt = {
            "radius_power_of_two": -int(exponent),
            "all_components_strictly_interior": all(inclusions),
            "failed_component_count": inclusions.count(False),
            "max_image_to_box_radius_ratio": max(ratios),
            "preconditioner_nonsingular": preconditioner_nonsingular,
            "weighted_infinity_contraction_strictly_below_one": contraction_strict,
            "weighted_infinity_contraction_upper_float": float(contraction_upper),
            "all_existence_and_uniqueness_hypotheses_verified": theorem_hypotheses,
        }
        attempts.append(attempt)
        if theorem_hypotheses:
            return {
                "verified": True,
                "existence_verified": True,
                "uniqueness_verified": True,
                "box": box,
                "image": [image[index, 0] for index in range(size)],
                "bordered_hessian": bordered,
                "preconditioner": preconditioner,
                "preconditioner_determinant": preconditioner_determinant,
                "preconditioner_nonsingular": preconditioner_nonsingular,
                "weighted_infinity_contraction_upper": contraction_upper,
                "attempts": attempts,
                "selected_attempt": attempt,
            }
    return {"verified": False, "attempts": attempts}


def _certify_symmetric_inertia_arb(matrix) -> dict[str, object]:
    """Certify inertia by a point congruence and signed Gershgorin clusters."""

    from flint import arb, arb_mat

    size = matrix.nrows()
    midpoint = np.asarray(
        [[float(matrix[i, j].mid()) for j in range(size)] for i in range(size)]
    )
    midpoint = 0.5 * (midpoint + midpoint.T)
    _, eigenvectors = np.linalg.eigh(midpoint)
    congruence = arb_mat(
        [
            [arb(repr(float(eigenvectors[i, j]))).mid() for j in range(size)]
            for i in range(size)
        ]
    )
    determinant = congruence.det()
    invertible = not determinant.contains(0)
    transformed = congruence.transpose() * matrix * congruence

    positive = 0
    negative = 0
    unresolved = 0
    margins: list[object] = []
    for i in range(size):
        off_diagonal = sum(
            (transformed[i, j].abs_upper() for j in range(size) if j != i),
            arb(0),
        )
        positive_margin = transformed[i, i].lower() - off_diagonal
        negative_margin = -transformed[i, i].upper() - off_diagonal
        if positive_margin > 0:
            positive += 1
            margins.append(positive_margin)
        elif negative_margin > 0:
            negative += 1
            margins.append(negative_margin)
        else:
            unresolved += 1
    minimum_margin = min(margins, key=float) if margins else arb(0)
    return {
        "verified": invertible and unresolved == 0,
        "congruence_invertible": invertible,
        "congruence_determinant": determinant,
        "positive": positive,
        "negative": negative,
        "zero_or_unresolved": unresolved,
        "minimum_signed_gershgorin_margin": minimum_margin,
    }


def _exact_tensor_sha256(model: StaticLowModePolynomial) -> str:
    lines = [
        f"{i},{j},{k}:{fraction_text(value)}"
        for (i, j, k), value in sorted(model.third_derivative_terms.items())
    ]
    return sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def evaluate_adaptive_static_certificate_arb(
    numerical_candidate: Mapping[str, object],
    *,
    precision: int,
) -> dict[str, object]:
    """Run one independent adaptive-chart Krawczyk and inertia proof."""

    if importlib.metadata.version("python-flint") != "0.9.0":
        raise RuntimeError("proof evaluation requires the pinned python-flint==0.9.0")
    from flint import ctx

    phase_waves = tuple(
        tuple(int(value) for value in wave)
        for wave in numerical_candidate["model"]["phase_waves"]  # type: ignore[index]
    )
    if phase_waves != ADAPTIVE_PHASE_WAVES:
        raise ValueError("candidate does not use the declared adaptive active-mode chart")
    model = StaticLowModePolynomial(phase_waves=ADAPTIVE_PHASE_WAVES)
    candidate = numerical_candidate["candidate"]  # type: ignore[index]
    initial = [
        float(value)
        for value in candidate["controls"] + candidate["multipliers"]  # type: ignore[index,operator]
    ]
    previous_precision = ctx.prec
    try:
        ctx.prec = int(precision)
        center, residual, last_correction = _refine_static_root_arb(model, initial)
        krawczyk = _krawczyk_static_box(model, center)
        if not krawczyk["verified"]:
            return {
                "precision_bits": int(precision),
                "krawczyk_verified": False,
                "attempts": krawczyk["attempts"],
            }
        inertia = _certify_symmetric_inertia_arb(krawczyk["bordered_hessian"])
        box = krawczyk["box"]
        selected_real_indices = [index - 2 for index in model.phase_indices]
        chart_regular = all(not box[index].contains(0) for index in selected_real_indices)
        digits = max(30, int(precision * 0.30103) - 8)
        max_residual = max((value.abs_upper() for value in residual), key=float)
        return {
            "precision_bits": int(precision),
            "krawczyk_verified": True,
            "existence_verified": krawczyk["existence_verified"],
            "uniqueness_verified": krawczyk["uniqueness_verified"],
            "krawczyk_attempts": krawczyk["attempts"],
            "selected_attempt": krawczyk["selected_attempt"],
            "preconditioner_nonsingular": krawczyk[
                "preconditioner_nonsingular"
            ],
            "preconditioner_determinant_ball": krawczyk[
                "preconditioner_determinant"
            ].str(digits),
            "weighted_infinity_contraction_upper_ball": krawczyk[
                "weighted_infinity_contraction_upper"
            ].str(digits),
            "interval_jacobian_regular": bool(
                krawczyk["preconditioner_nonsingular"]
                and krawczyk["selected_attempt"][
                    "weighted_infinity_contraction_strictly_below_one"
                ]
            ),
            "newton_last_correction_ball": last_correction.str(digits),
            "refined_residual_abs_upper_ball": max_residual.str(digits),
            "root_box_balls": [value.str(digits) for value in box],
            "adaptive_chart_regular": chart_regular,
            "selected_real_amplitude_balls": [
                box[index].str(digits) for index in selected_real_indices
            ],
            "bordered_inertia": {
                "verified": inertia["verified"],
                "congruence_invertible": inertia["congruence_invertible"],
                "congruence_determinant_ball": inertia[
                    "congruence_determinant"
                ].str(digits),
                "positive": inertia["positive"],
                "negative": inertia["negative"],
                "zero_or_unresolved": inertia["zero_or_unresolved"],
                "bordered_hessian_nonsingular": inertia["verified"],
                "constraint_jacobian_full_row_rank": inertia["verified"],
                "constraint_rank_argument": (
                    "If the 4x64 constraint Jacobian C lacked full row rank, "
                    "some nonzero mu would satisfy C^T mu=0, making (0,mu) a "
                    "null vector of the bordered Hessian. Its certified "
                    "nonsingularity therefore implies rank(C)=4."
                ),
                "minimum_signed_gershgorin_margin_ball": inertia[
                    "minimum_signed_gershgorin_margin"
                ].str(digits),
            },
        }
    finally:
        ctx.prec = previous_precision


def build_adaptive_static_certificate(
    numerical_candidate_path: str | Path,
    *,
    precisions: Sequence[int] = (256, 512),
) -> dict[str, object]:
    """Build a proof payload for the adaptive active-mode phase chart."""

    precisions = _validated_proof_precisions(precisions)
    path = Path(numerical_candidate_path)
    raw = path.read_bytes()
    numerical_candidate = json.loads(raw)
    model = StaticLowModePolynomial(phase_waves=ADAPTIVE_PHASE_WAVES)
    runs = [
        evaluate_adaptive_static_certificate_arb(
            numerical_candidate, precision=precision
        )
        for precision in precisions
    ]
    all_krawczyk = all(bool(run.get("krawczyk_verified")) for run in runs)
    all_preconditioners = all(
        bool(run.get("preconditioner_nonsingular")) for run in runs
    )
    all_interval_jacobians = all(
        bool(run.get("interval_jacobian_regular")) for run in runs
    )
    all_unique = all(
        bool(run.get("existence_verified"))
        and bool(run.get("uniqueness_verified"))
        for run in runs
    )
    all_inertia = all(
        bool(run.get("bordered_inertia", {}).get("verified"))
        and run["bordered_inertia"]["positive"] == 4  # type: ignore[index]
        and run["bordered_inertia"]["negative"] == 64  # type: ignore[index]
        and run["bordered_inertia"]["zero_or_unresolved"] == 0  # type: ignore[index]
        for run in runs
    )
    all_regular = all(bool(run.get("adaptive_chart_regular")) for run in runs)
    proved = (
        all_krawczyk
        and all_preconditioners
        and all_interval_jacobians
        and all_unique
        and all_inertia
        and all_regular
    )
    return {
        "schema_version": 1,
        "truth_label": (
            "proof_grade_adaptive_chart_strict_local_maximum"
            if proved
            else "adaptive_chart_interval_certificate_failed"
        ),
        "source": {
            "path": str(path),
            "sha256": sha256(raw).hexdigest(),
            "source_truth_label": numerical_candidate.get("truth_label"),
        },
        "exact_model": {
            "dimension": model.dimension,
            "constraint_count": 4,
            "phase_waves": [list(wave) for wave in ADAPTIVE_PHASE_WAVES],
            "phase_wave_matrix_determinant": 4,
            "nonzero_canonical_cubic_entries": len(model.third_derivative_terms),
            "exact_cubic_tensor_sha256": _exact_tensor_sha256(model),
            "viscosity": fraction_text(model.viscosity),
            "target_enstrophy": "100",
        },
        "proof_method": {
            "root": (
                "For g(z)=z-YF(z), Arb proves Y nonsingular, K(x,X) strictly "
                "inside X, and the weighted infinity norm of I-YJ(X) below "
                "one. Brouwer gives a fixed point, nonsingular Y turns it into "
                "F(z)=0, and the contraction bound proves uniqueness in X."
            ),
            "inertia": (
                "congruence by an exactly represented dyadic matrix; Arb proves "
                "that matrix invertible, then every transformed Gershgorin disc "
                "is separated from zero with 4 positive and 64 negative discs"
            ),
            "local_maximum": (
                "the bordered inertia and full-rank four-constraint Jacobian imply "
                "negative definiteness on the 60-dimensional tangent space"
            ),
        },
        "arb": {
            "dependency": _python_flint_dependency_record(),
            "runs": runs,
        },
        "checker_source": _checker_source_record("certify_static_adaptive.py"),
        "claims": {
            "unique_kkt_root_in_reported_box": all_krawczyk,
            "krawczyk_preconditioner_nonsingular": all_preconditioners,
            "interval_jacobian_regular_on_reported_box": all_interval_jacobians,
            "bordered_hessian_inertia_4_positive_64_negative": all_inertia,
            "adaptive_phase_chart_is_local_translation_slice": all_regular,
            "strict_local_maximum_modulo_translations_in_adaptive_chart": proved,
            "global_maximum": False,
        },
        "original_e_axis_chart_status": {
            "still_blocked": True,
            "reason": (
                "the numerical orbit collapses its selected e1/e2 amplitudes "
                "and retains two translation zero modes; no regular e-axis "
                "interval chart has been certified"
            ),
        },
        "limitations": {
            "finite_dimensional_low_mode_class_only": True,
            "global_optimality_not_proved": True,
            "navier_stokes_trajectory_statement": False,
        },
    }
