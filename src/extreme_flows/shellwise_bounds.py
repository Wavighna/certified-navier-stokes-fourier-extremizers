"""Exact SOS certificates for shellwise pieces of the stretching cubic.

The full 64-control inviscid spectral-norm problem remains open.  This module
isolates a proved building block: the component with shell pattern ``(1,1,2)``
obeys a sharp-looking, but here deliberately only *upper-bound*, estimate.
All objects use the raw canonical Fourier coordinates of
:class:`StaticLowModePolynomial`, so their coefficients remain rational.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from pathlib import Path
from typing import Iterable

from .certify import StaticLowModePolynomial
from .inviscid_sos_target import exact_stretching_cubic_terms
from .static_symmetry import IDENTITY_GENERATOR, signed_coordinate_matrices, static_symmetry_matrix


Q = Fraction
Monomial = tuple[int, ...]


def shell_of_coordinate(model: StaticLowModePolynomial, index: int) -> int:
    """Return ``|k|^2`` for one canonical real coordinate."""

    wave = model.pairs[index // 4]
    return sum(component * component for component in wave)


def _shell_indices(model: StaticLowModePolynomial, shell: int) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(model.dimension)
        if shell_of_coordinate(model, index) == shell
    )


def _add(terms: dict[Monomial, Fraction], monomial: Iterable[int], value: Fraction) -> None:
    monomial = tuple(sorted(monomial))
    terms[monomial] = terms.get(monomial, Q(0)) + value
    if not terms[monomial]:
        del terms[monomial]


def shell112_quartic_target(model: StaticLowModePolynomial | None = None) -> dict[Monomial, Fraction]:
    """Return the rational quartic target for the ``112`` shell estimate.

    Let ``A112`` be the part of stretching using shells ``(1,1,2)`` and let
    ``E_j`` denote enstrophy carried by shell ``j``.  If
    ``A112=sum_j M_j(y) z_j`` with ``y`` shell-one coordinates and ``z``
    shell-two coordinates, Cauchy reduces the desired estimate to positivity
    of the returned polynomial

    ``(4/3) E_1(y)^2 - 2 sum_j M_j(y)^2/d_j``.

    Here ``d_j`` is the exact diagonal coefficient in ``2E``.  Positivity
    therefore proves ``A112^2 <= (4/3) E1^2 E2``.
    """

    model = StaticLowModePolynomial() if model is None else model
    shell_one = _shell_indices(model, 1)
    shell_two = _shell_indices(model, 2)
    one_position = {index: position for position, index in enumerate(shell_one)}
    two_position = {index: position for position, index in enumerate(shell_two)}
    components: list[dict[tuple[int, int], Fraction]] = [defaultdict(Fraction) for _ in shell_two]
    for monomial, coefficient in exact_stretching_cubic_terms(model).items():
        if tuple(sorted(shell_of_coordinate(model, index) for index in monomial)) != (1, 1, 2):
            continue
        one_indices = tuple(sorted(one_position[index] for index in monomial if index in one_position))
        two_index = next(index for index in monomial if index in two_position)
        components[two_position[two_index]][one_indices] += coefficient

    target: dict[Monomial, Fraction] = {}
    for first, global_first in enumerate(shell_one):
        for second, global_second in enumerate(shell_one):
            _add(
                target,
                (first, first, second, second),
                Q(1, 3)
                * model.enstrophy_diagonal_exact[global_first]
                * model.enstrophy_diagonal_exact[global_second],
            )
    for position, component in enumerate(components):
        divisor = model.enstrophy_diagonal_exact[shell_two[position]]
        for left, left_coefficient in component.items():
            for right, right_coefficient in component.items():
                _add(target, left + right, -Q(2) * left_coefficient * right_coefficient / divisor)
    return target


def shell224_quartic_target(model: StaticLowModePolynomial | None = None) -> dict[Monomial, Fraction]:
    """Return the rational quartic target for the ``224`` shell estimate.

    Here ``A224=sum_j M_j(y) z_j`` with ``y`` on the shell ``|k|^2=2`` and
    ``z`` on shell four. Positivity of the returned polynomial proves
    ``A224^2 <= E2^2 E4``.
    """

    model = StaticLowModePolynomial() if model is None else model
    shell_two = _shell_indices(model, 2)
    shell_four = _shell_indices(model, 4)
    two_position = {index: position for position, index in enumerate(shell_two)}
    four_position = {index: position for position, index in enumerate(shell_four)}
    components: list[dict[tuple[int, int], Fraction]] = [defaultdict(Fraction) for _ in shell_four]
    for monomial, coefficient in exact_stretching_cubic_terms(model).items():
        if tuple(sorted(shell_of_coordinate(model, index) for index in monomial)) != (2, 2, 4):
            continue
        two_indices = tuple(sorted(two_position[index] for index in monomial if index in two_position))
        four_index = next(index for index in monomial if index in four_position)
        components[four_position[four_index]][two_indices] += coefficient

    target: dict[Monomial, Fraction] = {}
    for first, global_first in enumerate(shell_two):
        for second, global_second in enumerate(shell_two):
            _add(
                target,
                (first, first, second, second),
                Q(1, 4)
                * model.enstrophy_diagonal_exact[global_first]
                * model.enstrophy_diagonal_exact[global_second],
            )
    for position, component in enumerate(components):
        divisor = model.enstrophy_diagonal_exact[shell_four[position]]
        for left, left_coefficient in component.items():
            for right, right_coefficient in component.items():
                _add(target, left + right, -Q(2) * left_coefficient * right_coefficient / divisor)
    return target


def _degree_two_monomials() -> tuple[tuple[int, int], ...]:
    return tuple(combinations_with_replacement(range(12), 2))


def _signed_orbits(shell: int = 1) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return signed Gram-entry orbits under cube isometries/translations.

    Some entry orbits acquire sign ``-1`` under a stabilizer and are forced
    to vanish.  The returned root/sign data encode precisely this condition.
    """

    model = StaticLowModePolynomial()
    coordinates = _shell_indices(model, shell)
    position = {index: item for item, index in enumerate(coordinates)}
    monomials = tuple(combinations_with_replacement(range(len(coordinates)), 2))
    monomial_position = {monomial: item for item, monomial in enumerate(monomials)}

    def induced_action(generator):
        destination = []
        sign = []
        for column in coordinates:
            rows = [row for row in coordinates if generator[row][column]]
            if len(rows) != 1:
                raise ArithmeticError("shell-one action is not signed-permutational")
            destination.append(position[rows[0]])
            sign.append(int(generator[rows[0]][column]))
        inverse = [0] * len(coordinates)
        output_sign = [0] * len(coordinates)
        for source, target in enumerate(destination):
            inverse[target] = source
            output_sign[target] = sign[source]
        return (
            tuple(
                monomial_position[tuple(sorted((inverse[first], inverse[second])))]
                for first, second in monomials
            ),
            tuple(output_sign[first] * output_sign[second] for first, second in monomials),
        )

    actions = [
        induced_action(static_symmetry_matrix((IDENTITY_GENERATOR[0], shift)))
        for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    actions.extend(
        induced_action(static_symmetry_matrix((matrix, (0, 0, 0))))
        for matrix in signed_coordinate_matrices()
    )

    size = len(monomials)
    entries = tuple((first, second) for first in range(size) for second in range(first, size))
    entry_position = {entry: item for item, entry in enumerate(entries)}
    parent = list(range(len(entries)))
    sign_to_parent = [1] * len(entries)
    forced_zero = [False] * len(entries)

    def find(index: int) -> tuple[int, int]:
        if parent[index] == index:
            return index, 1
        direct_parent = parent[index]
        root, inherited_sign = find(direct_parent)
        sign_to_parent[index] *= inherited_sign
        parent[index] = root
        return root, sign_to_parent[index]

    def union(left: int, right: int, relation: int) -> None:
        # ``value[left] = relation * value[right]``.
        left_root, left_sign = find(left)
        right_root, right_sign = find(right)
        if left_root == right_root:
            if left_sign != relation * right_sign:
                forced_zero[left_root] = True
            return
        parent[left_root] = right_root
        sign_to_parent[left_root] = relation * right_sign * left_sign
        forced_zero[right_root] = forced_zero[right_root] or forced_zero[left_root]

    for destination, signs in actions:
        for item, (first, second) in enumerate(entries):
            target = entry_position[tuple(sorted((destination[first], destination[second])))]
            # Q_{source,source'} = sign*Q_{output,output'}.
            union(target, item, signs[first] * signs[second])
    roots = []
    signs = []
    for item in range(len(entries)):
        root, sign = find(item)
        roots.append(root)
        signs.append(sign)
    return entries, tuple(roots), tuple(forced_zero), tuple(signs), tuple(sorted(set(roots)))


def _gram_from_signed_orbits(
    shell: int, orbit_values: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], ...]:
    """Expand a signed-orbit value table to its exact symmetric Gram matrix."""

    entries, roots, forced_zero, signs, all_roots = _signed_orbits(shell)
    free_roots = tuple(root for root in all_roots if not forced_zero[root])
    if len(orbit_values) != len(free_roots):
        raise ArithmeticError("signed-orbit value table has the wrong length")
    position = {root: item for item, root in enumerate(free_roots)}
    dimension = len(_shell_indices(StaticLowModePolynomial(), shell))
    monomial_count = len(tuple(combinations_with_replacement(range(dimension), 2)))
    gram = [[Q(0) for _ in range(monomial_count)] for _ in range(monomial_count)]
    for item, (first, second) in enumerate(entries):
        root = roots[item]
        if forced_zero[root]:
            continue
        value = signs[item] * orbit_values[position[root]]
        gram[first][second] = value
        gram[second][first] = value
    return tuple(tuple(row) for row in gram)


def shell112_sos_gram() -> tuple[tuple[Fraction, ...], ...]:
    """Return an exact 78-by-78 PSD Gram matrix for the shell-112 target."""

    entries, roots, forced_zero, signs, all_roots = _signed_orbits(1)
    free_roots = tuple(root for root in all_roots if not forced_zero[root])
    if len(free_roots) != 19:
        raise ArithmeticError("the expected 19 shell-112 Gram orbits changed")
    position = {root: item for item, root in enumerate(free_roots)}
    # This rational point lies in the 10-parameter affine Gram family solved
    # by exact coefficient matching.  Its semidefiniteness is independently
    # established by rational Schur-complement pivots below.
    orbit_values = (
        Q(4, 3), Q(-2, 3), Q(4, 3), Q(-2, 3), Q(1, 3), Q(-2, 3), Q(1, 3),
        Q(4), Q(5, 2), Q(0), Q(0), Q(4), Q(-5, 2), Q(3), Q(2), Q(1),
        Q(0), Q(0), Q(2),
    )
    if len(orbit_values) != len(free_roots):
        raise AssertionError("shell-112 orbit-value table mismatch")
    return _gram_from_signed_orbits(1, orbit_values)


def shell224_sos_gram() -> tuple[tuple[Fraction, ...], ...]:
    """Return a candidate exact Gram matrix for the ``224`` quartic target.

    The sparse integer table was discovered by the symmetry-reduced SDP in
    ``scripts/search_shell224_sos.py`` and is checked from scratch below.
    """

    orbit_values = tuple(
        Q(value)
        for value in (
            16, -32, 16, -32, 16, -32, 16, -32, 16, 0, -32, 128, 128,
            0, 0, 0, 0, 128, -128, 128, -128, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 64, 64, 64, 64,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        )
    )
    return _gram_from_signed_orbits(2, orbit_values)


def _gram_terms(
    gram: tuple[tuple[Fraction, ...], ...], coordinate_dimension: int = 12
) -> dict[Monomial, Fraction]:
    terms: dict[Monomial, Fraction] = {}
    monomials = tuple(combinations_with_replacement(range(coordinate_dimension), 2))
    for first, left in enumerate(monomials):
        for second in range(first, len(monomials)):
            factor = Q(1) if first == second else Q(2)
            _add(terms, left + monomials[second], factor * gram[first][second])
    return terms


def _psd_pivots(gram: tuple[tuple[Fraction, ...], ...]) -> tuple[Fraction, ...]:
    """Exact semidefinite Schur-complement test, returning positive pivots."""

    remaining = [list(row) for row in gram]
    pivots: list[Fraction] = []
    size = len(remaining)
    cursor = 0
    while cursor < size:
        diagonals = [remaining[index][index] for index in range(cursor, size)]
        if any(value < 0 for value in diagonals):
            raise ArithmeticError("Gram matrix has a negative exact diagonal pivot")
        relative = next((index for index, value in enumerate(diagonals) if value > 0), None)
        if relative is None:
            if any(
                remaining[row][column]
                for row in range(cursor, size)
                for column in range(cursor, size)
            ):
                raise ArithmeticError("zero-diagonal Schur complement is not zero")
            break
        pivot_index = cursor + relative
        if pivot_index != cursor:
            remaining[cursor], remaining[pivot_index] = remaining[pivot_index], remaining[cursor]
            for row in remaining:
                row[cursor], row[pivot_index] = row[pivot_index], row[cursor]
        pivot = remaining[cursor][cursor]
        pivots.append(pivot)
        for row in range(cursor + 1, size):
            for column in range(cursor + 1, size):
                remaining[row][column] -= remaining[row][cursor] * remaining[cursor][column] / pivot
        cursor += 1
    return tuple(pivots)


def shell112_sos_certificate() -> dict[str, object]:
    """Return an exact certificate payload for the shell-112 inequality."""

    target = shell112_quartic_target()
    gram = shell112_sos_gram()
    reconstructed = _gram_terms(gram)
    if reconstructed != target:
        raise ArithmeticError("the shell-112 Gram expansion does not equal its target")
    pivots = _psd_pivots(gram)
    encoded = "\n".join(
        ",".join(str(value) for value in row) for row in gram
    ).encode("ascii")
    return {
        "truth_label": "proved_exact_sos_shell112_enstrophy_production_bound",
        "statement": (
            "For the (1,1,2) shell component A112 of the exact 16-pair "
            "stretching cubic, A112^2 <= (4/3) E1^2 E2, hence "
            "|A112| <= 2/sqrt(3) E1 sqrt(E2)."
        ),
        "scope": "one shellwise component in the fixed 64-real-coordinate Fourier class",
        "proof": {
            "degree_two_monomial_basis_size": 78,
            "rational_gram_sha256": sha256(encoded).hexdigest(),
            "exact_quartic_monomial_count": len(target),
            "positive_schur_pivot_count": len(pivots),
            "zero_schur_remainder_dimension": 78 - len(pivots),
            "all_exact_pivots_positive": all(value > 0 for value in pivots),
            "coefficient_identity_verified": True,
            "gram_positive_semidefinite_verified": True,
        },
        "claims": {
            "shell112_bound_proved": True,
            "shell112_constant_sharp_proved": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }


def shell224_sos_certificate() -> dict[str, object]:
    """Return an exact certificate payload for the shell-224 inequality."""

    target = shell224_quartic_target()
    gram = shell224_sos_gram()
    reconstructed = _gram_terms(gram, coordinate_dimension=24)
    if reconstructed != target:
        raise ArithmeticError("the shell-224 Gram expansion does not equal its target")
    pivots = _psd_pivots(gram)
    encoded = "\n".join(
        ",".join(str(value) for value in row) for row in gram
    ).encode("ascii")
    return {
        "truth_label": "proved_exact_sos_shell224_enstrophy_production_bound",
        "statement": (
            "For the (2,2,4) shell component A224 of the exact 16-pair "
            "stretching cubic, A224^2 <= E2^2 E4, hence |A224| <= E2 sqrt(E4)."
        ),
        "scope": "one shellwise component in the fixed 64-real-coordinate Fourier class",
        "proof": {
            "degree_two_monomial_basis_size": 300,
            "rational_gram_sha256": sha256(encoded).hexdigest(),
            "exact_quartic_monomial_count": len(target),
            "positive_schur_pivot_count": len(pivots),
            "zero_schur_remainder_dimension": 300 - len(pivots),
            "all_exact_pivots_positive": all(value > 0 for value in pivots),
            "coefficient_identity_verified": True,
            "gram_positive_semidefinite_verified": True,
        },
        "claims": {
            "shell224_bound_proved": True,
            "shell224_constant_sharp_proved": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }


def _bilinear123_orbits() -> tuple[
    tuple[tuple[int, int], ...], tuple[int, ...], tuple[bool, ...], tuple[int, ...], tuple[int, ...]
]:
    """Signed Gram-entry orbits for products of shell-one and shell-two variables."""

    model = StaticLowModePolynomial()
    one, two = _shell_indices(model, 1), _shell_indices(model, 2)
    one_position = {index: item for item, index in enumerate(one)}
    two_position = {index: item for item, index in enumerate(two)}
    basis = tuple((left, right) for left in range(len(one)) for right in range(len(two)))
    basis_position = {item: index for index, item in enumerate(basis)}

    def action(generator):
        def shell_action(coordinates, positions):
            destination, signs = [], []
            for column in coordinates:
                rows = [row for row in coordinates if generator[row][column]]
                if len(rows) != 1:
                    raise ArithmeticError("shell-123 action is not signed-permutational")
                destination.append(positions[rows[0]])
                signs.append(int(generator[rows[0]][column]))
            return destination, signs
        one_destination, one_signs = shell_action(one, one_position)
        two_destination, two_signs = shell_action(two, two_position)
        return (
            tuple(basis_position[(one_destination[left], two_destination[right])] for left, right in basis),
            tuple(one_signs[left] * two_signs[right] for left, right in basis),
        )

    actions = [
        action(static_symmetry_matrix((IDENTITY_GENERATOR[0], shift)))
        for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    actions.extend(action(static_symmetry_matrix((matrix, (0, 0, 0)))) for matrix in signed_coordinate_matrices())
    entries = tuple((left, right) for left in range(len(basis)) for right in range(left, len(basis)))
    entry_position = {entry: item for item, entry in enumerate(entries)}
    parent, sign_to_parent, forced_zero = list(range(len(entries))), [1] * len(entries), [False] * len(entries)

    def find(index: int) -> tuple[int, int]:
        if parent[index] == index:
            return index, 1
        direct_parent = parent[index]
        root, inherited = find(direct_parent)
        sign_to_parent[index] *= inherited
        parent[index] = root
        return root, sign_to_parent[index]

    def union(left: int, right: int, relation: int) -> None:
        left_root, left_sign = find(left)
        right_root, right_sign = find(right)
        if left_root == right_root:
            if left_sign != relation * right_sign:
                forced_zero[left_root] = True
            return
        parent[left_root] = right_root
        sign_to_parent[left_root] = relation * right_sign * left_sign
        forced_zero[right_root] = forced_zero[right_root] or forced_zero[left_root]

    for destination, signs in actions:
        for item, (left, right) in enumerate(entries):
            union(entry_position[tuple(sorted((destination[left], destination[right])))], item, signs[left] * signs[right])
    roots, signs = [], []
    for item in range(len(entries)):
        root, sign = find(item)
        roots.append(root)
        signs.append(sign)
    return entries, tuple(roots), tuple(forced_zero), tuple(signs), tuple(sorted(set(roots)))


def shell123_quartic_target(model: StaticLowModePolynomial | None = None) -> dict[Monomial, Fraction]:
    """Return the exact bilinear-basis quartic target for ``A123^2<=16E1E2E3``."""

    model = StaticLowModePolynomial() if model is None else model
    one, two, three = (_shell_indices(model, shell) for shell in (1, 2, 3))
    one_position = {index: item for item, index in enumerate(one)}
    two_position = {index: item for item, index in enumerate(two)}
    three_position = {index: item for item, index in enumerate(three)}
    components: list[dict[tuple[int, int], Fraction]] = [defaultdict(Fraction) for _ in three]
    for monomial, coefficient in exact_stretching_cubic_terms(model).items():
        if tuple(sorted(shell_of_coordinate(model, index) for index in monomial)) != (1, 2, 3):
            continue
        left = next(one_position[index] for index in monomial if index in one_position)
        middle = next(two_position[index] for index in monomial if index in two_position)
        linear = next(index for index in monomial if index in three_position)
        components[three_position[linear]][(left, 12 + middle)] += coefficient
    target: dict[Monomial, Fraction] = {}
    for left, global_left in enumerate(one):
        for middle, global_middle in enumerate(two):
            # 16E1E2 = 4*d1*d2*(y_i*z_j)^2.
            _add(target, (left, 12 + middle, left, 12 + middle), Q(4) * model.enstrophy_diagonal_exact[global_left] * model.enstrophy_diagonal_exact[global_middle])
    for position, component in enumerate(components):
        divisor = model.enstrophy_diagonal_exact[three[position]]
        for left, left_value in component.items():
            for right, right_value in component.items():
                _add(target, left + right, -Q(2) * left_value * right_value / divisor)
    return target


def shell334_quartic_target(model: StaticLowModePolynomial | None = None) -> dict[Monomial, Fraction]:
    """Return the quartic target whose positivity gives ``A334^2<=(2/27)E3^2E4``."""

    model = StaticLowModePolynomial() if model is None else model
    three, four = _shell_indices(model, 3), _shell_indices(model, 4)
    three_position = {index: item for item, index in enumerate(three)}
    four_position = {index: item for item, index in enumerate(four)}
    components: list[dict[tuple[int, int], Fraction]] = [defaultdict(Fraction) for _ in four]
    for monomial, coefficient in exact_stretching_cubic_terms(model).items():
        if tuple(sorted(shell_of_coordinate(model, index) for index in monomial)) != (3, 3, 4):
            continue
        pair = tuple(sorted(three_position[index] for index in monomial if index in three_position))
        linear = next(index for index in monomial if index in four_position)
        components[four_position[linear]][pair] += coefficient
    target: dict[Monomial, Fraction] = {}
    for left, global_left in enumerate(three):
        for right, global_right in enumerate(three):
            _add(target, (left, left, right, right), Q(1, 54) * model.enstrophy_diagonal_exact[global_left] * model.enstrophy_diagonal_exact[global_right])
    for position, component in enumerate(components):
        divisor = model.enstrophy_diagonal_exact[four[position]]
        for left, left_value in component.items():
            for right, right_value in component.items():
                _add(target, left + right, -Q(2) * left_value * right_value / divisor)
    return target


def shell334_sos_gram() -> tuple[tuple[Fraction, ...], ...]:
    """Load the sparse rational 136-by-136 Gram candidate for the 334 target."""

    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "artifacts" / "proofs" / "shell334_rational_gram_candidate.json").read_text(encoding="utf-8"))
    if data["dimension"] != 136:
        raise ArithmeticError("shell-334 candidate dimension changed")
    gram = [[Q(0) for _ in range(136)] for _ in range(136)]
    for left, right, value in data["entries"]:
        gram[left][right] = gram[right][left] = Q(value)
    return tuple(tuple(row) for row in gram)


def shell334_sos_certificate() -> dict[str, object]:
    """Return the exact certificate payload for the shell-334 bound."""

    target = shell334_quartic_target()
    gram = shell334_sos_gram()
    if _gram_terms(gram, coordinate_dimension=16) != target:
        raise ArithmeticError("the shell-334 Gram expansion does not equal its target")
    pivots = _psd_pivots(gram)
    encoded = "\n".join(",".join(str(value) for value in row) for row in gram).encode("ascii")
    return {
        "truth_label": "proved_exact_sos_shell334_enstrophy_production_bound",
        "statement": "For the (3,3,4) shell component A334, A334^2 <= (2/27) E3^2 E4.",
        "scope": "one shellwise component in the fixed 64-real-coordinate Fourier class",
        "proof": {
            "degree_two_monomial_basis_size": 136,
            "nonzero_upper_triangular_gram_entries": 208,
            "rational_gram_sha256": sha256(encoded).hexdigest(),
            "exact_quartic_monomial_count": len(target),
            "positive_schur_pivot_count": len(pivots),
            "zero_schur_remainder_dimension": 136 - len(pivots),
            "all_exact_pivots_positive": all(value > 0 for value in pivots),
            "coefficient_identity_verified": True,
            "gram_positive_semidefinite_verified": True,
        },
        "claims": {
            "shell334_bound_proved": True,
            "shell334_constant_sharp_proved": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }


def shell123_sos_gram() -> tuple[tuple[Fraction, ...], ...]:
    """Return the exact 288-by-288 bilinear Gram matrix for the shell-123 bound."""

    entries, roots, forced_zero, signs, all_roots = _bilinear123_orbits()
    free_roots = tuple(root for root in all_roots if not forced_zero[root])
    orbit_values = tuple(Q(value) for value in (
        64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, Q(512, 9), Q(-32, 9), Q(-32, 3), Q(32, 3), Q(-32, 9),
        Q(16, 3), Q(16, 9), 0, Q(-16, 9), Q(320, 3), Q(64, 3), Q(16, 3),
        Q(-16, 3), Q(-16, 3), 0, Q(16, 3), 64, 0, 0, 0, 0, 0, 128, 0, 0,
        0, 0,
    ))
    if len(free_roots) != 50 or len(orbit_values) != 50:
        raise ArithmeticError("shell-123 signed-orbit structure changed")
    position = {root: item for item, root in enumerate(free_roots)}
    gram = [[Q(0) for _ in range(288)] for _ in range(288)]
    for item, (left, right) in enumerate(entries):
        root = roots[item]
        if forced_zero[root]:
            continue
        value = signs[item] * orbit_values[position[root]]
        gram[left][right] = value
        gram[right][left] = value
    return tuple(tuple(row) for row in gram)


def _bilinear_gram_terms(gram: tuple[tuple[Fraction, ...], ...]) -> dict[Monomial, Fraction]:
    """Expand a Gram matrix on the 12-by-24 bilinear monomial basis exactly."""

    basis = tuple((left, 12 + right) for left in range(12) for right in range(24))
    terms: dict[Monomial, Fraction] = {}
    for left, first in enumerate(basis):
        for right in range(left, len(basis)):
            factor = Q(1) if left == right else Q(2)
            _add(terms, first + basis[right], factor * gram[left][right])
    return terms


def shell123_sos_certificate() -> dict[str, object]:
    """Return the exact but nonsharp shell-123 SOS certificate."""

    target = shell123_quartic_target()
    gram = shell123_sos_gram()
    if _bilinear_gram_terms(gram) != target:
        raise ArithmeticError("the shell-123 Gram expansion does not equal its target")
    pivots = _psd_pivots(gram)
    encoded = "\n".join(",".join(str(value) for value in row) for row in gram).encode("ascii")
    return {
        "truth_label": "proved_exact_sos_shell123_enstrophy_production_bound",
        "statement": "For the (1,2,3) shell component A123, A123^2 <= 16 E1 E2 E3.",
        "scope": "one shellwise component in the fixed 64-real-coordinate Fourier class",
        "proof": {
            "bilinear_monomial_basis_size": 288,
            "rational_gram_sha256": sha256(encoded).hexdigest(),
            "exact_quartic_monomial_count": len(target),
            "positive_schur_pivot_count": len(pivots),
            "zero_schur_remainder_dimension": 288 - len(pivots),
            "all_exact_pivots_positive": all(value > 0 for value in pivots),
            "coefficient_identity_verified": True,
            "gram_positive_semidefinite_verified": True,
        },
        "claims": {
            "shell123_bound_proved": True,
            "shell123_constant_sharp_proved": False,
            "full_64d_inviscid_globality_proved": False,
        },
    }


__all__ = [
    "shell112_quartic_target",
    "shell112_sos_certificate",
    "shell112_sos_gram",
    "shell224_quartic_target",
    "shell224_sos_certificate",
    "shell224_sos_gram",
    "shell123_quartic_target",
    "shell123_sos_certificate",
    "shell123_sos_gram",
    "shell334_quartic_target",
    "shell334_sos_certificate",
    "shell334_sos_gram",
]
