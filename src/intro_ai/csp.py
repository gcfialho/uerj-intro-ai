"""Backtracking search for constraint satisfaction problems with binary not-equal constraints.

This covers graph/map colouring. Options:

* ``mrv`` - choose the unassigned variable with the fewest values still consistent
  with the current assignment (ties broken by declaration order) instead of a
  fixed order;
* ``forward_checking`` - after each assignment remove the value from the domains of
  unassigned neighbours and backtrack at once if one of them becomes empty.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CSP:
    variables: tuple[str, ...]
    domains: Mapping[str, tuple[str, ...]]
    neighbours: Mapping[str, tuple[str, ...]]

    @classmethod
    def not_equal(
        cls,
        variables: tuple[str, ...],
        domains: Mapping[str, tuple[str, ...]],
        neighbours: Mapping[str, tuple[str, ...]],
    ) -> CSP:
        """Build a CSP, checking that the neighbour relation is symmetric."""
        for v, ns in neighbours.items():
            for n in ns:
                if v not in neighbours.get(n, ()):
                    raise ValueError(f"constraint {v} != {n} is not symmetric")
        missing = set(variables) - set(domains)
        if missing:
            raise ValueError(f"variables without a domain: {sorted(missing)}")
        return cls(
            tuple(variables),
            {v: tuple(domains[v]) for v in variables},
            {v: tuple(neighbours.get(v, ())) for v in variables},
        )

    def is_solution(self, assignment: Mapping[str, str]) -> bool:
        if set(assignment) != set(self.variables):
            return False
        return all(assignment[v] in self.domains[v] for v in self.variables) and all(
            assignment[v] != assignment[n] for v in self.variables for n in self.neighbours[v]
        )


@dataclass
class SolveResult:
    solutions: list[dict[str, str]] = field(default_factory=list)
    assignments: int = 0  # values assigned to a variable (search-tree nodes)
    backtracks: int = 0  # assignments undone without leading to any solution
    checks: int = 0  # pairwise constraint checks
    exhausted: bool = False  # True if the assignment budget ran out before the search ended


def solve(
    csp: CSP,
    *,
    all_solutions: bool = True,
    mrv: bool = False,
    forward_checking: bool = False,
    max_assignments: int | None = None,
) -> SolveResult:
    """Depth-first backtracking. Returns every solution (or the first one) plus counters.

    With ``max_assignments`` the search stops once that many assignments were made and
    ``exhausted`` is set; an empty ``solutions`` list then proves nothing.
    """
    result = SolveResult()
    assignment: dict[str, str] = {}

    class _BudgetExhaustedError(Exception):
        pass

    def legal_count(var: str, domains: Mapping[str, tuple[str, ...]]) -> int:
        used = {assignment[n] for n in csp.neighbours[var] if n in assignment}
        return sum(1 for x in domains[var] if x not in used)

    def select(domains: Mapping[str, tuple[str, ...]]) -> str:
        unassigned = [v for v in csp.variables if v not in assignment]
        if mrv:
            return min(unassigned, key=lambda v: legal_count(v, domains))
        return unassigned[0]

    def consistent(var: str, value: str) -> bool:
        for n in csp.neighbours[var]:
            if n in assignment:
                result.checks += 1
                if assignment[n] == value:
                    return False
        return True

    def recurse(domains: Mapping[str, tuple[str, ...]]) -> int:
        if len(assignment) == len(csp.variables):
            result.solutions.append(dict(assignment))
            return 1
        var = select(domains)
        found = 0
        for value in domains[var]:
            if not consistent(var, value):
                continue
            if max_assignments is not None and result.assignments >= max_assignments:
                raise _BudgetExhaustedError
            assignment[var] = value
            result.assignments += 1
            sub = 0
            if forward_checking:
                pruned = dict(domains)
                pruned[var] = (value,)
                wiped_out = False
                for n in csp.neighbours[var]:
                    if n not in assignment:
                        result.checks += len(pruned[n])
                        pruned[n] = tuple(x for x in pruned[n] if x != value)
                        if not pruned[n]:
                            wiped_out = True
                            break
                if not wiped_out:
                    sub = recurse(pruned)
            else:
                sub = recurse(domains)
            del assignment[var]
            if sub == 0:
                result.backtracks += 1
            found += sub
            if found and not all_solutions:
                break
        return found

    try:
        recurse(dict(csp.domains))
    except _BudgetExhaustedError:
        result.exhausted = True
    return result


def random_graph_colouring(n: int, mean_degree: float, k: int, seed: int) -> CSP:
    """k-colouring of an Erdos-Renyi graph G(n, p) with p = mean_degree / (n - 1)."""
    rng = np.random.default_rng(seed)
    p = mean_degree / (n - 1)
    names = [f"v{i}" for i in range(n)]
    nbrs: dict[str, list[str]] = {v: [] for v in names}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                nbrs[names[i]].append(names[j])
                nbrs[names[j]].append(names[i])
    colours = tuple(str(c) for c in range(k))
    return CSP.not_equal(
        tuple(names), {v: colours for v in names}, {v: tuple(x) for v, x in nbrs.items()}
    )
