import itertools

import pytest

from intro_ai.csp import CSP, random_graph_colouring, solve
from intro_ai.problems import AUSTRALIA_NEIGHBOURS, australia

CONFIGS = [
    {},
    {"mrv": True},
    {"forward_checking": True},
    {"mrv": True, "forward_checking": True},
]


def brute_force(csp: CSP) -> list[dict[str, str]]:
    out = []
    for values in itertools.product(*(csp.domains[v] for v in csp.variables)):
        a = dict(zip(csp.variables, values, strict=True))
        if csp.is_solution(a):
            out.append(a)
    return out


def key(solutions):
    return sorted(tuple(sorted(s.items())) for s in solutions)


@pytest.mark.parametrize("kw", CONFIGS)
def test_australia_has_18_valid_solutions(kw):
    csp = australia()
    result = solve(csp, **kw)
    # SA touches every other mainland region (3 choices); the ring WA-NT-Q-NSW-V must then
    # alternate the remaining two colours (2 choices); Tasmania is free (3 choices).
    assert len(result.solutions) == 18 == 3 * 2 * 3
    assert all(csp.is_solution(s) for s in result.solutions)
    assert key(result.solutions) == key(brute_force(csp))


@pytest.mark.parametrize("kw", CONFIGS)
def test_australia_two_colours_is_unsatisfiable(kw):
    result = solve(australia(("red", "green")), **kw)
    assert result.solutions == []
    assert result.backtracks > 0
    assert not result.exhausted


def test_first_solution_only():
    result = solve(australia(), all_solutions=False)
    assert len(result.solutions) == 1
    assert result.solutions[0] == {
        "WA": "red",
        "NT": "green",
        "SA": "blue",
        "Q": "red",
        "NSW": "green",
        "V": "red",
        "T": "red",
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kw", CONFIGS)
def test_small_random_graphs_match_brute_force(seed, kw):
    csp = random_graph_colouring(8, 3.0, 3, seed)
    assert key(solve(csp, **kw).solutions) == key(brute_force(csp))


def test_mrv_cuts_search_on_random_graphs():
    """On 3-colouring of G(40, p) MRV decides every instance far inside a budget plain
    backtracking exhausts."""
    plain = mrv = 0
    for seed in range(5):
        csp = random_graph_colouring(40, 4.0, 3, seed)
        plain += solve(csp, all_solutions=False, max_assignments=20_000).assignments
        r = solve(csp, all_solutions=False, mrv=True, forward_checking=True)
        assert not r.exhausted
        mrv += r.assignments
    assert mrv * 10 < plain


def test_budget_sets_exhausted():
    r = solve(australia(), max_assignments=3)
    assert r.exhausted
    assert r.assignments == 3


def test_rejects_asymmetric_constraints():
    with pytest.raises(ValueError, match="symmetric"):
        CSP.not_equal(("a", "b"), {"a": ("x",), "b": ("x",)}, {"a": ("b",), "b": ()})


def test_australia_adjacency_is_symmetric():
    for v, ns in AUSTRALIA_NEIGHBOURS.items():
        for n in ns:
            assert v in AUSTRALIA_NEIGHBOURS[n]
