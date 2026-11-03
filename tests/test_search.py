import math

import pytest

from intro_ai.problems import DELIVERY_POSITIONS, delivery_robot
from intro_ai.search import (
    STRATEGIES,
    Arc,
    GraphProblem,
    Path,
    check_heuristic,
    cost_to_goal,
    search,
)


@pytest.fixture
def delivery() -> GraphProblem:
    return delivery_robot()


def diamond(h: dict[str, float] | None = None) -> GraphProblem:
    """s -> a -> g costs 1 + 10, s -> b -> g costs 5 + 1: fewest arcs tie, cheapest is via b."""
    return GraphProblem(
        nodes=frozenset("sabg"),
        arcs=(Arc("s", "a", 1), Arc("s", "b", 5), Arc("a", "g", 10), Arc("b", "g", 1)),
        start="s",
        goals=frozenset("g"),
        heuristic=h or {},
    )


def is_valid_path(problem: GraphProblem, path: Path) -> bool:
    arcs = {(a.tail, a.head): a.cost for a in problem.arcs}
    steps = list(zip(path.nodes[:-1], path.nodes[1:], strict=True))
    return (
        path.nodes[0] == problem.start
        and problem.is_goal(path.end)
        and all(s in arcs for s in steps)
        and math.isclose(sum(arcs[s] for s in steps), path.cost)
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("mpp", [False, True])
def test_every_strategy_finds_a_valid_path(delivery, strategy, mpp):
    result = search(delivery, strategy, multiple_path_pruning=mpp)
    assert result.found
    assert is_valid_path(delivery, result.path)
    assert result.expansion_order[0] == "o103"
    assert result.expansion_order[-1] == "r123"
    assert result.expanded == len(result.expansion_order)


@pytest.mark.parametrize("strategy", ["ucs", "astar"])
@pytest.mark.parametrize("mpp", [False, True])
@pytest.mark.parametrize("tie_break", ["fifo", "lifo"])
def test_ucs_and_astar_are_optimal(delivery, strategy, mpp, tie_break):
    optimum = cost_to_goal(delivery)[delivery.start]
    result = search(delivery, strategy, multiple_path_pruning=mpp, tie_break=tie_break)
    assert result.cost == optimum == 41
    assert result.path.nodes == ("o103", "o109", "o119", "o123", "r123")


def test_bfs_returns_fewest_arcs():
    result = search(diamond(), "bfs")
    assert len(result.path.nodes) == 3
    assert result.path.nodes == ("s", "a", "g")  # first arc wins the tie
    assert search(diamond(), "ucs").path.nodes == ("s", "b", "g")


def test_dfs_follows_first_arc():
    assert search(diamond(), "dfs").path.nodes == ("s", "a", "g")


def test_greedy_is_not_optimal_but_astar_is():
    # A misleading-but-admissible heuristic: a looks closer than b.
    p = diamond({"s": 0, "a": 0, "b": 1, "g": 0})
    assert search(p, "greedy").cost == 11
    assert search(p, "astar").cost == 6


def test_delivery_heuristic_is_admissible_and_consistent(delivery):
    report = check_heuristic(delivery)
    assert report.admissible
    assert report.consistent
    assert report.true_cost["o103"] == 41
    assert report.true_cost["o109"] == 29
    assert math.isinf(report.true_cost["mail"])  # dead end


def test_checker_detects_bad_heuristics():
    over = check_heuristic(diamond({"s": 0, "a": 0, "b": 2, "g": 0}))
    assert not over.admissible
    assert over.overestimates == (("b", 2.0, 1.0),)
    # Admissible but inconsistent: h(s) = 6 > c(s, a) + h(a) = 1 + 0.
    inconsistent = check_heuristic(diamond({"s": 6, "a": 0, "b": 1, "g": 0}))
    assert inconsistent.admissible
    assert not inconsistent.consistent
    assert inconsistent.inconsistent_arcs == (("s", "a", 5.0),)


def test_heuristic_reduces_astar_effort(delivery):
    """The course heuristic saves work: A* expands fewer paths than lowest-cost-first."""
    for mpp in (False, True):
        ucs = search(delivery, "ucs", multiple_path_pruning=mpp)
        astar = search(delivery, "astar", multiple_path_pruning=mpp)
        assert astar.expanded < ucs.expanded


def test_zero_heuristic_turns_astar_into_ucs(delivery):
    """The original code's naming bug made A* run with h = 0, i.e. as lowest-cost-first."""
    blind = delivery.with_heuristic({})
    a, u = search(blind, "astar"), search(delivery, "ucs")
    assert (a.expanded, a.expansion_order, a.cost) == (u.expanded, u.expansion_order, u.cost)


def test_known_delivery_numbers(delivery):
    expanded = {s: search(delivery, s).expanded for s in STRATEGIES}
    assert expanded == {"dfs": 16, "bfs": 21, "ucs": 26, "greedy": 15, "astar": 16}
    assert search(delivery, "dfs").cost == 53
    # With last-in-first-out tie-breaking (the original's list sort), greedy is fooled.
    assert search(delivery, "greedy", tie_break="lifo").cost == 47


def test_multiple_path_pruning_never_expands_a_node_twice(delivery):
    for s in STRATEGIES:
        order = search(delivery, s, multiple_path_pruning=True).expansion_order
        assert len(order) == len(set(order))


def test_cycles_are_pruned():
    p = GraphProblem(
        nodes=frozenset("abg"),
        arcs=(Arc("a", "b"), Arc("b", "a"), Arc("b", "g")),
        start="a",
        goals=frozenset("g"),
    )
    for s in STRATEGIES:
        assert search(p, s).path.nodes == ("a", "b", "g")


def test_unreachable_goal_exhausts_frontier():
    p = GraphProblem(nodes=frozenset("abg"), arcs=(Arc("a", "b"),), start="a", goals=frozenset("g"))
    for s in STRATEGIES:
        r = search(p, s)
        assert not r.found
        assert math.isinf(r.cost)
        assert r.expanded == 2


def test_validation():
    with pytest.raises(ValueError, match="negative"):
        Arc("a", "b", -1)
    with pytest.raises(ValueError, match="unknown nodes"):
        GraphProblem(frozenset("a"), (Arc("a", "z"),), "a", frozenset("a"))
    with pytest.raises(ValueError, match="unknown strategy"):
        search(diamond(), "iddfs")
    with pytest.raises(ValueError, match="tie_break"):
        search(diamond(), "ucs", tie_break="random")


def test_positions_cover_every_node(delivery):
    assert set(DELIVERY_POSITIONS) == set(delivery.nodes)
