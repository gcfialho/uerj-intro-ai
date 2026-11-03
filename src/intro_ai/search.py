"""Graph search on an explicit, weighted, directed graph.

One generic frontier search drives every strategy; the strategies differ only in
which path leaves the frontier next:

* ``dfs``     - last in, first out
* ``bfs``     - first in, first out
* ``ucs``     - lowest path cost g(p) (lowest-cost-first / uniform-cost search)
* ``greedy``  - lowest heuristic value h(end of p) (greedy best-first)
* ``astar``   - lowest f(p) = g(p) + h(end of p)

Ties between equal priorities are broken first in, first out by default
(``tie_break="lifo"`` reverses that). The goal test is applied when a path is
*removed* from the frontier, which is what makes UCS and A* optimal.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

STRATEGIES = ("dfs", "bfs", "ucs", "greedy", "astar")


@dataclass(frozen=True)
class Arc:
    """A directed arc ``tail -> head`` with a non-negative cost."""

    tail: str
    head: str
    cost: float = 1.0

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"arc {self.tail}->{self.head} has negative cost {self.cost}")


@dataclass(frozen=True)
class GraphProblem:
    """A search problem on an explicit graph."""

    nodes: frozenset[str]
    arcs: tuple[Arc, ...]
    start: str
    goals: frozenset[str]
    heuristic: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = {a.tail for a in self.arcs} | {a.head for a in self.arcs} | {self.start}
        unknown |= set(self.goals) | set(self.heuristic)
        unknown -= self.nodes
        if unknown:
            raise ValueError(f"unknown nodes: {sorted(unknown)}")

    def neighbours(self, node: str) -> list[Arc]:
        """Outgoing arcs of ``node``, in declaration order."""
        return [a for a in self.arcs if a.tail == node]

    def h(self, node: str) -> float:
        """Heuristic estimate of the cost from ``node`` to a goal (0 if not given)."""
        return float(self.heuristic.get(node, 0.0))

    def is_goal(self, node: str) -> bool:
        return node in self.goals

    def with_heuristic(self, heuristic: Mapping[str, float]) -> GraphProblem:
        return GraphProblem(self.nodes, self.arcs, self.start, self.goals, dict(heuristic))


@dataclass(frozen=True)
class Path:
    """A path from the start node, stored as its node sequence and total cost."""

    nodes: tuple[str, ...]
    cost: float = 0.0

    @property
    def end(self) -> str:
        return self.nodes[-1]

    def extend(self, arc: Arc) -> Path:
        return Path((*self.nodes, arc.head), self.cost + arc.cost)

    def __str__(self) -> str:
        return " -> ".join(self.nodes)


@dataclass(frozen=True)
class SearchResult:
    strategy: str
    path: Path | None
    expanded: int
    expansion_order: tuple[str, ...]

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def cost(self) -> float:
        return self.path.cost if self.path is not None else math.inf


def search(
    problem: GraphProblem,
    strategy: str,
    *,
    multiple_path_pruning: bool = False,
    tie_break: str = "fifo",
    max_expansions: int = 100_000,
) -> SearchResult:
    """Run ``strategy`` on ``problem``.

    Cycle pruning is always on: a path is never extended to a node it already
    contains. With ``multiple_path_pruning`` a node is expanded at most once
    (the first path to reach it wins); with a consistent heuristic A* stays optimal.

    ``tie_break`` only affects the priority-queue strategies (ucs, greedy, astar).

    ``expanded`` counts the paths removed from the frontier and examined,
    including the one that reaches the goal.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if tie_break not in ("fifo", "lifo"):
        raise ValueError("tie_break must be 'fifo' or 'lifo'")

    counter = itertools.count(0, 1 if tie_break == "fifo" else -1)
    start = Path((problem.start,))

    if strategy in ("dfs", "bfs"):
        frontier: deque[Path] = deque([start])

        def push(p: Path) -> None:
            frontier.append(p)

        def pop() -> Path:
            return frontier.pop() if strategy == "dfs" else frontier.popleft()

        def empty() -> bool:
            return not frontier
    else:
        heap: list[tuple[float, int, Path]] = []

        def priority(p: Path) -> float:
            if strategy == "ucs":
                return p.cost
            if strategy == "greedy":
                return problem.h(p.end)
            return p.cost + problem.h(p.end)

        def push(p: Path) -> None:
            heapq.heappush(heap, (priority(p), next(counter), p))

        def pop() -> Path:
            return heapq.heappop(heap)[2]

        def empty() -> bool:
            return not heap

        push(start)

    expanded = 0
    order: list[str] = []
    closed: set[str] = set()
    while not empty():
        path = pop()
        if multiple_path_pruning:
            if path.end in closed:
                continue
            closed.add(path.end)
        expanded += 1
        order.append(path.end)
        if problem.is_goal(path.end):
            return SearchResult(strategy, path, expanded, tuple(order))
        if expanded >= max_expansions:
            break
        arcs = [a for a in problem.neighbours(path.end) if a.head not in path.nodes]
        # DFS pops from the end, so push in reverse to explore the first arc first.
        for arc in reversed(arcs) if strategy == "dfs" else arcs:
            push(path.extend(arc))
    return SearchResult(strategy, None, expanded, tuple(order))


def cost_to_goal(problem: GraphProblem) -> dict[str, float]:
    """Exact cheapest cost from every node to the nearest goal (Dijkstra on reversed arcs).

    Nodes that cannot reach a goal get ``math.inf``.
    """
    reverse: dict[str, list[Arc]] = {n: [] for n in problem.nodes}
    for arc in problem.arcs:
        reverse[arc.head].append(arc)
    dist = {n: math.inf for n in problem.nodes}
    heap: list[tuple[float, str]] = []
    for g in problem.goals:
        dist[g] = 0.0
        heap.append((0.0, g))
    heapq.heapify(heap)
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for arc in reverse[node]:
            nd = d + arc.cost
            if nd < dist[arc.tail]:
                dist[arc.tail] = nd
                heapq.heappush(heap, (nd, arc.tail))
    return dist


@dataclass(frozen=True)
class HeuristicReport:
    admissible: bool
    consistent: bool
    overestimates: tuple[tuple[str, float, float], ...]  # (node, h, h*)
    inconsistent_arcs: tuple[tuple[str, str, float], ...]  # (tail, head, violation)
    true_cost: Mapping[str, float]


def check_heuristic(problem: GraphProblem, tol: float = 1e-12) -> HeuristicReport:
    """Check admissibility (h <= h*) and consistency (h(u) <= c(u, v) + h(v), h(goal) = 0)."""
    h_star = cost_to_goal(problem)
    over = tuple(
        (n, problem.h(n), h_star[n])
        for n in sorted(problem.nodes)
        if problem.h(n) > h_star[n] + tol
    )
    bad_arcs = []
    for arc in problem.arcs:
        violation = problem.h(arc.tail) - (arc.cost + problem.h(arc.head))
        if violation > tol:
            bad_arcs.append((arc.tail, arc.head, violation))
    goal_nonzero = any(problem.h(g) > tol for g in problem.goals)
    return HeuristicReport(
        admissible=not over,
        consistent=not bad_arcs and not goal_nonzero,
        overestimates=over,
        inconsistent_arcs=tuple(bad_arcs),
        true_cost=h_star,
    )


def run_all(
    problem: GraphProblem,
    strategies: Iterable[str] = STRATEGIES,
    *,
    multiple_path_pruning: bool = False,
) -> list[SearchResult]:
    return [search(problem, s, multiple_path_pruning=multiple_path_pruning) for s in strategies]
