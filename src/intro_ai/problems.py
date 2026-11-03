"""Problem instances used in the coursework.

The delivery-robot graph is the textbook example from D. L. Poole and A. K. Mackworth,
*Artificial Intelligence: Foundations of Computational Agents* (Cambridge University
Press), chapter 3: a robot in an office building must travel from room ``o103`` to
room ``r123``. Node names follow the book (``mail``, ``ts``, ``storage``, ...); the arc
costs and heuristic values are the ones used in the course assignment.
"""

from __future__ import annotations

from .csp import CSP
from .search import Arc, GraphProblem

DELIVERY_ARCS: tuple[Arc, ...] = (
    Arc("ts", "mail", 6),
    Arc("o103", "ts", 8),
    Arc("o103", "b3", 4),
    Arc("o103", "o109", 12),
    Arc("o109", "o119", 16),
    Arc("o109", "o111", 4),
    Arc("b1", "c2", 3),
    Arc("b1", "b2", 6),
    Arc("b2", "b4", 3),
    Arc("b3", "b1", 4),
    Arc("b3", "b4", 7),
    Arc("b4", "o109", 7),
    Arc("c1", "c3", 8),
    Arc("c2", "c3", 6),
    Arc("c2", "c1", 4),
    Arc("o123", "o125", 4),
    Arc("o123", "r123", 4),
    Arc("o119", "o123", 9),
    Arc("o119", "storage", 7),
)

DELIVERY_HEURISTIC: dict[str, float] = {
    "mail": 26,
    "ts": 23,
    "o103": 21,
    "o109": 24,
    "o111": 27,
    "o119": 11,
    "o123": 4,
    "o125": 6,
    "r123": 0,
    "b1": 13,
    "b2": 15,
    "b3": 17,
    "b4": 18,
    "c1": 6,
    "c2": 10,
    "c3": 12,
    "storage": 12,
}

# Hand-placed layout, only used for drawing.
DELIVERY_POSITIONS: dict[str, tuple[float, float]] = {
    "mail": (-9, 5),
    "ts": (-9, 2),
    "o103": (-6, 0),
    "o109": (2, 0),
    "o111": (5, 0),
    "o119": (2, 5),
    "o123": (-2, 6),
    "o125": (-5, 6),
    "r123": (-2, 9),
    "storage": (5, 5),
    "b3": (-6, -3),
    "b4": (0, -3),
    "b1": (-4, -6),
    "b2": (-1, -6),
    "c2": (-4, -9),
    "c1": (-7, -11),
    "c3": (-1, -11),
}


def delivery_robot() -> GraphProblem:
    """The delivery-robot problem: from ``o103`` to ``r123``."""
    return GraphProblem(
        nodes=frozenset(DELIVERY_HEURISTIC),
        arcs=DELIVERY_ARCS,
        start="o103",
        goals=frozenset({"r123"}),
        heuristic=dict(DELIVERY_HEURISTIC),
    )


AUSTRALIA_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "WA": ("NT", "SA"),
    "NT": ("WA", "SA", "Q"),
    "SA": ("WA", "NT", "Q", "NSW", "V"),
    "Q": ("NT", "SA", "NSW"),
    "NSW": ("SA", "Q", "V"),
    "V": ("SA", "NSW"),
    "T": (),
}

AUSTRALIA_NAMES: dict[str, str] = {
    "WA": "Western Australia",
    "NT": "Northern Territory",
    "SA": "South Australia",
    "Q": "Queensland",
    "NSW": "New South Wales",
    "V": "Victoria",
    "T": "Tasmania",
}

# Rough geographic layout, only used for drawing.
AUSTRALIA_POSITIONS: dict[str, tuple[float, float]] = {
    "WA": (0.0, 2.0),
    "NT": (2.2, 3.3),
    "SA": (2.4, 1.3),
    "Q": (4.3, 3.0),
    "NSW": (4.6, 1.3),
    "V": (4.0, 0.2),
    "T": (4.3, -1.2),
}


def australia(colours: tuple[str, ...] = ("red", "green", "blue")) -> CSP:
    """Map colouring of the seven Australian states and territories."""
    return CSP.not_equal(
        variables=tuple(AUSTRALIA_NEIGHBOURS),
        domains={v: colours for v in AUSTRALIA_NEIGHBOURS},
        neighbours=AUSTRALIA_NEIGHBOURS,
    )
