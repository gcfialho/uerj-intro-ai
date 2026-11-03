"""Command line: ``intro-ai search | csp | mlp | all``."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from . import problems
from .csp import random_graph_colouring, solve
from .search import STRATEGIES, check_heuristic, search

CSP_CONFIGS = {
    "plain": {},
    "mrv": {"mrv": True},
    "fc": {"forward_checking": True},
    "mrv+fc": {"mrv": True, "forward_checking": True},
}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _jsonable(x: float) -> float | None:
    return None if math.isinf(x) else x


def cmd_search(args: argparse.Namespace) -> dict:
    problem = problems.delivery_robot()
    report = check_heuristic(problem)
    print(f"Delivery robot: {problem.start} -> {sorted(problem.goals)}")
    print(
        f"heuristic admissible={report.admissible} consistent={report.consistent} "
        f"(h(start)={problem.h(problem.start):g}, h*(start)={report.true_cost[problem.start]:g})"
    )
    rows = []
    header = f"{'strategy':<8} {'pruning':<9} {'cost':>5} {'expanded':>9}  path"
    print(header)
    for pruning in (False, True):
        for s in STRATEGIES:
            r = search(problem, s, multiple_path_pruning=pruning, tie_break=args.tie_break)
            tag = "mpp" if pruning else "cycle"
            print(f"{s:<8} {tag:<9} {r.cost:>5g} {r.expanded:>9}  {r.path}")
            rows.append(
                {
                    "strategy": s,
                    "pruning": tag,
                    "cost": r.cost,
                    "expanded": r.expanded,
                    "path": list(r.path.nodes) if r.path else None,
                    "expansion_order": list(r.expansion_order),
                }
            )
    blind = problem.with_heuristic({})
    ablation = []
    print("with h = 0 (the original naming bug made both searches use h = 0):")
    for s in ("greedy", "astar"):
        r = search(blind, s, tie_break=args.tie_break)
        print(f"{s:<8} {'cycle':<9} {r.cost:>5g} {r.expanded:>9}  {r.path}")
        ablation.append({"strategy": s, "cost": r.cost, "expanded": r.expanded})
    out = {
        "tie_break": args.tie_break,
        "heuristic": {
            "admissible": report.admissible,
            "consistent": report.consistent,
            "h": dict(problem.heuristic),
            "h_star": {k: _jsonable(v) for k, v in sorted(report.true_cost.items())},
        },
        "results": rows,
        "zero_heuristic": ablation,
    }
    if not args.no_files:
        _write_json(args.results_dir / "search.json", out)
        if not args.no_plots:
            from . import plots

            args.img_dir.mkdir(parents=True, exist_ok=True)
            astar = search(problem, "astar", tie_break=args.tie_break)
            plots.delivery_graph(
                problem, problems.DELIVERY_POSITIONS, astar, args.img_dir / "delivery-astar.png"
            )
            plots.search_effort(
                [search(problem, s, tie_break=args.tie_break) for s in STRATEGIES],
                args.img_dir / "search-effort.png",
            )
    return out


def cmd_csp(args: argparse.Namespace) -> dict:
    aus = problems.australia()
    australia_rows = []
    print("Australia, 3 colours, all solutions")
    print(f"{'config':<8} {'solutions':>9} {'assignments':>12} {'backtracks':>11} {'checks':>7}")
    for name, kw in CSP_CONFIGS.items():
        r = solve(aus, **kw)
        assert all(aus.is_solution(s) for s in r.solutions)
        print(
            f"{name:<8} {len(r.solutions):>9} {r.assignments:>12} {r.backtracks:>11} {r.checks:>7}"
        )
        australia_rows.append(
            {
                "config": name,
                "solutions": len(r.solutions),
                "assignments": r.assignments,
                "backtracks": r.backtracks,
                "checks": r.checks,
            }
        )
    two = solve(problems.australia(("red", "green")))
    print(f"Australia, 2 colours: {len(two.solutions)} solutions")

    n, degree, k = args.n, args.degree, 3
    print(
        f"\nRandom graph {k}-colouring: {args.instances} instances, n={n}, mean degree {degree}, "
        f"first solution, budget {args.budget:,} assignments"
    )
    print(f"{'config':<8} {'decided':>8} {'sat':>4} {'unsat':>6} {'median':>9} {'max':>9}")
    bench = []
    for name, kw in CSP_CONFIGS.items():
        counts, sat, unsat = [], 0, 0
        for seed in range(args.instances):
            csp = random_graph_colouring(n, degree, k, seed)
            r = solve(csp, all_solutions=False, max_assignments=args.budget, **kw)
            counts.append(r.assignments)
            if r.solutions:
                sat += 1
            elif not r.exhausted:
                unsat += 1
        med = statistics.median(counts)
        print(f"{name:<8} {sat + unsat:>8} {sat:>4} {unsat:>6} {med:>9g} {max(counts):>9}")
        bench.append(
            {
                "config": name,
                "decided": sat + unsat,
                "satisfiable": sat,
                "unsatisfiable": unsat,
                "median_assignments": med,
                "max_assignments": max(counts),
            }
        )
    out = {
        "australia": australia_rows,
        "australia_first_solution": solve(aus, all_solutions=False).solutions[0],
        "australia_two_colours_solutions": len(two.solutions),
        "random_benchmark": {
            "n": n,
            "mean_degree": degree,
            "colours": k,
            "instances": args.instances,
            "budget": args.budget,
            "rows": bench,
        },
    }
    if not args.no_files:
        _write_json(args.results_dir / "csp.json", out)
        if not args.no_plots:
            from . import plots

            args.img_dir.mkdir(parents=True, exist_ok=True)
            plots.csp_solution(
                aus,
                out["australia_first_solution"],
                problems.AUSTRALIA_POSITIONS,
                problems.AUSTRALIA_NAMES,
                args.img_dir / "australia-colouring.png",
            )
    return out


def cmd_mlp(args: argparse.Namespace) -> list[dict]:
    from .mlp import run_experiments

    runs = run_experiments(seed=args.seed, max_epochs=args.max_epochs)
    print(
        f"{'experiment':<15} {'epochs':>6} {'params':>8} {'train acc':>9} {'test acc':>8} "
        f"{'train loss':>10} {'test loss':>9}"
    )
    for r in runs:
        print(
            f"{r['name']:<15} {r['epochs']:>6} {r['params']:>8} {r['train_acc']:>9.4f} "
            f"{r['test_acc']:>8.4f} {r['train_loss']:>10.4f} {r['test_loss']:>9.4f}"
        )
    if not args.no_files:
        _write_json(args.results_dir / "mlp.json", runs)
        if not args.no_plots:
            from . import plots

            args.img_dir.mkdir(parents=True, exist_ok=True)
            plots.mlp_curves(runs, args.img_dir / "mlp-curves.png")
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intro-ai", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-dir", type=Path, default=Path("results"))
    common.add_argument("--img-dir", type=Path, default=Path("docs/img"))
    common.add_argument("--no-plots", action="store_true", help="skip the PNG figures")
    common.add_argument("--no-files", action="store_true", help="print only, write nothing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", parents=[common], help="graph search on the delivery robot")
    p.add_argument("--tie-break", choices=("fifo", "lifo"), default="fifo")

    def add_csp_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--n", type=int, default=40, help="vertices per random graph")
        q.add_argument("--degree", type=float, default=4.0, help="mean degree")
        q.add_argument("--instances", type=int, default=20)
        q.add_argument("--budget", type=int, default=200_000, help="max assignments per run")

    def add_mlp_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--max-epochs", type=int, default=None, help="cap epochs (smoke runs)")

    add_csp_args(sub.add_parser("csp", parents=[common], help="map colouring by backtracking"))
    add_mlp_args(sub.add_parser("mlp", parents=[common], help="NumPy MLP on 8x8 digits"))
    p = sub.add_parser("all", parents=[common], help="run everything")
    p.add_argument("--tie-break", choices=("fifo", "lifo"), default="fifo")
    add_csp_args(p)
    add_mlp_args(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("search", "all"):
        cmd_search(args)
    if args.command in ("csp", "all"):
        print()
        cmd_csp(args)
    if args.command in ("mlp", "all"):
        print()
        cmd_mlp(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
