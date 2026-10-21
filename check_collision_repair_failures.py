#!/usr/bin/env python3
"""Exercise repaired native collision failure states and geometric edge cases.

Every call runs in a fresh process with a time limit.  The native extension is
checked against its build manifest, and Shapely independently audits every
successful layout.  This file contains no collision solver.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


STATUSES = {"converged", "stalled", "iteration_limit", "infeasible_bounds",
            "time_limit", "evaluation_limit"}


def controls():
    square = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    rect = [[0., 0.], [3., 0.], [3., 1.], [0., 1.]]
    concave = [[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 3.], [0., 3.]]

    def case(name, polygons, ids, xy, width=10., height=6., eps=.01,
             rotates=None, max_iterations=2000, expected="converged", unchanged=False):
        return {"name": name, "polygons": polygons, "polygon_ids": ids,
                "actions_xy": xy, "rotates": rotates or [0.] * len(ids),
                "width": width, "height": height, "eps": eps,
                "max_iterations": max_iterations, "expected_status": expected,
                "expect_unchanged": unchanged}

    rows = [
        case("feasible_shifted_keeps_actions", [square, concave], [0, 1],
             [[1., 1.], [3., 1.]], unchanged=True),
        case("feasible_rotations_keep_actions", [rect], [0, 0], [[2., 1.], [7., 1.]],
             rotates=[math.radians(37.), math.pi / 2], unchanged=True),
        case("concave_void_is_not_collision", [concave, square], [0, 1],
             [[0., 0.], [1.5, 1.5]], unchanged=True),
        case("touching_keeps_actions", [square], [0, 0], [[0., 0.], [1., 0.]],
             unchanged=True),
        case("overlap_below_eps_keeps_actions", [square], [0, 0], [[0., 0.], [.999, 0.]],
             unchanged=True),
        case("rotated_rectangle_overlap", [rect], [0, 0], [[2., 1.], [3., 2.]],
             rotates=[math.radians(37.), math.pi / 2]),
        case("reversed_winding_concave_overlap", [list(reversed(concave)), concave],
             [0, 1], [[0., 0.], [.7, .7]]),
        case("one_rotated_piece_keeps_actions", [rect], [0], [[2., 1.]],
             rotates=[math.radians(37.)], unchanged=True),
        case("negative_origin_is_not_success", [square], [0], [[-.25, 1.]],
             expected="infeasible_bounds"),
        case("upper_bound_is_not_success", [square], [0], [[9.25, 1.]],
             expected="infeasible_bounds"),
        case("oversized_rotated_piece", [rect], [0], [[1., 0.]], 2., 2.,
             rotates=[math.pi / 2], expected="infeasible_bounds"),
        case("stalled_fixed_full_box", [square], [0, 0], [[0., 0.], [0., 0.]],
             1., 1., eps=0., expected="stalled"),
        case("stalled_overfull_layout_rolls_back", [square], [0, 0, 0],
             [[0., 0.], [0., 0.], [0., 0.]], 2., 1., eps=0., expected="stalled"),
        case("zero_budget_overlap_rolls_back", [square], [0, 0],
             [[0., 0.], [.25, 0.]], eps=0., max_iterations=0,
             expected="iteration_limit"),
        case("zero_budget_feasible_is_success", [square], [0, 0],
             [[1., 1.], [3., 1.]], max_iterations=0, unchanged=True),
        case("one_round_budget_rolls_back", [square], [0] * 5,
             [[0., 0.] for _ in range(5)], 5., 1., eps=0., max_iterations=1,
             expected="iteration_limit"),
    ]
    invalid = ["empty_piece_list", "missing_rotation", "missing_translation",
               "negative_polygon_id", "out_of_range_polygon_id",
               "nonfinite_rotation", "nonfinite_translation", "nonfinite_vertex",
               "too_few_vertices", "degenerate_polygon", "zero_area_self_crossing_polygon",
               "nonfinite_width", "zero_width", "negative_height", "nonfinite_eps",
               "negative_eps", "negative_iteration_limit", "unbounded_iteration_limit"]
    for mutation in invalid:
        row = case("reject_" + mutation, [square], [0], [[0., 0.]])
        row["invalid_mutation"] = mutation
        rows.append(row)
    return rows


def mutate(case):
    """Build invalid native values without putting NaN/Infinity in saved JSON."""
    mutation = case.get("invalid_mutation")
    if mutation == "empty_piece_list":
        case["polygon_ids"], case["rotates"], case["actions_xy"] = [], [], []
    elif mutation == "missing_rotation":
        case["rotates"] = []
    elif mutation == "missing_translation":
        case["actions_xy"] = []
    elif mutation == "negative_polygon_id":
        case["polygon_ids"][0] = -1
    elif mutation == "out_of_range_polygon_id":
        case["polygon_ids"][0] = len(case["polygons"])
    elif mutation == "nonfinite_rotation":
        case["rotates"][0] = float("nan")
    elif mutation == "nonfinite_translation":
        case["actions_xy"][0][0] = float("inf")
    elif mutation == "nonfinite_vertex":
        case["polygons"][0][0][1] = float("nan")
    elif mutation == "too_few_vertices":
        case["polygons"][0] = [[0., 0.], [1., 0.]]
    elif mutation == "degenerate_polygon":
        case["polygons"][0] = [[0., 0.], [1., 0.], [2., 0.]]
    elif mutation == "zero_area_self_crossing_polygon":
        case["polygons"][0] = [[0., 0.], [1., 1.], [0., 1.], [1., 0.]]
    elif mutation == "nonfinite_width":
        case["width"] = float("inf")
    elif mutation == "zero_width":
        case["width"] = 0.
    elif mutation == "negative_height":
        case["height"] = -1.
    elif mutation == "nonfinite_eps":
        case["eps"] = float("nan")
    elif mutation == "negative_eps":
        case["eps"] = -.01
    elif mutation == "negative_iteration_limit":
        case["max_iterations"] = -1
    elif mutation == "unbounded_iteration_limit":
        case["max_iterations"] = 2001
    elif mutation:
        raise ValueError("Unknown invalid-input mutation: " + mutation)
    return case


def audit(case, actions_xy):
    from shapely.geometry import Polygon
    placed = []
    for polygon_id, angle, (tx, ty) in zip(case["polygon_ids"], case["rotates"], actions_xy):
        c, s = math.cos(angle), math.sin(angle)
        shape = Polygon([(x * c - y * s + tx, x * s + y * c + ty)
                         for x, y in case["polygons"][polygon_id]])
        if not shape.is_valid or shape.area <= 0:
            raise ValueError("Invalid geometry in independent audit")
        placed.append(shape)
    areas = [a.intersection(b).area for i, a in enumerate(placed) for b in placed[i + 1:]]
    bounds = [min(p.bounds[0] for p in placed), min(p.bounds[1] for p in placed),
              max(p.bounds[2] for p in placed), max(p.bounds[3] for p in placed)]
    return {"max_pairwise_area": max(areas, default=0.), "sum_pairwise_area": sum(areas),
            "bounds_xyxy": bounds, "in_bounds": bounds[0] >= -1e-9 and bounds[1] >= -1e-9
            and bounds[2] <= case["width"] + 1e-9 and bounds[3] <= case["height"] + 1e-9}


def worker(args):
    build = json.loads((args.extension / "build.json").read_text())
    sys.path.insert(0, str(args.extension.resolve()))
    import collision_probe
    native = Path(collision_probe.__file__).resolve()
    if native.parent != args.extension.resolve() or hashlib.sha256(native.read_bytes()).hexdigest() != build["extension_sha256"]:
        raise ValueError("Imported extension does not match the isolated repair build")
    case = mutate(json.loads(args.case.read_text()))
    before = json.dumps(case, sort_keys=True)
    started = time.perf_counter()
    try:
        result = collision_probe.overlap_repair(case["polygon_ids"], case["rotates"],
            case["actions_xy"], case["polygons"], case["width"], case["height"],
            case["eps"], case["max_iterations"])
        output = {"kind": "result", "result": result}
    except Exception as exc:
        output = {"kind": "exception", "type": type(exc).__name__, "message": str(exc)}
    output["elapsed_seconds"] = time.perf_counter() - started
    output["caller_inputs_unchanged"] = before == json.dumps(case, sort_keys=True)
    print(json.dumps(output, allow_nan=False))


def check(case, output):
    if case.get("invalid_mutation"):
        return {"invalid_input_rejected": output["kind"] == "exception"
                and output["type"] == "ValueError", "caller_inputs_unchanged": output["caller_inputs_unchanged"]}, None
    if output["kind"] != "result":
        raise ValueError("Unexpected native exception: " + output.get("message", ""))
    result = output["result"]
    actions = result["actions_xy"]
    count_ok = len(actions) == len(case["actions_xy"])
    finite = count_ok and all(len(action) == 2 and all(math.isfinite(v) for v in action)
                              for action in actions)
    if not finite:
        raise ValueError("Native output changed piece count or returned nonfinite/wrong-sized actions")
    after = audit(case, actions)
    converged = result["status"] == "converged"
    unchanged = actions == case["actions_xy"]
    checks = {"caller_inputs_unchanged": output["caller_inputs_unchanged"],
              "finite_actions_and_piece_count": finite,
              "explicit_known_status": result["status"] in STATUSES,
              "expected_status": result["status"] == case["expected_status"],
              "bounded_iterations": isinstance(result["iterations"], int)
              and 0 <= result["iterations"] <= case["max_iterations"],
              "diagnostics_available": isinstance(result.get("diagnostics"), dict),
              "rollback_flag_consistent": result["rolled_back"] is (not converged),
              "all_failure_actions_rolled_back": converged or unchanged,
              "successful_geometry_meets_eps": not converged or after["max_pairwise_area"] <= case["eps"] + 1e-9,
              "successful_geometry_in_bounds": not converged or after["in_bounds"],
              "feasible_actions_preserved": not case["expect_unchanged"] or unchanged}
    return checks, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=10.)
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        worker(args)
        return
    if args.output is None or not 0 < args.timeout <= 60:
        parser.error("--output and a timeout in (0,60] seconds are required")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in controls():
        path = args.output / (case["name"] + "_input.json")
        path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        row = {"name": case["name"], "input": case,
               "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "status": "failed", "gate_pass": False}
        if not case.get("invalid_mutation"):
            row["before"] = audit(case, case["actions_xy"])
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--extension", str(args.extension.resolve()), "--case", str(path.resolve())],
                capture_output=True, text=True, timeout=args.timeout,
                env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
            (args.output / (case["name"] + ".log")).write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError("Native worker returncode %d" % completed.returncode)
            output = json.loads(completed.stdout)
            checks, after = check(case, output)
            row.update(status="completed", output=output, checks=checks, gate_pass=all(checks.values()))
            if after is not None:
                row["after"] = after
        except (subprocess.TimeoutExpired, ValueError, RuntimeError, KeyError, TypeError) as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        rows.append(row)
        print(json.dumps({"name": row["name"], "gate_pass": row["gate_pass"]}), flush=True)
    report = {"scope": "Synthetic repair failure and geometry controls; no dataset ablation",
              "official_metric_threshold_unchanged": 50, "strict_audit_tolerance": 1e-9,
              "worker_timeout_seconds": args.timeout,
              "build_manifest_sha256": hashlib.sha256((args.extension / "build.json").read_bytes()).hexdigest(),
              "gate_pass": all(row["gate_pass"] for row in rows), "rows": rows}
    (args.output / "repair_failures.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate_pass": report["gate_pass"], "cases": len(rows),
                      "passed": sum(row["gate_pass"] for row in rows)}, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
