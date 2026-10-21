#!/usr/bin/env python3
"""Run bounded synthetic controls against isolated collision implementations.

Each native call runs in a fresh process with a wall-time limit. The gate also
checks the geometry after the function returns, rather than trusting its flag.
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

from diagnose_results import geometry


def controls():
    square = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    concave = [[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 3.], [0., 3.]]
    def case(name, category, polygons, ids, xy, width=10., height=6., eps=.01):
        return {"name": name, "category": category, "polygons": polygons, "polygon_ids": ids,
                "actions": [[x, y, 1., 0.] for x, y in xy], "width": width, "height": height,
                "native_solver_area_eps": eps}
    square10 = [[10*x, 10*y] for x, y in square]
    return [
        case("touching", "touching", [square], [0, 0], [[0., 0.], [1., 0.]]),
        case("shallow_detectable", "shallow_overlap", [square], [0, 0], [[0., 0.], [.98, 0.]]),
        case("shallow_below_eps", "precision_control", [square], [0, 0], [[0., 0.], [.999, 0.]]),
        case("obvious_overlap", "obvious_overlap", [square], [0, 0], [[0., 0.], [.5, .3]]),
        case("concave_overlap", "concave", [concave], [0, 0], [[0., 0.], [.7, .7]]),
        case("near_lower_boundary", "near_strip_boundary", [square], [0, 0], [[0., 0.], [0., .8]], 2., 2.),
        case("near_upper_boundary", "near_strip_boundary", [square], [0, 0], [[0., 1.], [0., .8]], 2., 2.),
        case("tight_height_feasible", "near_strip_boundary", [square], [0, 0], [[0., 0.], [0., .2]], 2., 1.2),
        case("caller_eps_boundary", "caller_eps_near_boundary", [square10], [0, 0], [[0., 0.], [0., 8.]], 20., 20., 1.),
        case("caller_eps_tight_height", "caller_eps_tight_height", [square10], [0, 0], [[0., 0.], [0., 2.]], 20., 12., 1.),
        case("already_feasible", "already_no_overlap", [square, concave], [0, 1], [[1., 1.], [3., 1.]]),
    ]


def audit(case, actions):
    info, placed = geometry(case["polygons"], case["polygon_ids"], actions)
    bounds = info["bounds_xyxy"]
    info["max_height"] = bounds[3]
    info["negative_coordinate_magnitude"] = max(0., -bounds[0], -bounds[1])
    info["height_violation"] = max(0., -bounds[1], bounds[3] - case["height"])
    info["reference_width_violation"] = max(0., -bounds[0], bounds[2] - case["width"])
    info["bounding_box_utilization"] = info["total_polygon_area"] / ((bounds[2]-bounds[0]) * (bounds[3]-bounds[1]))
    info["strict_zero_overlap"] = info["max_pairwise_area"] <= 1e-9
    return info, placed


def worker(args):
    sys.path.insert(0, str(args.extension.resolve()))
    import collision_probe
    module_path = Path(collision_probe.__file__).resolve()
    build = json.loads((args.extension / "build.json").read_text())
    if (module_path.parent != args.extension.resolve()
            or hashlib.sha256(module_path.read_bytes()).hexdigest() != build["extension_sha256"]):
        raise ValueError("Imported extension does not match the isolated build")
    case = json.loads(args.case.read_text())
    started = time.perf_counter()
    inputs = (case["polygon_ids"], [math.atan2(a[3], a[2]) for a in case["actions"]],
              [a[:2] for a in case["actions"]], case["polygons"], case["width"],
              case["height"], case["native_solver_area_eps"])
    if args.repair:
        if not build.get("algorithm_modified") or not hasattr(collision_probe, "overlap_repair"):
            raise ValueError("The isolated build does not contain the requested repair")
        result = collision_probe.overlap_repair(*inputs, max_iterations=args.max_iterations)
    else:
        result = collision_probe.overlap_probe(*inputs)
    result["elapsed_seconds"] = time.perf_counter() - started
    xy_values = result.pop("actions_xy")
    if len(xy_values) != len(case["actions"]) or any(len(xy) != 2 for xy in xy_values):
        raise ValueError("Native output changed piece count or action dimensions")
    result["actions"] = [list(xy) + a[2:] for xy, a in zip(xy_values, case["actions"])]
    print(json.dumps(result, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=10.)
    parser.add_argument("--repair", action="store_true", help="Run the bounded repair entry point")
    parser.add_argument("--max-iterations", type=int, default=2000,
                        help="Accepted-step budget for --repair; zero checks rollback on exhaustion")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_iterations < 0:
        parser.error("--max-iterations must be nonnegative")
    if args.case:
        worker(args)
        return
    if args.output is None or not 0 < args.timeout <= 60:
        parser.error("--output and a positive timeout of at most 60 seconds are required")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in controls():
        path = args.output / (case["name"] + "_input.json")
        path.write_text(json.dumps(case, indent=2) + "\n")
        row = {"name": case["name"], "category": case["category"],
               "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "native_solver_area_eps": case["native_solver_area_eps"],
               "input": case, "before": audit(case, case["actions"])[0], "status": "failed"}
        try:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--extension", str(args.extension.resolve()), "--case", str(path.resolve())]
            if args.repair:
                command.extend(["--repair", "--max-iterations", str(args.max_iterations)])
            completed = subprocess.run(command,
                capture_output=True, text=True, timeout=args.timeout,
                env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
            (args.output / (case["name"] + ".log")).write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError("native worker returncode %d" % completed.returncode)
            result = json.loads(completed.stdout)
            actions = result["actions"]
            finite = all(math.isfinite(v) for action in actions for v in action)
            count_ok = len(actions) == len(case["actions"])
            if not finite or not count_ok:
                raise ValueError("Nonfinite output or changed piece count")
            after, placed_after = audit(case, actions)
            before, placed_before = audit(case, case["actions"])
            # Translation-only output must preserve every vertex offset.
            shape_error = max(abs((b2-b1) - (a2-a1))
                for p, q in zip(placed_before, placed_after)
                for (ax, ay), (bx, by) in zip(p.exterior.coords, q.exterior.coords)
                for a1, a2, b1, b2 in [(p.exterior.coords[0][0], ax, q.exterior.coords[0][0], bx),
                                      (p.exterior.coords[0][1], ay, q.exterior.coords[0][1], by)])
            checks = {"finite": finite, "piece_count_preserved": count_ok,
                      "rigid_shapes_preserved": shape_error <= 1e-9,
                      "no_new_height_violation": after["height_violation"] <= before["height_violation"] + 1e-9,
                      "no_new_width_violation": after["reference_width_violation"] <= before["reference_width_violation"] + 1e-9,
                      "overlap_not_worse": after["sum_pairwise_area"] <= before["sum_pairwise_area"] + 1e-9,
                      "solver_did_not_hit_iteration_limit":
                          (result["status"] if args.repair else result["return_path"]) != "iteration_limit",
                      "returned_geometry_meets_solver_eps": after["max_pairwise_area"] <= case["native_solver_area_eps"] + 1e-9}
            if args.repair:
                checks.update(repair_reported_convergence=result["status"] == "converged",
                              repair_did_not_roll_back=result["rolled_back"] is False)
            row.update(status="completed", after=after, output=result, checks=checks,
                       max_shape_offset_error=shape_error, gate_pass=all(checks.values()))
        except (subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                def as_text(value):
                    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
                (args.output / (case["name"] + ".log")).write_text(
                    as_text(exc.stdout) + as_text(exc.stderr) + "\nWorker exceeded wall-time limit.\n")
            row["error"] = type(exc).__name__ + ": " + str(exc)
            row["gate_pass"] = False
        rows.append(row)
        print(json.dumps({"name": row["name"], "status": row["status"], "gate_pass": row["gate_pass"]}), flush=True)
    report = {"scope": ("Synthetic controls only; isolated bounded collision repair, no dataset ablation"
                         if args.repair else
                         "Synthetic controls only; the original inactive rm_overlap, no dataset ablation"),
              "algorithm_modified": args.repair,
              "native_iteration_limit": args.max_iterations if args.repair else 2000,
              "worker_timeout_seconds": args.timeout, "strict_area_tolerance": 1e-9,
              "official_metric_threshold_unchanged": 50,
              "solver_eps_note": ".01 is the original rm_overlap default; 1 is the inactive caller argument; both are independent of official score threshold 50",
              "gate_note": "Plan invariants plus bounded completion and independent check of the solver's own area epsilon",
              "gate_pass": all(row["gate_pass"] for row in rows), "rows": rows,
              "build_manifest_sha256": hashlib.sha256((args.extension / "build.json").read_bytes()).hexdigest()}
    (args.output / "controls.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # A failed algorithm gate is an experimental finding, not a harness error.
    print(json.dumps({"gate_pass": report["gate_pass"], "cases": len(rows),
                      "passed": sum(row["gate_pass"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
