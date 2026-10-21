#!/usr/bin/env python3
"""Audit repair safety on 16 frozen enhanced candidates and original GTs.

This is an enhanced-candidate safety regression, not the planned three-way
ablation starting from raw candidates.  It neither reruns inference nor replaces
the official native evaluator, and its results do not extrapolate to 128 cases.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time


EXPECTED_MANIFEST_SHA256 = "9b0fd899a72e29027906751fe9caf4386f760eddf3d17124a5b9fe49732090c6"
FAILURE_SAMPLES = [25, 45, 46, 47, 54, 62, 99, 115]
SUCCESS_SAMPLES = [0, 17, 35, 56, 74, 91, 109, 127]
STATUSES = {"converged", "stalled", "iteration_limit", "infeasible_bounds",
            "time_limit", "evaluation_limit"}
VARIANTS = ("enhanced_actions", "gt_actions")
EPS = 1.
AREA_TOL = 1e-9
BOUND_TOL = 1e-9
SHAPE_TOL = 1e-7


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_frozen(root):
    manifest_path = root / "manifest.json"
    manifest_sha = digest(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen manifest is not the previously verified 16-case manifest")
    sidecar = (root / "manifest.sha256").read_text(encoding="ascii").split()
    if not sidecar or sidecar[0] != manifest_sha:
        raise ValueError("Frozen manifest sidecar mismatch")
    manifest = read_json(manifest_path)
    config = manifest.get("configuration", {})
    entries = manifest.get("cases", [])
    if (manifest.get("status") != "frozen" or len(entries) != 16
            or config.get("reference_width") != 8000
            or config.get("dental_strip_height") != 1200
            or config.get("official_overlap_area_threshold") != 50
            or config.get("strict_overlap_area_tolerance") != AREA_TOL):
        raise ValueError("Unexpected frozen scope or geometry configuration")
    expected_groups = {index: "failure_development" for index in FAILURE_SAMPLES}
    expected_groups.update({index: "success_regression" for index in SUCCESS_SAMPLES})
    if sorted(entry.get("sample_index") for entry in entries) != sorted(expected_groups):
        raise ValueError("Unexpected or duplicate frozen sample coverage")
    for entry in entries:
        index = entry["sample_index"]
        if (entry.get("group") != expected_groups[index]
                or entry.get("path") != "cases/sample_%03d.json" % index):
            raise ValueError("Unexpected frozen case identity")
        checked_case(root, entry)
    return manifest, manifest_sha


def checked_case(root, entry):
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe frozen case path")
    path = (root / Path(*relative.parts)).resolve()
    if root not in path.parents or digest(path) != entry["sha256"]:
        raise ValueError("Frozen case path or SHA-256 mismatch")
    case = read_json(path)
    for key in ("sample_index", "candidate_index", "group"):
        if case.get(key) != entry[key]:
            raise ValueError("Frozen case identity disagrees with manifest: " + key)
    ids, polygons = case.get("polygon_ids"), case.get("polygons")
    if not isinstance(ids, list) or not ids or not isinstance(polygons, list):
        raise ValueError("Missing frozen contours or IDs")
    if any(type(index) is not int or index < 0 or index >= len(polygons) for index in ids):
        raise ValueError("Invalid frozen polygon ID")
    for variant in VARIANTS:
        actions = case.get(variant)
        if (not isinstance(actions, list) or len(actions) != len(ids)
                or any(not isinstance(action, list) or len(action) != 4
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in action)
                       or (action[2] == 0 and action[3] == 0) for action in actions)):
            raise ValueError("Invalid frozen actions: " + variant)
    return path, case


def checked_build(root):
    path = root / "build.json"
    build = read_json(path)
    if (build.get("algorithm_modified") is not True or build.get("returncode") != 0
            or build.get("source_unchanged") is not True
            or build.get("repair_entry_point") != "collision_probe.overlap_repair"):
        raise ValueError("Expected the successful isolated repair build")
    return build, digest(path)


def audit(case, actions, width, height):
    from shapely.geometry import Polygon
    placed = []
    for polygon_id, action in zip(case["polygon_ids"], actions):
        angle = math.atan2(action[3], action[2])
        c, s = math.cos(angle), math.sin(angle)
        shape = Polygon([(x * c - y * s + action[0], x * s + y * c + action[1])
                         for x, y in case["polygons"][polygon_id]])
        if not shape.is_valid or shape.area <= 0:
            raise ValueError("Invalid independently transformed geometry")
        placed.append(shape)
    pairs = []
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if (a.bounds[2] < b.bounds[0] or b.bounds[2] < a.bounds[0]
                    or a.bounds[3] < b.bounds[1] or b.bounds[3] < a.bounds[1]):
                continue
            area = float(a.intersection(b).area)
            if area > 0:
                pairs.append((area, i, j))
    pairs.sort(reverse=True)
    bounds = [min(p.bounds[0] for p in placed), min(p.bounds[1] for p in placed),
              max(p.bounds[2] for p in placed), max(p.bounds[3] for p in placed)]
    total_area = sum(p.area for p in placed)
    box_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    max_area = pairs[0][0] if pairs else 0.
    return {"max_pairwise_area": max_area, "sum_pairwise_area": sum(p[0] for p in pairs),
            "pair_count_over_1e_9": sum(p[0] > AREA_TOL for p in pairs),
            "pair_count_over_50": sum(p[0] > 50 for p in pairs),
            "total_polygon_area": total_area, "bounds_xyxy": bounds,
            "height_violation": max(0., -bounds[1], bounds[3] - height),
            "width_violation": max(0., -bounds[0], bounds[2] - width),
            "in_bounds": bounds[0] >= -BOUND_TOL and bounds[1] >= -BOUND_TOL
            and bounds[2] <= width + BOUND_TOL and bounds[3] <= height + BOUND_TOL,
            "strict_zero_overlap": max_area <= AREA_TOL,
            "float64_reference_max_pair_le_50": max_area <= 50,
            "bounding_box_utilization": total_area / box_area,
            "top_pairs": [{"area": area, "piece_indices": [i, j]} for area, i, j in pairs[:3]]}, placed


def displacement(before, after):
    distances = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(before, after)]
    return {"max_translation": max(distances, default=0.),
            "sum_translation": sum(distances), "moved_piece_count": sum(v > 0 for v in distances)}


def native_status(row):
    native = row.get("native")
    return native.get("status") if isinstance(native, dict) else None


def shape_offset_error(before, after):
    worst = 0.
    for a, b in zip(before, after):
        ac, bc = list(a.exterior.coords), list(b.exterior.coords)
        if len(ac) != len(bc):
            raise ValueError("Changed shape vertex count")
        for av, bv in zip(ac, bc):
            for axis in (0, 1):
                worst = max(worst, abs((av[axis] - ac[0][axis]) - (bv[axis] - bc[0][axis])))
    return worst


def worker(args):
    if digest(args.extension / "build.json") != args.build_sha256:
        raise ValueError("Build manifest changed before worker")
    build, _ = checked_build(args.extension)
    if digest(args.frozen / "manifest.json") != args.manifest_sha256:
        raise ValueError("Frozen manifest changed before worker")
    manifest = read_json(args.frozen / "manifest.json")
    entries = [row for row in manifest["cases"] if row["sample_index"] == args.sample_index]
    if len(entries) != 1 or entries[0]["sha256"] != args.case_sha256:
        raise ValueError("Worker case differs from the validated frozen input")
    case_path, case = checked_case(args.frozen, entries[0])
    sys.path.insert(0, str(args.extension))
    import collision_probe
    module_path = Path(collision_probe.__file__).resolve()
    if module_path.parent != args.extension or digest(module_path) != build["extension_sha256"]:
        raise ValueError("Imported module differs from the isolated repair build")
    actions = case[args.variant]
    started = time.perf_counter()
    result = collision_probe.overlap_repair(case["polygon_ids"],
        [math.atan2(action[3], action[2]) for action in actions],
        [action[:2] for action in actions], case["polygons"],
        manifest["configuration"]["reference_width"],
        manifest["configuration"]["dental_strip_height"], EPS, 2000)
    elapsed = time.perf_counter() - started
    if (digest(case_path) != args.case_sha256
            or digest(args.frozen / "manifest.json") != args.manifest_sha256
            or digest(args.extension / "build.json") != args.build_sha256
            or digest(module_path) != build["extension_sha256"]):
        raise ValueError("Frozen input or build changed during native call")
    print(json.dumps({"result": result, "native_elapsed_seconds": elapsed,
                      "extension_sha256": build["extension_sha256"]}, allow_nan=False))


def guard(case, variant, before, placed_before, result, width, height):
    original = case[variant]
    xy = result["actions_xy"]
    if (not isinstance(xy, list) or len(xy) != len(original)
            or any(not isinstance(value, list) or len(value) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in value) for value in xy)):
        raise ValueError("Native actions are nonfinite or have changed dimensions")
    actions = [position + action[2:] for position, action in zip(xy, original)]
    after, placed_after = audit(case, actions, width, height)
    converged = result["status"] == "converged"
    unchanged = actions == original
    error = shape_offset_error(placed_before, placed_after)
    checks = {"known_native_status": result["status"] in STATUSES,
              "bounded_iterations": type(result["iterations"]) is int and 0 <= result["iterations"] <= 2000,
              "native_diagnostics_available": isinstance(result.get("diagnostics"), dict),
              "finite_actions_and_piece_count": True,
              "rigid_shapes_preserved": error <= SHAPE_TOL,
              "directions_preserved": all(a[2:] == b[2:] for a, b in zip(original, actions)),
              "native_rollback_flag_consistent": result["rolled_back"] is (not converged),
              "native_failure_restores_original_actions": converged or unchanged,
              "no_new_height_violation": after["height_violation"] <= before["height_violation"] + BOUND_TOL,
              "no_new_width_violation": after["width_violation"] <= before["width_violation"] + BOUND_TOL,
              "sum_overlap_not_worse": after["sum_pairwise_area"] <= before["sum_pairwise_area"] + AREA_TOL,
              "native_success_meets_independent_eps": not converged or after["max_pairwise_area"] <= EPS + AREA_TOL,
              "native_success_inside_reference_bounds": not converged or after["in_bounds"],
              "success_regression_reference_50_preserved": variant != "enhanced_actions"
              or case["group"] != "success_regression" or after["float64_reference_max_pair_le_50"],
              "original_gt_strict_reference": variant != "gt_actions" or before["strict_zero_overlap"],
              "strict_original_gt_actions_unchanged": variant != "gt_actions"
              or not before["strict_zero_overlap"] or unchanged}
    return actions, after, checks, error


def run_variant(args, entry, case, variant, width, height, build_sha, manifest_sha):
    original = case[variant]
    before, placed_before = audit(case, original, width, height)
    row = {"variant": variant, "before": before, "input_actions": original,
           "native": None, "native_after": None, "native_actions": None,
           "guard_pass": False, "guard_checks": {}, "effective_actions": original,
           "effective_after": before, "effective_status": "worker_error_rolled_back",
           "effective_rolled_back": True, "max_shape_offset_error": None}
    log = args.output / ("sample_%03d_%s.log" % (entry["sample_index"], variant))
    started = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--extension", str(args.extension), "--frozen", str(args.frozen),
            "--sample-index", str(entry["sample_index"]), "--variant", variant,
            "--build-sha256", build_sha, "--manifest-sha256", manifest_sha,
            "--case-sha256", entry["sha256"]], capture_output=True, text=True, timeout=12.,
            env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("Native worker returncode %d" % completed.returncode)
        output = json.loads(completed.stdout)
        row["native"] = output["result"]
        row["native_elapsed_seconds"] = output["native_elapsed_seconds"]
        native_actions, native_after, checks, error = guard(case, variant, before,
            placed_before, output["result"], width, height)
        row.update(native_actions=native_actions, native_after=native_after, guard_checks=checks,
                   guard_pass=all(checks.values()), max_shape_offset_error=error,
                   native_displacement=displacement(original, native_actions))
        if row["guard_pass"]:
            row.update(effective_actions=native_actions, effective_after=native_after,
                       effective_status=output["result"]["status"],
                       effective_rolled_back=output["result"]["rolled_back"])
        else:
            row["effective_status"] = "independent_guard_rejected_rolled_back"
    except (subprocess.TimeoutExpired, ValueError, KeyError, TypeError, RuntimeError) as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            log.write_text("Worker timed out after 12 seconds; original actions retained.\n", encoding="utf-8")
        if row["native"] is not None:
            row["effective_status"] = "independent_guard_rejected_rolled_back"
    row["worker_wall_seconds"] = time.perf_counter() - started
    row["effective_displacement"] = displacement(original, row["effective_actions"])
    after = row["effective_after"]
    row["effective_safety_checks"] = {
        "finite_actions_and_piece_count": len(row["effective_actions"]) == len(original)
        and all(len(action) == 4 and all(math.isfinite(v) for v in action)
                for action in row["effective_actions"]),
        "rigid_shapes_preserved": row["effective_actions"] == original
        or row["max_shape_offset_error"] is not None and row["max_shape_offset_error"] <= SHAPE_TOL,
        "directions_preserved": all(a[2:] == b[2:] for a, b in zip(original, row["effective_actions"])),
        "no_new_height_violation": after["height_violation"] <= before["height_violation"] + BOUND_TOL,
        "no_new_width_violation": after["width_violation"] <= before["width_violation"] + BOUND_TOL,
        "sum_overlap_not_worse": after["sum_pairwise_area"] <= before["sum_pairwise_area"] + AREA_TOL,
        "success_regression_reference_50_preserved": variant != "enhanced_actions"
        or case["group"] != "success_regression" or after["float64_reference_max_pair_le_50"],
        "strict_original_gt_actions_unchanged": variant != "gt_actions"
        or not before["strict_zero_overlap"] or row["effective_actions"] == original}
    row["effective_safety_pass"] = all(row["effective_safety_checks"].values())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    for option in ("sample-index", "variant", "build-sha256", "manifest-sha256", "case-sha256"):
        parser.add_argument("--" + option, type=int if option == "sample-index" else str, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.extension, args.frozen = args.extension.resolve(), args.frozen.resolve()
    if args.variant:
        if args.variant not in VARIANTS:
            parser.error("Unknown frozen action variant")
        worker(args)
        return
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.resolve()
    if any(args.output == root or root in args.output.parents or args.output in root.parents
           for root in (args.extension, args.frozen)):
        raise ValueError("Output must be separate from frozen inputs and extension build")
    manifest, manifest_sha = validate_frozen(args.frozen)
    build, build_sha = checked_build(args.extension)
    args.output.mkdir(parents=True, exist_ok=False)
    width, height = manifest["configuration"]["reference_width"], manifest["configuration"]["dental_strip_height"]
    rows = []
    for entry in manifest["cases"]:
        _, case = checked_case(args.frozen, entry)
        row = {"sample_index": entry["sample_index"], "group": entry["group"],
               "candidate_index": entry["candidate_index"], "input_path": entry["path"],
               "input_sha256": entry["sha256"], "variants": {}}
        for variant in VARIANTS:
            row["variants"][variant] = run_variant(args, entry, case, variant, width, height, build_sha, manifest_sha)
            result = row["variants"][variant]
            print(json.dumps({"sample_index": entry["sample_index"], "variant": variant,
                              "native_status": native_status(result),
                              "guard_pass": result["guard_pass"], "effective_status": result["effective_status"]}), flush=True)
        checked_case(args.frozen, entry)
        rows.append(row)
    if digest(args.frozen / "manifest.json") != manifest_sha or digest(args.extension / "build.json") != build_sha:
        raise ValueError("Frozen manifest or build manifest changed during run")
    for entry in manifest["cases"]:
        checked_case(args.frozen, entry)
    variants = [variant for row in rows for variant in row["variants"].values()]
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Enhanced-candidate safety regression on 16 fixed development/regression cases plus original GTs; not the planned raw-start three-way ablation; no inference/resampling/training; not official native evaluation; no extrapolation to 128 cases",
              "frozen_manifest_sha256": manifest_sha, "build_manifest_sha256": build_sha,
              "extension_sha256": build["extension_sha256"],
              "checker_sha256": digest(Path(__file__).resolve()),
              "configuration": {"width": width, "height": height, "solver_area_eps": EPS,
                                "max_iterations": 2000, "worker_timeout_seconds": 12,
                                "official_overlap_area_threshold_unchanged": 50,
                                "strict_area_tolerance": AREA_TOL, "boundary_tolerance": BOUND_TOL,
                                "shape_offset_tolerance": SHAPE_TOL},
              "independent_guard_policy": "Preserve native output verbatim; independently reject and restore original actions if native contract or float64 safety audit fails. A guarded rollback is never counted as native repair success.",
              "gate_pass": all(v["guard_pass"] for v in variants),
              "effective_safety_pass": all(v["effective_safety_pass"] for v in variants),
              "summary": {"cases": len(rows), "native_calls": len(variants),
                          "guard_passed": sum(v["guard_pass"] for v in variants),
                          "native_converged": sum(native_status(v) == "converged" for v in variants),
                          "independently_accepted_converged": sum(v["guard_pass"] and native_status(v) == "converged" for v in variants),
                          "guard_rejected_or_worker_failed": sum(not v["guard_pass"] for v in variants)},
              "rows": rows}
    write_json(args.output / "saved_cases.json", report)
    print(json.dumps({"gate_pass": report["gate_pass"], "effective_safety_pass": report["effective_safety_pass"],
                      **report["summary"]}, indent=2))
    if not report["gate_pass"] or not report["effective_safety_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
