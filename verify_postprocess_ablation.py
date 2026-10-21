#!/usr/bin/env python3
"""Independently verify the frozen 16-case postprocessing experiment.

Verification establishes evidence integrity, not reproduction of paper results.
This checker never reruns a solver, changes an output, or excludes a failed job.
Geometry is reconstructed independently with Shapely's affine transformations.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

MANIFEST_SHA = "9b0fd899a72e29027906751fe9caf4386f760eddf3d17124a5b9fe49732090c6"
COMMIT = "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95"
SAMPLES = (0, 17, 25, 35, 45, 46, 47, 54, 56, 62, 74, 91, 99, 109, 115, 127)
FAILURES = (25, 45, 46, 47, 54, 62, 99, 115)
ROUTES = ("gap_only", "repair_only", "repair_then_gap")
STATUSES = {"converged", "stalled", "iteration_limit", "infeasible_bounds",
            "time_limit", "evaluation_limit"}
PRECISIONS = ("float64", "float32")
HEIGHT = 1200.
AREA_EPS = 1e-9
GEOMETRY_ATOL = 1e-7


class VerificationError(ValueError):
    def __init__(self, code, location="suite"):
        self.code, self.location = code, location
        super().__init__(code)


def require(condition, code, location="suite"):
    if not condition:
        raise VerificationError(code, location)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject(value):
        raise VerificationError("nonfinite_json_constant", path.name)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def contained(root, name, location="suite"):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and
            "\\" not in name, "unsafe_relative_evidence_path", location)
    path = (root / Path(*relative.parts)).resolve()
    require(root in path.parents, "evidence_path_outside_root", location)
    return path


def close(actual, expected, location, atol=1e-10, rtol=1e-12):
    """Exact structure with a declared tolerance only for finite numbers."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                "dictionary_keys_disagree", location)
        for key in expected:
            close(actual[key], expected[key], location + "/" + str(key), atol, rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                "list_length_disagrees", location)
        for index, value in enumerate(expected):
            close(actual[index], value, location + "/" + str(index), atol, rtol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected,
                "value_disagrees", location)
    elif isinstance(expected, (int, float)):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and
                math.isfinite(float(actual)) and math.isfinite(float(expected)) and
                math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                "numeric_value_disagrees", location)
    else:
        raise VerificationError("unsupported_comparison_type", location)


def checked_actions(actions, count, location):
    require(isinstance(actions, list) and len(actions) == count,
            "action_piece_count_changed", location)
    require(all(isinstance(a, list) and len(a) == 4 and
                all(type(v) in (int, float) and math.isfinite(v) for v in a) and
                (a[2] != 0 or a[3] != 0) for a in actions),
            "invalid_action_or_rotation", location)


def fp32(actions):
    # struct's IEEE-754 conversion is independent of the runner's numpy cast.
    return [[struct.unpack("=f", struct.pack("=f", value))[0] for value in a]
            for a in actions]


def geometry(case, actions, location):
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon
    checked_actions(actions, len(case["polygon_ids"]), location)
    placed = []
    original_area = 0.
    for index, action in zip(case["polygon_ids"], actions):
        original = Polygon(case["polygons"][index])
        require(original.is_valid and original.area > 0, "invalid_input_contour", location)
        original_area += float(original.area)
        angle = math.atan2(action[3], action[2])
        shape = translate(rotate(original, angle, origin=(0., 0.), use_radians=True),
                          xoff=action[0], yoff=action[1])
        require(shape.is_valid and shape.area > 0, "invalid_placed_contour", location)
        placed.append(shape)
    bounds = [min(p.bounds[0] for p in placed), min(p.bounds[1] for p in placed),
              max(p.bounds[2] for p in placed), max(p.bounds[3] for p in placed)]
    areas = []
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if (a.bounds[2] < b.bounds[0] or b.bounds[2] < a.bounds[0] or
                    a.bounds[3] < b.bounds[1] or b.bounds[3] < a.bounds[1]):
                continue
            area = float(a.intersection(b).area)
            if area > 0:
                areas.append((area, i, j))
    areas.sort(reverse=True)
    total = math.fsum(p.area for p in placed)
    maximum = max((v[0] for v in areas), default=0.)
    overlap = math.fsum(v[0] for v in areas)
    height_violation = max(0., -bounds[1], bounds[3] - HEIGHT)
    within = height_violation <= AREA_EPS
    return {"status": "measured", "piece_count": len(placed),
            "bounds_xyxy": bounds, "total_polygon_area": total,
            "original_polygon_area": original_area,
            "rigid_area_absolute_error": abs(total-original_area),
            "max_pairwise_area": maximum, "sum_pairwise_area": overlap,
            "pair_count_over_1e_9": sum(v[0] > AREA_EPS for v in areas),
            "pair_count_over_50": sum(v[0] > 50. for v in areas),
            "positive_area_pair_count": len(areas),
            "sum_pairwise_overlap_fraction": overlap/total,
            "span_x": bounds[2]-bounds[0], "span_y": bounds[3]-bounds[1],
            "height_violation": height_violation,
            "lower_y_violation": max(0., -bounds[1]),
            "upper_y_violation": max(0., bounds[3]-HEIGHT),
            "bbox_utilization": total / ((bounds[2]-bounds[0])*(bounds[3]-bounds[1])),
            "strip_utilization": total / (HEIGHT*(bounds[2]-bounds[0])),
            "strict_overlap_valid": maximum <= AREA_EPS,
            "eps50_overlap_valid": maximum <= 50.,
            "strict_strip_valid": maximum <= AREA_EPS and within,
            "eps50_strip_valid": maximum <= 50. and within,
            "within_strip": within, "strip_height": HEIGHT,
            "zero_rotation_vector_count": 0,
            "horizontal_domain": "infinite_strip_no_x_wall",
            "area_eps": 50., "strict_area_eps": AREA_EPS, "boundary_eps": AREA_EPS,
            "top_pairs": [{"area": value, "piece_indices": [i, j]}
                          for value, i, j in areas[:3]]}


def check_geometry(actual, expected, location):
    """Check recorded geometry values, including classification near a limit."""
    require(isinstance(actual, dict), "missing_reference_geometry", location)
    for key, value in expected.items():
        if key == "top_pairs":
            # Nearly equal tiny areas can reorder pairs between affine routines;
            # all aggregate areas and threshold counts remain checked above.
            continue
        require(key in actual, "missing_reference_geometry_field", location + "/" + key)
        close(actual[key], value, location + "/" + key, GEOMETRY_ATOL, 1e-11)


def git_snapshot(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                     text=True).strip()
    require(head == COMMIT and not status, "official_upstream_not_clean_pinned")


def comparison(before, after, original, output):
    shifts = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(original, output)]
    distances = [math.hypot(x, y) for x, y in shifts]
    means = [math.fsum(v[k] for v in shifts)/len(shifts) for k in (0, 1)]
    delta = {key: after[key]-before[key] for key in
             ("max_pairwise_area", "sum_pairwise_area", "height_violation",
              "bbox_utilization", "strip_utilization", "span_x", "span_y")}
    rotations = all(a[2:] == b[2:] for a, b in zip(original, output))
    worse = delta["max_pairwise_area"] > AREA_EPS or delta["sum_pairwise_area"] > AREA_EPS
    gauge = rotations and all(y == 0 for _, y in shifts) and all(
        abs(x-means[0]) <= AREA_EPS for x, _ in shifts)
    invariants = {key: math.isclose(before[key], after[key], rel_tol=1e-10, abs_tol=1e-7)
                  for key in ("max_pairwise_area", "sum_pairwise_area", "bbox_utilization",
                              "strip_utilization", "height_violation", "span_x", "span_y",
                              "total_polygon_area")}
    return {"delta": delta, "overlap_worse": worse,
            "overlap_improved": not worse and (delta["max_pairwise_area"] < -AREA_EPS or
                                               delta["sum_pairwise_area"] < -AREA_EPS),
            "bounds_improved": delta["height_violation"] < -AREA_EPS,
            "bounds_worse": delta["height_violation"] > AREA_EPS,
            "bbox_utilization_drop": delta["bbox_utilization"] < -1e-12,
            "strip_utilization_drop": delta["strip_utilization"] < -1e-12,
            "became_eps50_strip_valid": not before["eps50_strip_valid"] and after["eps50_strip_valid"],
            "lost_eps50_strip_validity": before["eps50_strip_valid"] and not after["eps50_strip_valid"],
            "became_strict_strip_valid": not before["strict_strip_valid"] and after["strict_strip_valid"],
            "lost_strict_strip_validity": before["strict_strip_valid"] and not after["strict_strip_valid"],
            "translation_rms": math.sqrt(math.fsum(v*v for v in distances)/len(distances)),
            "translation_max": max(distances), "translation_mean_xy": means,
            "relative_translation_rms": math.sqrt(math.fsum(
                (x-means[0])**2+(y-means[1])**2 for x, y in shifts)/len(shifts)),
            "rotation_unchanged": rotations,
            "rotation_component_max_change": max(abs(b[k]-a[k])
                                                 for a, b in zip(original, output) for k in (2, 3)),
            "x_gauge_only_change": gauge,
            "x_gauge_invariance": all(invariants.values()) if gauge else None,
            "x_gauge_invariance_checks": invariants if gauge else None}


def check_comparison(actual, expected, location):
    # Changes and translation use saved values; their geometry was checked above.
    # This avoids letting tiny independent affine rounding change a delta's sign.
    close(actual, expected, location, atol=1e-10, rtol=1e-11)


def native_values(module, case, actions):
    import numpy as np
    ids = case["polygon_ids"]
    result = list(module.cal_util_all([ids],
        [[math.atan2(a[3], a[2]) for a in actions]],
        [[a[:2] for a in actions]], case["polygons"], 50.)[0])
    require(len(result) == 7 and all(math.isfinite(v) for v in result),
            "native_metric_replay_invalid")
    # The seven raw fields stay double; only the official wrapper is float32.
    values = np.asarray(result, dtype=np.float32)
    box = (values[5]-values[3])*(values[6]-values[4])
    return {"raw_native_vector": result, "threshold_area": 50.,
            "official_valid": bool(values[1] < 50.),
            "official_fp32_bbox_utilization": float(values[0]/box),
            "official_fp32_thresholded_overlap_percent": float(values[2]/values[0]*100),
            "scope": "Unmodified calutil, same float32 metric conversion as official evaluator; overlap only"}


def check_repair(job, original, location):
    native, output = job["repair_native"], job["repair_actions"]
    if job["path"] == "gap_only":
        require(native is None and output is None, "gap_route_has_hidden_repair", location)
        return None
    require(isinstance(native, dict) and native.get("status") in STATUSES,
            "missing_or_unknown_repair_status", location)
    require(type(native.get("iterations")) is int and 0 <= native["iterations"] <= 2000,
            "repair_iteration_budget_disagrees", location)
    require(native.get("rolled_back") is (native["status"] != "converged"),
            "repair_rollback_flag_disagrees", location)
    require(native.get("diagnostics", {}).get("horizontal_domain") == "unbounded",
            "repair_was_not_infinite_strip", location)
    xy = native.get("actions_xy")
    require(isinstance(xy, list) and len(xy) == len(original),
            "repair_piece_count_disagrees", location)
    expected = [list(value)+a[2:] for value, a in zip(xy, original)]
    checked_actions(expected, len(original), location)
    require(output == expected, "repair_intermediate_not_native_output", location)
    if native["rolled_back"]:
        require(output == original, "repair_failure_did_not_restore_original", location)
    return output


def validate_inputs(args):
    require(digest(args.frozen / "manifest.json") == MANIFEST_SHA,
            "frozen_manifest_is_not_prior_verified_manifest")
    manifest = read_json(args.frozen / "manifest.json")
    sidecar = (args.frozen / "manifest.sha256").read_text(encoding="ascii").split()
    require(sidecar and sidecar[0] == MANIFEST_SHA, "frozen_manifest_sidecar_disagrees")
    require(manifest["status"] == "frozen" and len(manifest["cases"]) == 16,
            "unexpected_frozen_scope")
    entries = {row["sample_index"]: row for row in manifest["cases"]}
    require(set(entries) == set(SAMPLES), "frozen_sample_coverage_disagrees")
    cases = {}
    for index, entry in entries.items():
        path = contained(args.frozen, entry["path"])
        require(digest(path) == entry["sha256"], "frozen_case_hash_disagrees", entry["path"])
        case = read_json(path)
        require(case["sample_index"] == index and case["candidate_index"] == entry["candidate_index"],
                "frozen_case_identity_disagrees", entry["path"])
        expected_group = "failure_development" if index in FAILURES else "success_regression"
        require(case["group"] == entry["group"] == expected_group,
                "frozen_baseline_group_disagrees", entry["path"])
        require(case["tensor_dtypes"]["raw_actions"] == "torch.float32" and
                case["tensor_dtypes"]["gt_actions"] == "torch.float32",
                "unexpected_input_precision", entry["path"])
        for field in ("raw_actions", "gt_actions"):
            checked_actions(case[field], len(case["polygon_ids"]), entry["path"]+"/"+field)
            require(fp32(case[field]) == case[field], "frozen_input_is_not_fp32", entry["path"])
        cases[index] = case
    build = read_json(args.extension / "build.json")
    require(build.get("source_unchanged") is True and build.get("returncode") == 0 and
            build.get("upstream_commit") == COMMIT and
            build.get("strip_repair_entry_point") == "collision_probe.overlap_repair_strip",
            "repair_build_provenance_disagrees")
    extensions = list(args.extension.glob("collision_probe*.so"))
    require(len(extensions) == 1 and digest(extensions[0]) == build["extension_sha256"],
            "repair_extension_bytes_disagree_with_build_manifest")
    require(digest(args.extension / "source" / "main.cpp") == build["instrumented_main_sha256"] and
            digest(args.extension / "source" / "collision_repair.hpp") == build["repair_header_sha256"],
            "compiled_native_source_or_header_hash_disagrees")
    git_snapshot(args.upstream)
    for name, expected in build["source_sha256"].items():
        require(digest(args.upstream / "external" / "rmspacing_cpp" / name) == expected,
                "original_native_source_hash_disagrees", name)
    source = (args.extension / "source" / "main.cpp").read_text(encoding="utf-8")
    function = source[source.index("vector<vector<pair<double,double>>> rm_spacing_all("):
                      source.index("PYBIND11_MODULE(")]
    function = re.sub(r"//[^\n]*", "", function)
    require(re.findall(r"\bwork([xy])\s*\(", function) == ["x", "y", "x", "y", "x"],
            "official_gap_pass_order_changed")
    inventory = read_json(args.results / "frozen_inputs.json")
    roots = {"frozen": args.frozen, "build": args.extension,
             "scripts": Path(__file__).resolve().parent, "native_metrics": args.native_metrics}
    require(isinstance(inventory, dict) and len(inventory) >= 23,
            "frozen_input_inventory_incomplete")
    for name, expected in inventory.items():
        prefix, relative = name.split("/", 1)
        require(prefix in roots, "unexpected_input_inventory_prefix", name)
        require(digest(contained(roots[prefix], relative, name)) == expected,
                "frozen_run_input_hash_disagrees", name)
    require(inventory.get("frozen/manifest.json") == MANIFEST_SHA and
            inventory.get("build/build.json") == digest(args.extension / "build.json"),
            "input_inventory_missing_manifest_or_build")
    for entry in entries.values():
        require(inventory.get("frozen/"+entry["path"]) == entry["sha256"],
                "frozen_case_omitted_from_run_inventory", entry["path"])
    sys.path.insert(0, str(args.native_metrics))
    module = importlib.import_module("calutil")
    module_path = Path(module.__file__).resolve()
    require(module_path.parent == args.native_metrics and
            inventory.get("native_metrics/"+module_path.name) == digest(module_path),
            "native_metric_extension_not_frozen")
    return manifest, entries, cases, build, inventory, module


def validate_outputs(args):
    metadata = read_json(args.results / "metadata.json")
    require(metadata.get("status") == "completed" and metadata.get("changed_inputs") == [],
            "runner_input_integrity_not_completed")
    outputs = metadata["output_sha256"]
    present = {path.relative_to(args.results).as_posix()
               for path in args.results.rglob("*") if path.is_file() and path.name != "metadata.json"}
    require(set(outputs) == present, "output_inventory_coverage_disagrees")
    for name, expected in outputs.items():
        require(digest(contained(args.results, name)) == expected,
                "recorded_output_hash_disagrees", name)
    return metadata, outputs


def verify(args):
    from postprocess_metrics import summarize
    manifest, entries, cases, build, inventory, module = validate_inputs(args)
    metadata, outputs = validate_outputs(args)
    require(metadata["frozen_input_count"] == len(inventory), "frozen_input_count_disagrees")
    plan, summary = read_json(args.results / "plan.json"), read_json(args.results / "summary.json")
    expected_plan = {"samples": [e["sample_index"] for e in manifest["cases"]],
        "sources": {"pred": "raw_actions", "gt": "gt_actions"}, "paths": list(ROUTES),
        "planned_jobs": 96, "outputs_per_job": list(PRECISIONS), "representatives_per_sample": 1,
        "gt_controls": 16, "height": HEIGHT, "horizontal_walls": None, "repair_epsilon": 1.,
        "repair_max_iterations": 2000, "worker_timeout_seconds": 15., "training": False,
        "new_sampling": False, "manifest_sha256": MANIFEST_SHA,
        "input_policy": "Frozen raw representative and original GT, no normalization, clamping or reselection",
        "gap_policy": "Unmodified official rm_spacing_all, x/y/x/y/x; numeric width8000 unused by gap code",
        "rollback_policy": "Repair internal failure rollback retained; no external guard rollback of gap outputs"}
    for key, value in expected_plan.items():
        close(plan.get(key), value, "plan/"+key)
    require(summary.get("status") == "completed" and summary.get("planned_jobs") == 96 and
            summary.get("completed_jobs") == 96 and len(summary.get("rows", [])) == 96,
            "incomplete_planned_job_coverage")
    expected_keys = {(index, source, route) for index in SAMPLES
                     for source in ("pred", "gt") for route in ROUTES}
    seen, rows = set(), {p: [] for p in PRECISIONS}
    repair_statuses, rollbacks, failed = Counter(), Counter(), 0
    native_success_reference_eps1_failures = []
    reference_geometry_checks, native_replays = 0, 0
    original_metrics, private_errors = {}, []
    repair_gap_rows, repair_gap_matched_rows, repair_gap_details, cast_rows = [], [], [], []
    for job in summary["rows"]:
        key = (job["sample_index"], job["source"], job["path"])
        require(key in expected_keys and key not in seen, "duplicate_or_unplanned_job")
        seen.add(key)
        index, source, route = key
        relative = "jobs/sample_%03d_%s_%s.json" % key
        require(relative in outputs and read_json(contained(args.results, relative)) == job,
                "job_file_disagrees_with_summary", relative)
        case, entry = cases[index], entries[index]
        require(job["candidate_index"] == case["candidate_index"] and
                job["case_sha256"] == entry["sha256"] and
                job["baseline_group"] == ("failure" if index in FAILURES else "success"),
                "job_identity_disagrees_with_frozen_case", relative)
        original = case["raw_actions" if source == "pred" else "gt_actions"]
        if job["status"] == "failed":
            failed += 1
            require(job.get("stages") == {}, "failed_job_has_unclassified_output", relative)
            private_errors.append({"sample_index": index, "source": source, "path": route,
                                   "error_type": job.get("error", "unspecified")})
            for precision in PRECISIONS:
                rows[precision].append({"sample_index": index, "source": source,
                    "baseline_group": job["baseline_group"], "path": route,
                    "status": "failed", "after": None, "comparison": None,
                    "output_precision": precision})
                if route == "repair_then_gap":
                    failed_row = {"sample_index": index, "source": source,
                        "baseline_group": job["baseline_group"], "path": route,
                        "status": "failed", "after": None, "comparison": None,
                        "output_precision": precision}
                    repair_gap_rows.append(failed_row)
                    repair_gap_matched_rows.append(dict(failed_row))
            cast_rows.append({"sample_index": index, "source": source,
                "baseline_group": job["baseline_group"], "path": route,
                "status": "failed", "after": None, "comparison": None,
                "output_precision": "float32"})
            continue
        require(job["status"] == "completed", "unknown_job_status", relative)
        require(job["input_field"] == ("raw_actions" if source == "pred" else "gt_actions") and
                job["input_actions"] == original, "raw_or_gt_input_was_transformed", relative)
        require(job["extension_sha256"] == build["extension_sha256"] and
                job["calutil_sha256"] == digest(Path(module.__file__)),
                "job_native_extension_hash_disagrees", relative)
        if (index, source) not in original_metrics:
            original_metrics[index, source] = geometry(case, original, relative+"/before")
        before = original_metrics[index, source]
        check_geometry(job["before"], before, relative+"/before")
        reference_geometry_checks += 1
        close(job["before_native"], native_values(module, case, original),
              relative+"/before_native", GEOMETRY_ATOL, 1e-11)
        native_replays += 1
        intermediate = check_repair(job, original, relative)
        repair_after = None
        if route != "gap_only":
            status = job["repair_native"]["status"]
            repair_statuses[status] += 1
            rollbacks[str(job["repair_native"]["rolled_back"]).lower()] += 1
            repair_after = geometry(case, intermediate, relative+"/repair")
            reference_geometry_checks += 1
            if status == "converged":
                if repair_after["max_pairwise_area"] > 1.+AREA_EPS or not repair_after["within_strip"]:
                    native_success_reference_eps1_failures.append(
                        {"sample_index": index, "source": source, "path": route})
        if route == "repair_only":
            require(job["gap_input_actions"] is None and job["gap_passes"] == [],
                    "repair_route_has_hidden_gap", relative)
            require(job["stages"]["float64"]["actions"] == intermediate,
                    "repair_only_final_not_native_return", relative)
        else:
            require(job["gap_input_actions"] == (original if route == "gap_only" else intermediate),
                    "gap_input_has_hidden_transform", relative)
            require(job["gap_passes"] == ["x", "y", "x", "y", "x"],
                    "gap_pass_ledger_disagrees", relative)
        require(set(job["stages"]) == set(PRECISIONS), "precision_output_coverage_disagrees", relative)
        wide = job["stages"]["float64"]["actions"]
        require(job["stages"]["float32"]["actions"] == fp32(wide),
                "fp32_output_is_not_single_final_cast", relative)
        for precision in PRECISIONS:
            stage = job["stages"][precision]
            actions = stage["actions"]
            checked_actions(actions, len(original), relative+"/"+precision)
            require(all(a[2:] == b[2:] for a, b in zip(original, actions)),
                    "rotation_components_changed", relative+"/"+precision)
            independent = geometry(case, actions, relative+"/"+precision)
            check_geometry(stage["after"], independent, relative+"/"+precision+"/after")
            reference_geometry_checks += 1
            expected_comparison = comparison(job["before"], stage["after"], original, actions)
            check_comparison(stage["comparison"], expected_comparison,
                             relative+"/"+precision+"/comparison")
            close(stage["native"], native_values(module, case, actions),
                  relative+"/"+precision+"/native", GEOMETRY_ATOL, 1e-11)
            native_replays += 1
            rows[precision].append({"sample_index": index, "source": source,
                "baseline_group": job["baseline_group"], "path": route, "status": "completed",
                "after": stage["after"], "comparison": stage["comparison"],
                "output_precision": precision})
            if route == "repair_then_gap":
                # Geometry here is independently reconstructed on both sides.
                # The first view includes the final write's precision effect;
                # the matched view casts the repair baseline the same way.
                before_matched_actions = intermediate if precision == "float64" else fp32(intermediate)
                before_matched = repair_after if precision == "float64" else geometry(
                    case, before_matched_actions, relative+"/repair_float32")
                if precision == "float32":
                    reference_geometry_checks += 1
                gap_change = comparison(repair_after, independent, intermediate, actions)
                matched_change = comparison(before_matched, independent, before_matched_actions, actions)
                shared = {"sample_index": index, "source": source,
                          "baseline_group": job["baseline_group"], "path": route,
                          "status": "completed", "after": independent,
                          "output_precision": precision}
                repair_gap_rows.append(dict(shared, comparison=gap_change))
                repair_gap_matched_rows.append(dict(shared, comparison=matched_change))
                fields = ("max_pairwise_area", "sum_pairwise_area", "height_violation",
                          "strip_utilization", "within_strip", "eps50_strip_valid", "strict_strip_valid")
                repair_gap_details.append({"sample_index": index, "source": source,
                    "output_precision": precision,
                    "repair_native_float64": {k: repair_after[k] for k in fields},
                    "repair_matched_precision": {k: before_matched[k] for k in fields},
                    "final_after_gap": {k: independent[k] for k in fields},
                    "change_from_native_repair": gap_change,
                    "change_from_matched_precision_repair": matched_change})
        wide_metrics = job["stages"]["float64"]["after"]
        narrow_metrics = job["stages"]["float32"]["after"]
        cast_rows.append({"sample_index": index, "source": source,
            "baseline_group": job["baseline_group"], "path": route, "status": "completed",
            "after": narrow_metrics,
            "comparison": comparison(wide_metrics, narrow_metrics, wide,
                                     job["stages"]["float32"]["actions"]),
            "output_precision": "float32"})
    require(seen == expected_keys and failed == summary["failed_jobs"],
            "planned_or_failed_job_count_disagrees")
    recomputed = {precision: summarize(rows[precision]) for precision in PRECISIONS}
    close(summary["aggregates"], recomputed, "summary/aggregates")
    for precision in PRECISIONS:
        overall = recomputed[precision]["overall"]
        require(overall["planned_outputs"] == 96 and overall["failed_outputs"] == failed,
                "failed_job_denominator_dropped", precision)
    require(len(repair_gap_rows) == 64 and len(repair_gap_matched_rows) == 64 and len(cast_rows) == 96,
            "intermediate_or_precision_denominator_dropped")
    # Recheck all bytes after CPU replay; no input or output may change unnoticed.
    validate_inputs(args)
    validate_outputs(args)
    git_snapshot(args.upstream)
    return {"schema_version": 1, "status": "verified",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "meaning": "Evidence integrity and available provenance, not algorithm success or full paper reproduction",
            "scope": "16 selected raw development representatives and 16 original GT controls; no resampling, training or 512-candidate evaluation",
            "planned_jobs": 96, "failed_jobs": failed, "planned_precision_outputs": 192,
            "measured_precision_outputs": 2*(96-failed),
            "geometry_reconstructions_checked": reference_geometry_checks,
            "native_metric_replays_checked": native_replays,
            "native_repair_status_counts": dict(sorted(repair_statuses.items())),
            "native_repair_rollback_counts": dict(sorted(rollbacks.items())),
            "native_converged_reference_eps1_failures": native_success_reference_eps1_failures,
            "failed_job_records": private_errors,
            "checks": {"all_96_jobs_and_192_precision_slots_retained": True,
                       "original_raw_and_gt_inputs_unchanged": True,
                       "no_x_wall_and_height_1200": True,
                       "official_gap_five_pass_source_and_ledger": True,
                       "repair_failure_internal_rollback_retained": True,
                       "no_external_guard_rollback": True,
                       "all_fp32_outputs_single_final_cast": True,
                       "rotation_components_unchanged": True,
                       "independent_affine_reference_geometry": True,
                       "unchanged_calutil_replayed_on_cpu": True,
                       "recorded_changes_recomputed": True,
                       "all_summary_denominators_and_values_recomputed": True,
                       "all_input_output_hashes_and_clean_upstream_rechecked": True},
            "frozen_manifest_sha256": MANIFEST_SHA,
            "build_manifest_sha256": digest(args.extension / "build.json"),
            "extension_sha256": build["extension_sha256"],
            "input_inventory_sha256": digest(args.results / "frozen_inputs.json"),
            "input_inventory_count": len(inventory), "output_inventory_count": len(outputs),
            "input_inventory": inventory,
            "output_inventory": outputs,
            "verifier_sha256": digest(Path(__file__).resolve()),
            "geometry_absolute_tolerance": GEOMETRY_ATOL,
            "geometry_relative_tolerance": 1e-11,
            "geometry_method": "Shapely.affinity.rotate(use_radians=True) then translate; no runner geometry imported",
            "summary_method": "Shared aggregation formulas over independently checked geometry and independently recomputed comparisons; planned denominators additionally checked",
            "repair_to_gap_effects": {
                "native_baseline_note": "Native repair FP64 intermediate to final output; float32 final view includes its write conversion",
                "matched_precision_note": "Repair intermediate and final output use the same precision, with no solver rerun",
                "planned_precision_outputs": 64,
                "relative_to_native_repair": summarize(repair_gap_rows),
                "relative_to_matched_precision_repair": summarize(repair_gap_matched_rows),
                "rows": repair_gap_details},
            "float32_final_cast_effects": {
                "note": "Final FP64 output to its one-time FP32 write; no rerun and no extra repair",
                "planned_outputs": 96, "aggregates": summarize(cast_rows)},
            "aggregates": recomputed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "extension", "native-metrics", "upstream", "results", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("frozen", "extension", "native_metrics", "upstream", "results", "output"):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output.exists(), "verification_output_already_exists")
    require(args.results not in args.output.parents and args.output != args.results,
            "verification_output_must_not_modify_run_inventory")
    try:
        result = verify(args)
    except VerificationError as exc:
        result = {"schema_version": 1, "status": "failed", "error_code": exc.code,
                  "evidence_location": exc.location,
                  "verifier_sha256": digest(Path(__file__).resolve())}
    except Exception as exc:
        result = {"schema_version": 1, "status": "failed",
                  "error_code": "unexpected_"+type(exc).__name__,
                  "verifier_sha256": digest(Path(__file__).resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "planned_jobs", "failed_jobs",
          "geometry_reconstructions_checked", "error_code", "evidence_location") if key in result},
          allow_nan=False))
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
