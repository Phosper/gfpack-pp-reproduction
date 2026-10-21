#!/usr/bin/env python3
"""Verify a six-run development suite and write a path-free public summary.

``verified`` means that the saved evidence is internally consistent and its
available provenance has been checked on the machine holding the inputs. It
does not mean that the released checkpoint reproduces the paper. Failed runs
remain in every planned-candidate denominator. No model inference is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from paper_metrics import aggregate_metrics, sample_metrics


COMMIT = "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95"
PROFILES = ("release_weights_constraints", "paper_mechanism_release_scale", "paper_settings")
SAMPLES = (0, 45)
STAGES = {
    "raw": "predictions_raw.pt",
    "enhanced_float64": "predictions_enhanced_float64.pt",
    "enhanced_fp32": "predictions_enhanced_fp32.pt",
}
CANDIDATES = 512
GEOMETRY_ATOL = 1e-7


class VerificationError(ValueError):
    """A public error code and a suite-relative evidence location."""

    def __init__(self, code, location="suite"):
        self.code, self.location = code, location
        super().__init__(code)


def require(condition, code, location="suite"):
    if not condition:
        raise VerificationError(code, location)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root, name, location):
    path = (root / name).resolve()
    require(root in path.parents, "evidence_path_outside_run", location)
    return path


def close_tree(actual, expected, location, atol=1e-10, rtol=1e-12):
    """Compare recomputed formulas without exposing private JSON values."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                "recomputed_dictionary_keys_disagree", location)
        for key in expected:
            close_tree(actual[key], expected[key], location + "/" + str(key), atol, rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                "recomputed_list_length_disagrees", location)
        for index, value in enumerate(expected):
            close_tree(actual[index], value, location + "/" + str(index), atol, rtol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected,
                "recomputed_value_disagrees", location)
    elif isinstance(expected, (int, float)):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and
                math.isfinite(float(actual)) and math.isfinite(float(expected)) and
                math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                "recomputed_number_disagrees", location)
    else:
        raise VerificationError("unsupported_evidence_value", location)


def verify_stage(report, sample_index, location):
    """Rebuild feasibility flags, utilities, selections and all denominators."""
    require(isinstance(report.get("samples"), list) and len(report["samples"]) == 1,
            "stage_must_contain_one_sample", location)
    saved = report["samples"][0]
    require(saved["sample_index"] == sample_index and
            saved["planned_candidate_count"] == CANDIDATES and
            len(saved["candidates"]) == CANDIDATES,
            "stage_candidate_coverage_disagrees", location)
    candidates = saved["candidates"]
    require([row["candidate_index"] for row in candidates] == list(range(CANDIDATES)),
            "stage_candidate_indices_disagree", location)
    native_status = {row["native_status"] for row in candidates}
    if native_status == {"measured"}:
        require(all(isinstance(row["native_official_valid"], bool) and
                    row["native_official_valid"] ==
                    (row["native_reported_thresholded_overlap_percent"] == 0.)
                    for row in candidates), "native_thresholded_overlap_validity_disagrees", location)
        native = {
            "candidate_count": CANDIDATES,
            "official_valid": [row["native_official_valid"] for row in candidates],
            "utilization": [row["native_bbox_utilization"] for row in candidates],
            "bounding_boxes_xyxy": [row["native_bounds_xyxy"] for row in candidates],
            "reported_overlap_area_percent": [row["native_reported_thresholded_overlap_percent"]
                                              for row in candidates],
        }
    else:
        require(native_status == {"not_measured"}, "mixed_native_measurement_status", location)
        native = None
    references = [row["reference"] for row in candidates]
    rebuilt = sample_metrics(sample_index, native, references,
                             planned_candidate_count=CANDIDATES,
                             status=saved["status"], error=saved.get("error"))
    close_tree(saved, rebuilt, location + "/sample")
    recomputed = aggregate_metrics([rebuilt])
    close_tree(report, recomputed, location + "/aggregate")
    return rebuilt, recomputed


def geometry(polygons, ids, actions):
    """Independent GEOS geometry: affinity transforms and all polygon pairs."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon
    require(len(actions) == len(ids), "geometry_action_count_disagrees")
    shapes = []
    for poly_id, action in zip(ids, actions):
        require(len(action) == 4 and all(math.isfinite(float(x)) for x in action),
                "geometry_nonfinite_action")
        shape = Polygon(polygons[poly_id])
        shape = rotate(shape, math.atan2(action[3], action[2]), origin=(0., 0.), use_radians=True)
        shape = translate(shape, xoff=action[0], yoff=action[1])
        require(shape.is_valid and shape.area > 0, "geometry_invalid_contour")
        shapes.append(shape)
    intersections = [shape.intersection(other).area for index, shape in enumerate(shapes)
                     for other in shapes[index + 1:]]
    return {
        "bounds_xyxy": [min(p.bounds[0] for p in shapes), min(p.bounds[1] for p in shapes),
                        max(p.bounds[2] for p in shapes), max(p.bounds[3] for p in shapes)],
        "total_polygon_area": math.fsum(p.area for p in shapes),
        "max_pairwise_area": max(intersections, default=0.),
        "sum_pairwise_area": math.fsum(intersections),
    }


def audit_geometry(run, stage, report, location):
    import torch
    sample = read_json(run / "input_sample.json")
    polygons = read_json(run / "input_geometry.json")["official_max_contours"]
    ids = sample["polygon_ids"]
    require(sample["sample_index"] == report["samples"][0]["sample_index"] and
            len(ids) == 48 and all(type(index) is int and 0 <= index < len(polygons) for index in ids) and
            len(sample["mask"]) == len(ids) and
            all(value == 0 for value in sample["mask"]),
            "unexpected_dental_piece_count_or_padding", location)
    actions = torch.load(run / STAGES[stage], map_location="cpu", weights_only=True)
    require(isinstance(actions, torch.Tensor) and tuple(actions.shape) == (CANDIDATES, len(ids), 4)
            and torch.isfinite(actions).all().item(), "prediction_tensor_shape_or_finiteness", location)
    expected_dtype = torch.float64 if stage == "enhanced_float64" else torch.float32
    require(actions.dtype == expected_dtype, "prediction_tensor_storage_dtype_disagrees", location)
    candidates = report["samples"][0]["candidates"]
    measured = [row for row in candidates if row["reference"].get("status") == "measured"]
    require(bool(measured), "saved_actions_lack_reference_geometry", location)
    selected = report["criteria"]["reference_eps50_fixed_strip"]["sample_selections"][0]["best"][
        "reference_strip_utilization"]["candidate_index"]
    maximum = max(measured, key=lambda row: (row["reference"]["max_pairwise_area"],
                                            -row["candidate_index"]))["candidate_index"]
    max_sum = max(measured, key=lambda row: (row["reference"]["sum_pairwise_area"],
                                            -row["candidate_index"]))["candidate_index"]
    indices = sorted({maximum, max_sum, selected if selected is not None else 0})
    checks = []
    for index in indices:
        saved = candidates[index]["reference"]
        require(saved.get("status") == "measured", "selected_geometry_unmeasured", location)
        actual = geometry(polygons, ids, actions[index].tolist())
        maximum_error = 0.
        for key, value in actual.items():
            close_tree(value, saved[key], location + "/candidate_" + str(index) + "/" + key,
                       atol=GEOMETRY_ATOL, rtol=0.)
            if isinstance(value, list):
                maximum_error = max(maximum_error, *(abs(x-y) for x, y in zip(value, saved[key])))
            else:
                maximum_error = max(maximum_error, abs(value - saved[key]))
        checks.append({"candidate_index": index,
                       "selection_reasons": [name for name, chosen in
                                             (("best_strip", selected), ("max_pairwise_overlap", maximum),
                                              ("max_sum_overlap", max_sum),
                                              ("no_feasible_fallback", 0 if selected is None else None))
                                             if chosen == index],
                       "maximum_absolute_error": maximum_error, **actual})
    return checks


def compact_metrics(report):
    criteria = report["criteria"]
    native = criteria["native_eps50_overlap_only"]
    strip = criteria["reference_eps50_fixed_strip"]
    strict = criteria["reference_strict_fixed_strip"]
    references = [candidate["reference"] for sample in report["samples"]
                  for candidate in sample["candidates"]
                  if candidate["reference"].get("status") == "measured"]
    fractions = [row["sum_pairwise_area"] / row["total_polygon_area"] for row in references]
    return {
        "planned_candidates": report["planned_candidates"],
        "native_eps50_valid": native["feasible_candidates"],
        "reference_eps50_strip_valid": strip["feasible_candidates"],
        "reference_strict_strip_valid": strict["feasible_candidates"],
        "reference_strict_overlap_only_valid": criteria["reference_strict_overlap_only"]["feasible_candidates"],
        "unmeasured_native_candidates": native["unmeasured_candidates"],
        "unmeasured_reference_candidates": strip["unmeasured_candidates"],
        "best_native_bbox_utilization_failure_as_zero": native["utilization"]["native_bbox_utilization"][
            "mean_best_all_samples_failure_as_zero"],
        "best_strip_feasible_bbox_utilization_failure_as_zero": strip["utilization"][
            "reference_bbox_utilization"]["mean_best_all_samples_failure_as_zero"],
        "best_strip_utilization_failure_as_zero": strip["utilization"]["reference_strip_utilization"][
            "mean_best_all_samples_failure_as_zero"],
        "mean_unthresholded_pairwise_overlap_percent":
            100. * math.fsum(fractions) / len(fractions) if fractions else None,
        "overlap_measured_candidate_denominator": len(fractions),
    }


def check_sampler(run, meta, location):
    sampler = read_json(run / "sampler.json")
    profile = meta["profile"]["profile"]
    enabled = meta["constraints_enabled"]
    require(sampler["status"] == "completed" and sampler["num_steps"] == 128 and
            sampler["batch_size"] == CANDIDATES and sampler["polygon_count"] == 48 and
            sampler["model_training_flag"] is True and len(sampler["steps"]) == 128,
            "sampler_configuration_disagrees", location)
    require(sampler["profile"] == profile and sampler["seed"] == 3407 and
            sampler["constraints_enabled"] == enabled,
            "sampler_profile_or_seed_disagrees", location)
    expected_calls = 32 if enabled else 0
    require(sampler["constraint_callback_calls"] == expected_calls,
            "constraint_callback_count_disagrees", location)
    callbacks = read_json(run / "constraint_callbacks.json")
    require(len(callbacks) == expected_calls and
            [r["callback_index"] for r in callbacks] == list(range(1, expected_calls + 1)),
            "constraint_callback_log_disagrees", location)
    steps = sampler["steps"]
    require([row["step_1based"] for row in steps] == list(range(1, 129)) and
            [row["step_1based"] for row in steps if row["constraint_active"]] ==
            (list(range(97, 129)) if enabled else []), "constraint_schedule_disagrees", location)
    expected_min = .1 if profile == "paper_settings" else 1.
    for data in (sampler, meta["profile"]):
        require(data["sigma_min"] == expected_min and data["sigma_max"] == 1000. and
                data["rotation_feature_scale"] == (1000. if profile == "paper_settings" else 500.) and
                data["drift_step_multiplier"] == (14. if profile == PROFILES[0] else 1.),
                "profile_numerical_settings_disagree", location)
    post = read_json(run / "post.json")
    require(post["max_iterations"] == 100 and 0 <= post["iterations_executed"] <= 100 and
            len(post["trace"]) == post["iterations_executed"] and len(post["statuses"]) == CANDIDATES and
            len(post["gap_rollbacks_per_candidate"]) == CANDIDATES and
            post["horizontal_right_wall"] is None and post["height"] == 1200. and
            post["rotation_components_preserved"] is True,
            "post_configuration_disagrees", location)
    require(set(post["statuses"]) <= {"native_converged", "iteration_limit"} and
            [row["iteration_1based"] for row in post["trace"]] ==
            list(range(1, post["iterations_executed"] + 1)), "post_status_or_trace_disagrees", location)
    public = {
        "sampler_steps": len(steps), "constraint_callbacks": expected_calls,
        "post_iterations": post["iterations_executed"],
        "post_status_counts": {status: post["statuses"].count(status) for status in sorted(set(post["statuses"]))},
        "post_gap_rollbacks": sum(post["gap_rollbacks_per_candidate"]),
        "peak_state_max_abs": max(row["state_max_abs"] for row in steps),
        "peak_score_max_abs": max(row["score_max_abs"] for row in steps),
        "peak_repulsion_max_norm": max(row["repulsion_max_norm"] for row in steps),
        "peak_constraint_displacement_max_norm": max(row["constraint_displacement_max_norm"] for row in steps),
    }
    for key in ("elapsed_seconds", "gpu_peak_allocated_bytes", "model_parameters",
                "model_trainable_parameters", "state_dict_elements"):
        if key in meta:
            public[key] = meta[key]
    return public


def verify(results):
    plan = read_json(results / "plan.json")
    summary = read_json(results / "summary.json")
    expected_profiles = [{"name": profile, "no_constraints": index == 0}
                         for index, profile in enumerate(PROFILES)]
    require(plan.get("samples") == list(SAMPLES) and plan.get("profiles") == expected_profiles and
            plan.get("candidates") == CANDIDATES and plan.get("steps") == 128 and
            plan.get("post_iterations") == 100 and plan.get("seed") == 3407,
            "unexpected_suite_plan")
    require(summary.get("status") in ("completed", "completed_with_failures"), "suite_not_finished")
    for key in plan:
        close_tree(summary.get(key), plan[key], "summary/" + key)
    expected = [(sample, profile) for sample in SAMPLES for profile in PROFILES]
    rows = summary["rows"]
    require([(row["sample_index"], row["profile"]) for row in rows] == expected,
            "missing_duplicated_or_reordered_runs")
    output_names = [row["output"] for row in rows]
    require(len(set(output_names)) == 6, "duplicate_output_directory")
    public_rows, checkpoint_hashes, frozen_inputs, upstreams = [], set(), {}, set()
    weight_counts = set()
    stage_samples = {(profile, stage): [] for profile in PROFILES for stage in STAGES}
    counts = {"verified_output_files": 0, "verified_frozen_inputs": 0, "geometry_candidates": 0,
              "failed_run_unmanifested_artifacts": 0}
    checkpoint_paths = set()
    for row in rows:
        name = row["output"]
        require(isinstance(name, str) and Path(name).name == name and not Path(name).is_absolute(),
                "run_output_must_be_relative_directory")
        run = contained(results, name, name)
        require(row["status"] in ("completed", "failed", "timeout", "timed_out"),
                "unexpected_run_status", name)
        public = {"output": name, "sample_index": row["sample_index"], "profile": row["profile"],
                  "execution_status": row["status"], "planned_candidates": CANDIDATES,
                  "stages": {}, "geometry_checks": {}}
        for key in ("returncode", "elapsed_seconds"):
            if key in row:
                public[key] = row[key]
        meta_path = run / "metadata.json"
        meta = read_json(meta_path) if meta_path.is_file() else None
        completed = row["status"] == "completed"
        require(not completed or meta is not None, "completed_run_missing_metadata", name)
        if meta is not None:
            require(meta["sample_index"] == row["sample_index"] and
                    meta["profile"]["profile"] == row["profile"] and
                    meta["planned_candidates"] == CANDIDATES and not meta["smoke"] and
                    meta["seed"] == 3407 and meta["post_iterations"] == 100 and
                    meta["constraints_enabled"] == (row["profile"] != PROFILES[0]) and
                    meta.get("changed_inputs") == [], "run_configuration_or_frozen_inputs_disagree", name)
            require(not completed or meta["status"] == "completed", "run_completion_status_disagrees", name)
            require(meta["paper_full_reproduction"] is False and
                    meta["checkpoint_training_configuration_verified"] is False and
                    meta["architecture_boundary_encoder_matches_paper"] is False and
                    meta["paper_testset_identity_verified"] is False,
                    "unsupported_full_reproduction_claim", name)
            if completed:
                require(row.get("returncode", 0) == 0, "completed_run_has_failure_exit", name)
            for key in ("model_parameters", "state_dict_elements"):
                if key in row:
                    require(row[key] == meta.get(key), "suite_model_count_disagrees", name)
            hashes = meta.get("input_sha256", {})
            require(bool(hashes), "metadata_missing_frozen_inputs", name)
            for source, expected_hash in hashes.items():
                path = Path(source).resolve()
                if path in frozen_inputs:
                    require(frozen_inputs[path] == expected_hash, "cross_run_frozen_hash_disagrees", name)
                else:
                    require(path.is_file() and digest(path) == expected_hash, "live_frozen_input_changed", name)
                    frozen_inputs[path] = expected_hash
                    counts["verified_frozen_inputs"] += 1
                if path.name == "score_model.pth":
                    checkpoint_paths.add(path)
                if path.name == "score_model.py" and path.parent.name == "src":
                    upstreams.add(path.parent.parent)
            checkpoint_hashes.add(meta["checkpoint_sha256"])
            if "model_parameters" in meta:
                weight_counts.add((meta["model_parameters"], meta.get("model_trainable_parameters"),
                                   meta["state_dict_elements"]))
            outputs = meta.get("output_sha256", {})
            require(bool(outputs), "metadata_missing_output_hashes", name)
            for relative, expected_hash in outputs.items():
                path = contained(run, relative, name)
                require(path.is_file() and digest(path) == expected_hash, "saved_output_hash_disagrees", name)
                counts["verified_output_files"] += 1
            actual_names = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file() and path != meta_path}
            require(actual_names == set(outputs), "output_hash_manifest_coverage_disagrees", name)
            builds = [path for path in hashes if Path(path).name == "build.json"]
            require(len(builds) == 1, "missing_or_ambiguous_native_build_manifest", name)
            build_path = Path(builds[0]).resolve()
            build = read_json(build_path)
            require(build.get("paper_constraints") is True and build.get("source_unchanged") is True and
                    build.get("upstream_commit") == COMMIT and build.get("returncode") == 0,
                    "native_build_configuration_disagrees", name)
            binaries = [Path(path).resolve() for path in hashes
                        if Path(path).name.startswith("collision_probe") and Path(path).suffix in (".so", ".pyd")]
            require(len(binaries) == 1 and binaries[0].parent == build_path.parent and
                    digest(binaries[0]) == build["extension_sha256"], "native_extension_binding_disagrees", name)
            for relative, key in (("source/main.cpp", "instrumented_main_sha256"),
                                  ("source/collision_repair.hpp", "repair_header_sha256"),
                                  ("source/paper_constraints.hpp", "paper_header_sha256"),
                                  ("instrumentation.patch", "patch_sha256"),
                                  ("repair.patch", "repair_patch_sha256"), ("paper.patch", "paper_patch_sha256")):
                path = build_path.parent / relative
                require(path.is_file() and digest(path) == build[key], "native_build_source_hash_disagrees", name)
            require(len(upstreams) == 1, "missing_or_ambiguous_upstream", name)
            upstream = next(iter(upstreams))
            for relative, expected_hash in build["source_sha256"].items():
                path = contained((upstream / "external/rmspacing_cpp").resolve(), relative, name)
                require(path.is_file() and digest(path) == expected_hash, "pinned_native_source_changed", name)
            if completed:
                public.update(check_sampler(run, meta, name))
        else:
            # A killed process may never execute its finally block. Such outputs
            # are retained but cannot be promoted to verified measured results.
            require(not completed and row.get("returncode", 1) != 0,
                    "metadata_missing_without_failed_execution", name)
            counts["failed_run_unmanifested_artifacts"] += sum(path.is_file() for path in run.rglob("*"))
            public["output_manifest_status"] = "unavailable_after_failed_execution"
        failure_report = None
        if (run / "failed_metrics.json").is_file() and meta is not None:
            _, failure_report = verify_stage(read_json(run / "failed_metrics.json"), row["sample_index"],
                                            name + "/failed_metrics")
        for stage in STAGES:
            metric_path = run / (stage + "_metrics.json")
            if meta is not None and metric_path.is_file():
                saved, report = verify_stage(read_json(metric_path), row["sample_index"], name + "/" + stage)
                require((run / STAGES[stage]).is_file(), "measured_stage_missing_actions", name + "/" + stage)
                checks = audit_geometry(run, stage, report, name + "/" + stage)
                counts["geometry_candidates"] += len(checks)
                public["geometry_checks"][stage] = checks
                if completed:
                    require(saved["status"] == "completed", "completed_run_has_failed_stage", name + "/" + stage)
            else:
                require(not completed, "completed_run_missing_stage_metrics", name + "/" + stage)
                saved = sample_metrics(row["sample_index"], None, planned_candidate_count=CANDIDATES,
                                       status="failed", error={"type": "FailedExecution", "message": "Unmeasured stage"})
                report = aggregate_metrics([saved])
                public["geometry_checks"][stage] = []
            stage_samples[(row["profile"], stage)].append(saved)
            public["stages"][stage] = compact_metrics(report)
        public_rows.append(public)
    require(len(checkpoint_hashes) == 1 and len(checkpoint_paths) == 1 and len(upstreams) == 1 and
            len(weight_counts) == 1, "shared_checkpoint_or_model_provenance_unconfirmed")
    checkpoint = next(iter(checkpoint_paths))
    require(digest(checkpoint) == next(iter(checkpoint_hashes)), "checkpoint_manifest_binding_disagrees")
    upstream = next(iter(upstreams))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, timeout=15)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=upstream,
                            capture_output=True, text=True, timeout=15)
    require(head.returncode == 0 and head.stdout.strip() == COMMIT and
            status.returncode == 0 and not status.stdout.strip(), "upstream_not_clean_pinned_source")
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and all(isinstance(value, torch.Tensor) for value in state.values()),
            "checkpoint_is_not_tensor_state_dict")
    state_elements = sum(value.numel() for value in state.values())
    sys.path.insert(0, str(upstream))
    module = importlib.import_module("src.score_model")
    require(Path(module.__file__).resolve() == upstream / "src/score_model.py", "wrong_model_module_loaded")
    model = module.PolygonPackingTransformer(
        marginal_prob_std_func=lambda x, t: (x, 1000. ** t), device=torch.device("cpu"))
    model.load_state_dict(state, strict=True)
    actual_counts = (sum(value.numel() for value in model.parameters()),
                     sum(value.numel() for value in model.parameters() if value.requires_grad), state_elements)
    require(actual_counts == next(iter(weight_counts)), "independent_weight_parameter_counts_disagree")
    del model, state
    profiles = {}
    for profile in PROFILES:
        profiles[profile] = {}
        for stage in STAGES:
            report = aggregate_metrics(stage_samples[(profile, stage)])
            require(report["planned_candidates"] == 1024 and report["planned_samples"] == 2,
                    "profile_aggregate_lost_failed_candidates")
            profiles[profile][stage] = compact_metrics(report)
    total_per_stage = {stage: sum(public["stages"][stage]["planned_candidates"] for public in public_rows)
                       for stage in STAGES}
    require(all(value == 3072 for value in total_per_stage.values()), "suite_lost_planned_candidates")
    completed_count = sum(row["status"] == "completed" for row in rows)
    return {
        "status": "verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "paper_full_reproduction": False, "scope": "Integrity and numerical evidence only; two development samples, not an independent benchmark",
        "planned_runs": 6, "completed_runs": completed_count, "failed_runs_retained": 6-completed_count,
        "planned_candidates_per_stage": total_per_stage, "source_clean_and_pinned": True,
        "upstream_commit": COMMIT, "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "model_parameters": actual_counts[0], "model_trainable_parameters": actual_counts[1],
        "state_dict_elements": actual_counts[2], "geometry_absolute_tolerance": GEOMETRY_ATOL,
        "geometry_method": "CPU tensors loaded with weights_only; independent GEOS affinity transforms and all-pairs intersections",
        "units": {"utilization": "fraction", "unthresholded_overlap": "percent of summed polygon area",
                  "strip_height": 1200., "horizontal_right_wall": None},
        "failure_policy": "All failed/unmeasured runs retain 512 candidates per stage; unmanifested remnants are excluded from measured results",
        "strict_tolerance_is_local_audit_convention": True,
        "rotation_scale_scope": "GT encoding metadata; generated directions are not rescaled or reprojected",
        **counts, "profiles": profiles, "runs": public_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New public JSON file; never overwrites")
    args = parser.parse_args()
    results, output = args.results.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Verification output must be new")
    exit_code = 0
    try:
        result = verify(results)
    except VerificationError as exc:
        result = {"status": "failed", "paper_full_reproduction": False,
                  "error_code": exc.code, "evidence_location": exc.location}
        exit_code = 1
    except Exception as exc:
        # Do not export exception messages, tracebacks, account names or paths.
        result = {"status": "failed", "paper_full_reproduction": False,
                  "error_code": "verification_runtime_error", "error_type": type(exc).__name__}
        exit_code = 1
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "planned_runs", "completed_runs", "failed_runs_retained",
                                                  "geometry_candidates", "error_code", "error_type") if key in result}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
