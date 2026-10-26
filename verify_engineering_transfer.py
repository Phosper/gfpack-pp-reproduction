#!/usr/bin/env python3
"""Independently verify the frozen 128 x 1 engineering transfer experiment.

No optimizer, inference model, native extension, or training code is imported.
Endpoints use affine Shapely reconstruction and struct float32 conversion;
accepted moves from BOTH distinct starts are replayed against every contour.
The report contains measurements and identities, never actions or contours.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import verify_compact_layouts as independent
from verify_compact_portfolio import replay_pipeline

require = independent.require
read = independent.read
sha = independent.sha
close = independent.close
geometry = independent.geometry
check_geometry = independent.check_geometry
fp32 = independent.fp32
actions_checked = independent.actions_checked
EPS = independent.EPS
ATOL = independent.GEOMETRY_ATOL
HEIGHT = independent.HEIGHT
SOURCES = {"pred": "raw_actions", "gt": "gt_actions"}
TOTAL_SAMPLES = 128
PLANNED_INPUTS = 256
STATUSES = ("completed", "execution_failed", "timeout")
ENDPOINTS = ("raw", "enhanced", "raw_one_start", "seed", "selected")
REQUIRED_SCRIPTS = (
    "freeze_engineering_transfer.py", "run_engineering_transfer.py",
    "prepare_transfer_seeds.py", "compact_portfolio.py", "compact_layout.py",
    "translation_intervals.py", "postprocess_metrics.py",
    "check_collision_saved_cases.py", "verify_engineering_transfer.py",
    "verify_compact_layouts.py", "verify_compact_portfolio.py",
)


def finite_nonnegative(value, location):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid_elapsed_seconds", location)
    return value


def endpoint_geometries(job):
    """Only completed jobs may contribute a selected endpoint to aggregates."""
    baseline = job.get("baseline_geometry") or {}
    answer = {"raw": baseline.get("raw"), "enhanced": baseline.get("enhanced"),
              "raw_one_start": None, "seed": None, "selected": None}
    if job["status"] == "completed":
        attempts = job["portfolio"]["attempts"]
        answer.update(raw_one_start=attempts[0]["after"], seed=attempts[1]["after"],
                      selected=job["after"])
    return answer


def endpoint_summary(values, denominator):
    measured = [value for value in values if value is not None]
    result = {
        "denominator": denominator, "measured_outputs": len(measured),
        "missing_or_failed_outputs": denominator-len(measured),
        "strip_utilization_measured": independent.stats(
            [value["strip_utilization"] for value in measured]),
        "span_x_measured": independent.stats([value["span_x"] for value in measured]),
        "max_pairwise_area_measured": independent.stats(
            [value["max_pairwise_area"] for value in measured]),
        "height_violation_measured": independent.stats(
            [value["height_violation"] for value in measured]),
    }
    for name in ("strict_strip_valid", "eps50_strip_valid", "within_strip"):
        count = sum(bool(value[name]) for value in measured)
        result[name] = count
        result[name+"_rate"] = count/denominator if denominator else None
    for label, flag in (("strict", "strict_strip_valid"),
                        ("area50", "eps50_strip_valid")):
        valid = [value["strip_utilization"] for value in measured if value[flag]]
        result[label+"_valid_strip_utilization"] = independent.stats(valid)
        result["mean_"+label+"_valid_utilization_failures_zero"] = (
            math.fsum(valid)/denominator if denominator else None)
    return result


def paired_metrics(selected, reference, identities):
    denominator = len(identities)
    paired = [(i, a, b) for i, (a, b) in enumerate(zip(selected, reference))
              if a is not None and b is not None]
    deltas = [(a["strip_utilization"]-b["strip_utilization"], identities[i])
              for i, a, b in paired]
    answer = {
        "denominator": denominator, "available_pairs": len(paired),
        "missing_or_failed_pairs": denominator-len(paired),
        "all_available_strip_utilization_delta": independent.stats(
            [delta for delta, _ in deltas]),
        "worst_available_strip_utilization_delta": (
            {"delta": min(deltas, key=lambda item: (item[0], item[1]))[0],
             "sample_index": min(deltas, key=lambda item: (item[0], item[1]))[1]}
            if deltas else None),
    }
    for label, flag in (("strict", "strict_strip_valid"),
                        ("area50", "eps50_strip_valid")):
        joint = [(a, b) for _, a, b in paired if a[flag] and b[flag]]
        losses = [identities[i] for i, (a, b) in enumerate(zip(selected, reference))
                  if b is not None and b[flag] and (a is None or not a[flag])]
        gains = [identities[i] for i, (a, b) in enumerate(zip(selected, reference))
                 if a is not None and a[flag] and (b is None or not b[flag])]
        selected_sum = math.fsum(a["strip_utilization"] for a in selected
                                 if a is not None and a[flag])
        reference_sum = math.fsum(b["strip_utilization"] for b in reference
                                  if b is not None and b[flag])
        answer[label] = {
            "joint_valid_pairs": len(joint),
            "joint_valid_strip_utilization_delta": independent.stats(
                [a["strip_utilization"]-b["strip_utilization"] for a, b in joint]),
            "validity_lost_samples": losses, "validity_gained_samples": gains,
            "mean_valid_utilization_failures_zero": {
                "selected": selected_sum/denominator if denominator else None,
                "reference": reference_sum/denominator if denominator else None,
                "delta": (selected_sum-reference_sum)/denominator if denominator else None,
            },
        }
    return answer


def gt_regression(group):
    """Report efficacy regressions without conflating them with bad evidence."""
    failures, strict, area50, boundary, width = [], [], [], [], []
    records = []
    for job in group:
        baseline = endpoint_geometries(job)["raw"]
        selected = endpoint_geometries(job)["selected"]
        index = job["sample_index"]
        if selected is None:
            failures.append(index)
        flags = {
            "strict_validity_lost": bool(baseline is not None and
                baseline["strict_strip_valid"] and
                (selected is None or not selected["strict_strip_valid"])),
            "area50_validity_lost": bool(baseline is not None and
                baseline["eps50_strip_valid"] and
                (selected is None or not selected["eps50_strip_valid"])),
            "boundary_validity_lost": bool(baseline is not None and
                baseline["within_strip"] and
                (selected is None or not selected["within_strip"])),
            "width_increased": bool(baseline is not None and selected is not None and
                selected["span_x"] > baseline["span_x"]+ATOL),
        }
        for name, indices in (("strict_validity_lost", strict),
                              ("area50_validity_lost", area50),
                              ("boundary_validity_lost", boundary),
                              ("width_increased", width)):
            if flags[name]:
                indices.append(index)
        records.append({"sample_index": index, **flags,
            "selected_missing": selected is None,
            "span_delta": selected["span_x"]-baseline["span_x"]
                if selected is not None and baseline is not None else None})
    return {
        "denominator": len(group), "failed_samples": failures,
        "strict_validity_lost_samples": strict, "area50_validity_lost_samples": area50,
        "boundary_validity_lost_samples": boundary, "width_increased_samples": width,
        "width_comparison_absolute_tolerance": ATOL,
        "all_original_strict_validity_preserved": not strict,
        "all_original_area50_validity_preserved": not area50,
        "all_original_boundary_validity_preserved": not boundary,
        "all_measured_widths_nonincreasing": not width,
        "samples": records,
    }


def aggregate_checked_jobs(jobs, development_samples):
    """Shared arithmetic only; the verifier supplies independently measured jobs.

    Every input, including execution failures and timeouts, must remain in jobs.
    The returned object is the complete value of summary.json['aggregates'].
    This function imports no solver and is safe for a runner to import.
    """
    development = set(development_samples)
    ordered = sorted(jobs, key=lambda job: (job["sample_index"], job["source"]))
    identities = [(job["sample_index"], job["source"]) for job in ordered]
    require(len(set(identities)) == len(identities), "duplicate_aggregate_input")
    require(all(job["status"] in STATUSES and job["source"] in SOURCES
                for job in ordered), "invalid_aggregate_input")
    groups = []
    for scope in ("all", "development", "transfer"):
        for source in SOURCES:
            group = [job for job in ordered if job["source"] == source and
                     (scope == "all" or
                      (job["sample_index"] in development) == (scope == "development"))]
            n = len(group)
            endpoints = [endpoint_geometries(job) for job in group]
            selected = [row["selected"] for row in endpoints]
            state = Counter(job["status"] for job in group)
            timing = {}
            for key in ("seed_generation", "raw_solver", "distinct_seed_solver",
                        "solver_total", "total_with_seed_generation"):
                values = [job["timing_seconds"][key] for job in group
                          if isinstance(job.get("timing_seconds"), dict) and
                          type(job["timing_seconds"].get(key)) in (int, float)]
                timing[key] = {"measured_inputs": len(values),
                               "total_seconds": math.fsum(values),
                               "seconds": independent.stats(values)}
            result = {
                "group": scope, "source": source, "planned_inputs": n,
                "completed_inputs": state["completed"],
                "execution_failed_inputs": state["execution_failed"],
                "timeout_inputs": state["timeout"],
                "selected_start_counts": dict(sorted(Counter(
                    str(job["portfolio"]["selected_start"]) for job in group
                    if job["status"] == "completed").items())),
                "endpoints": {name: endpoint_summary([row[name] for row in endpoints], n)
                              for name in ENDPOINTS},
                "paired_same_candidate": {name: paired_metrics(selected,
                    [row[name] for row in endpoints], [job["sample_index"] for job in group])
                    for name in ENDPOINTS if name != "selected"},
                "timing_seconds": timing,
            }
            if source == "gt":
                result["gt_regression"] = gt_regression(group)
            groups.append(result)
    state = Counter(job["status"] for job in ordered)
    return {"schema_version": 1, "planned_inputs": len(ordered),
            "completed_inputs": state["completed"],
            "execution_failed_inputs": state["execution_failed"],
            "timeout_inputs": state["timeout"],
            "development_samples": sorted(development),
            "denominator_policy": "Every planned source remains in its denominator; missing, failed and invalid selected layouts contribute zero valid utilization",
            "by_source_group": groups}


def inventory(root, label):
    metadata = read(root/"metadata.json")
    require(metadata["status"] == "completed" and metadata["changed_inputs"] == [],
            "incomplete_or_changed_inputs", label)
    files = {path.relative_to(root).as_posix() for path in root.rglob("*")
             if path.is_file() and path != root/"metadata.json"}
    recorded = metadata["output_sha256"]
    require(isinstance(recorded, dict) and files == set(recorded),
            "output_inventory_incomplete", label)
    for relative, digest in recorded.items():
        require(sha(independent.contained(root, relative)) == digest,
                "output_hash_disagrees", label+"/"+relative)
    return metadata


def check_seed_bundle(seedroot, frozen, manifest, metadata):
    inputs = read(seedroot/"inputs.json")
    require(isinstance(inputs, dict) and metadata["input_count"] == len(inputs),
            "seed_input_inventory_disagrees")
    portable = metadata.get("portable_source_paths", {})
    source_paths = metadata.get("source_paths", {})
    require(isinstance(portable, dict) and isinstance(source_paths, dict),
            "seed_source_mapping_missing")
    require(set(portable).issubset(inputs) and set(source_paths) == set(inputs),
            "seed_source_mapping_coverage")
    require(all(isinstance(value, str) and value for value in source_paths.values()),
            "seed_source_path_record_invalid")
    expected_frozen = {"frozen/manifest.json": sha(frozen/"manifest.json")}
    expected_frozen.update({"frozen/"+entry["path"]: entry["sha256"]
                            for entry in manifest["cases"]})
    require(set(expected_frozen).issubset(inputs), "seed_frozen_inputs_incomplete")
    extension_hashes = set()
    worker_bound = False
    for key, digest in inputs.items():
        # Even when the original frozen file is locally available, any copied
        # portable snapshot must describe precisely those same input bytes.
        if key in portable:
            relative = portable[key]
            require(relative in metadata["output_sha256"],
                    "portable_seed_source_not_in_output_inventory", key)
            snapshot = independent.contained(seedroot, relative)
            require(sha(snapshot) == digest, "portable_seed_snapshot_disagrees", key)
        if key.startswith("frozen/"):
            path = independent.contained(frozen, key.split("/", 1)[1])
        else:
            require(key in portable, "nonportable_seed_source", key)
            relative = portable[key]
            require(relative in metadata["output_sha256"],
                    "portable_seed_source_not_in_output_inventory", key)
            path = independent.contained(seedroot, relative)
        require(sha(path) == digest, "seed_input_hash_disagrees", key)
        if key in expected_frozen:
            require(digest == expected_frozen[key], "seed_not_same_frozen_input", key)
        if key.lower().endswith((".so", ".pyd")):
            extension_hashes.add(digest)
        if key == "scripts/prepare_transfer_seeds.py":
            require(digest == sha(Path(__file__).with_name("prepare_transfer_seeds.py")),
                    "seed_worker_source_changed")
            worker_bound = True
    require(extension_hashes and worker_bound, "seed_worker_or_extension_not_bound")
    require("probe/build.json" in portable, "seed_native_build_manifest_missing")
    build = read(independent.contained(seedroot, portable["probe/build.json"]))
    require(build.get("strip_repair_entry_point") == "collision_probe.overlap_repair_strip"
            and build.get("returncode") == 0 and build.get("source_unchanged") is True
            and build.get("extension_sha256") in extension_hashes,
            "seed_native_build_policy_disagrees")
    plan = read(seedroot/"plan.json")
    expected_plan = {"planned_inputs": PLANNED_INPUTS, "height": HEIGHT, "epsilon": 1.,
        "maximum_iterations": 2000, "sampling": False, "training": False,
        "manifest_sha256": sha(frozen/"manifest.json"), "same_candidate_only": True}
    for key, value in expected_plan.items():
        require(key in plan, "seed_plan_field_missing", key)
        close(plan[key], value, "seed_plan/"+key)
    require(plan.get("extension_sha256") in extension_hashes,
            "seed_plan_extension_hash_disagrees")
    return inputs, extension_hashes


def check_inputs(root, frozen, seedroot, manifest, metadata, seedmeta):
    inputs = read(root/"inputs.json")
    scriptroot = Path(__file__).resolve().parent
    require(isinstance(inputs, dict), "input_inventory_not_dictionary")
    required_scripts = {"scripts/"+name for name in REQUIRED_SCRIPTS}
    require(required_scripts.issubset(inputs), "core_script_hash_inventory_incomplete")
    expected = {"seeds/metadata.json": sha(seedroot/"metadata.json")}
    # The frozen evaluation contract and original development manifest are
    # inputs as well as cases; require the entire portable freeze to be bound.
    expected.update({"frozen/"+path.relative_to(frozen).as_posix(): sha(path)
                     for path in frozen.rglob("*") if path.is_file()})
    expected.update({"seeds/"+relative: digest
                     for relative, digest in seedmeta["output_sha256"].items()})
    for key in inputs:
        if key.startswith("scripts/"):
            expected[key] = sha(independent.contained(scriptroot, key.split("/", 1)[1]))
    require(inputs == expected and metadata["input_count"] == len(expected),
            "runner_input_inventory_disagrees")
    for implementation in manifest.get("implementation_files", []):
        key = "scripts/"+implementation["path"]
        if key in inputs:
            require(inputs[key] == implementation["sha256"],
                    "implementation_changed_since_freeze", key)
    return inputs


def check_plan(plan, development):
    expected = {
        "samples": list(range(TOTAL_SAMPLES)), "sources": SOURCES,
        "development_samples": sorted(development), "planned_inputs": PLANNED_INPUTS,
        "height": HEIGHT, "horizontal_walls": None,
        "strict_area_tolerance": EPS, "boundary_tolerance": EPS,
        "reference_area50": 50., "seconds_per_attempt": 45.,
        "sweeps_per_attempt": 4, "sampling": False, "training": False,
    }
    for key, value in expected.items():
        require(key in plan, "plan_field_missing", key)
        close(plan[key], value, "plan/"+key)
    require(isinstance(plan.get("seed_directory"), str) and
            Path(plan["seed_directory"]).is_absolute(), "seed_directory_not_absolute")


def checked_seed(job, original, rawcast, case, entry, source, seedroot,
                 seedmeta, extension_hashes, location):
    relative = "jobs/sample_%03d_%s.json" % (entry["sample_index"], source)
    require(relative in seedmeta["output_sha256"], "seed_job_not_in_inventory", location)
    seed = read(seedroot/relative)
    require(job.get("seed_sha256") == seedmeta["output_sha256"][relative] and
            job.get("seed") == seed, "embedded_seed_not_same_hashed_seed", location)
    for key, value in (("sample_index", entry["sample_index"]), ("source", source),
                       ("candidate_index", entry["candidate_index"]),
                       ("case_sha256", entry["sha256"])):
        require(seed.get(key) == value, "seed_identity_disagrees", location+"/"+key)
    require(seed["input_actions"] == original, "seed_not_same_frozen_candidate", location)
    actions_checked(seed["repair_actions"], len(case["polygon_ids"]), location+"/seed")
    require(all(a[2:] == b[2:] for a, b in zip(seed["repair_actions"], original)),
            "seed_changed_original_rotation", location)
    cast = fp32(seed["repair_actions"])
    actions_checked(cast, len(case["polygon_ids"]), location+"/seed_cast",
                    stored=True, rotations=rawcast)
    finite_nonnegative(seed["elapsed_seconds"], location+"/seed_elapsed")
    require(seed["extension_sha256"] in extension_hashes,
            "seed_extension_not_bound_to_portable_snapshot", location)
    require(isinstance(seed.get("native_result"), dict), "seed_native_result_missing", location)
    return seed, cast


def check_completed(case, original, rawcast, seed, seedcast, job, location, counts):
    raw = job["raw_result"]
    raw_geometry, _, checked = replay_pipeline(case, original, raw, location+"/raw")
    counts.update(checked)
    counts["raw_pipelines_replayed"] += 1
    portfolio = job["portfolio"]
    duplicated = seedcast == rawcast
    require(type(portfolio["seed_reused_raw_result"]) is bool and
            portfolio["seed_reused_raw_result"] == duplicated,
            "duplicate_seed_reuse_disagrees", location)
    if duplicated:
        require(portfolio["seed_result"] is None, "duplicate_seed_was_recomputed", location)
        seed_result, seed_geometry = raw, raw_geometry
        counts["duplicate_starts_reused"] += 1
    else:
        seed_result = portfolio["seed_result"]
        require(isinstance(seed_result, dict), "distinct_seed_result_missing", location)
        seed_geometry, _, checked = replay_pipeline(case, seed["repair_actions"],
            seed_result, location+"/distinct_seed")
        counts.update(checked)
        counts["distinct_seed_pipelines_replayed"] += 1
    attempts = portfolio["attempts"]
    require(isinstance(attempts, list) and len(attempts) == 2 and
            [attempt["start"] for attempt in attempts] == ["raw", "legacy_repair"],
            "attempt_coverage_or_order_disagrees", location)
    for attempt, result, measured in zip(attempts, (raw, seed_result),
                                         (raw_geometry, seed_geometry)):
        check_geometry(attempt["after"], measured, location+"/attempt_"+attempt["start"])
        require(attempt["repair_status"] == result["repair_status"] and
                attempt["compaction_status"] == result["compaction_status"],
                "attempt_status_disagrees", location)
        require(result.get("storage_precision") == "float32_at_every_accepted_endpoint",
                "pipeline_storage_policy_disagrees", location)
    feasible = [i for i, measured in enumerate((raw_geometry, seed_geometry))
                if measured["strict_strip_valid"]]
    chosen = min(feasible, key=lambda i: (attempts[i]["after"]["span_x"], i)) if feasible else None
    expected_start = ("raw", "legacy_repair")[chosen] if chosen is not None else None
    require(portfolio["selected_start"] == expected_start and
            portfolio["status"] == ("strict_feasible" if chosen is not None
                                     else "no_strict_feasible_attempt"),
            "selection_not_strict_minimum_span", location)
    selected_result = raw if chosen in (None, 0) else seed_result
    output = portfolio["actions"]
    require(output == selected_result["actions"], "selected_actions_not_chosen_attempt", location)
    actions_checked(output, len(case["polygon_ids"]), location+"/selected",
                    stored=True, rotations=rawcast)
    require(fp32(output) == output, "second_float32_cast_changes_selected_output", location)
    require(job.get("stored_float32") is True,
            "recorded_float32_storage_assertion_missing_or_false", location)
    selected = geometry(case, output, location+"/selected")
    check_geometry(job["after"], selected, location+"/selected")
    if chosen is not None:
        require(selected["strict_strip_valid"] and selected["span_x"] <=
                min((raw_geometry, seed_geometry)[i]["span_x"] for i in feasible)+ATOL,
                "selected_not_independent_feasible_minimum", location)
    if raw_geometry["strict_strip_valid"]:
        require(selected["strict_strip_valid"] and
                selected["span_x"] <= raw_geometry["span_x"]+ATOL,
                "portfolio_regressed_feasible_raw_attempt", location)
    solver_seconds = raw["elapsed_seconds"]+(0. if duplicated else seed_result["elapsed_seconds"])
    close(portfolio["new_solver_seconds_without_legacy_seed_generation"], solver_seconds,
          location+"/portfolio_solver_seconds", 1e-9)
    expected_timing = {
        "seed_generation": seed["elapsed_seconds"], "raw_solver": raw["elapsed_seconds"],
        "distinct_seed_solver": 0. if duplicated else seed_result["elapsed_seconds"],
        "solver_total": solver_seconds,
        "total_with_seed_generation": solver_seconds+seed["elapsed_seconds"],
    }
    for key, value in expected_timing.items():
        require(key in job["timing_seconds"], "timing_field_missing", location+"/"+key)
        finite_nonnegative(job["timing_seconds"][key], location+"/"+key)
        close(job["timing_seconds"][key], value, location+"/timing/"+key, 1e-9)
    before = geometry(case, original, location+"/before")
    compared = independent.comparison(before, selected, original, output)
    close(job["comparison"], compared, location+"/comparison", ATOL)
    checked_job = dict(job)
    checked_portfolio = dict(portfolio)
    checked_portfolio["attempts"] = [dict(attempt, after=measured) for attempt, measured
                                    in zip(attempts, (raw_geometry, seed_geometry))]
    checked_job.update(portfolio=checked_portfolio, after=selected, comparison=compared)
    counts["stored_float32_outputs_checked"] += 1
    counts["rotation_invariance_after_cast_checked"] += 1
    return checked_job, {"raw_one_start": independent.brief(raw_geometry),
                         "seed": independent.brief(seed_geometry),
                         "selected": independent.brief(selected),
                         "selected_start": expected_start, "duplicate_start": duplicated}


def verify(args):
    from freeze_engineering_transfer import validate_transfer_frozen, checked_transfer_case

    frozen, root = args.frozen.resolve(), args.results.resolve()
    manifest, manifest_sha = validate_transfer_frozen(frozen)
    development = manifest["development_samples"]
    require(sorted(development) == list(independent.SAMPLES), "development_sample_set_changed")
    entries = manifest["cases"]
    require(len(entries) == TOTAL_SAMPLES and
            sorted(entry["sample_index"] for entry in entries) == list(range(TOTAL_SAMPLES)),
            "frozen_128_case_coverage_incomplete")
    metadata = inventory(root, "results")
    plan, summary = read(root/"plan.json"), read(root/"summary.json")
    check_plan(plan, development)
    seedroot = Path(plan["seed_directory"]).resolve()
    seedmeta = inventory(seedroot, "seeds")
    seedinputs, extension_hashes = check_seed_bundle(seedroot, frozen, manifest, seedmeta)
    inputs = check_inputs(root, frozen, seedroot, manifest, metadata, seedmeta)
    paths = {"jobs/sample_%03d_%s.json" % (entry["sample_index"], source)
             for entry in entries for source in SOURCES}
    require({path for path in metadata["output_sha256"] if path.startswith("jobs/")} == paths,
            "job_file_coverage_incomplete")
    require({path for path in seedmeta["output_sha256"] if path.startswith("jobs/")} == paths,
            "seed_file_coverage_incomplete")
    require(summary["status"] == "completed" and summary["planned_inputs"] == PLANNED_INPUTS
            and summary["completed_inputs"] == len(summary["jobs"]) == PLANNED_INPUTS,
            "summary_input_denominator_incomplete")
    indexed = {job["path"]: job["sha256"] for job in summary["jobs"]}
    require(len(indexed) == PLANNED_INPUTS and set(indexed) == paths and
            all(metadata["output_sha256"][path] == digest for path, digest in indexed.items()),
            "summary_job_hash_coverage_disagrees")
    jobs, samples, counts = [], [], Counter()
    for entry in sorted(entries, key=lambda row: row["sample_index"]):
        _, case = checked_transfer_case(frozen, entry)
        index, n = entry["sample_index"], len(case["polygon_ids"])
        group = "development" if index in development else "transfer"
        for source, field in SOURCES.items():
            location = "sample_%03d_%s" % (index, source)
            relative = "jobs/"+location+".json"
            job = read(root/relative)
            for key, value in (("sample_index", index), ("source", source),
                               ("input_field", field), ("candidate_index", entry["candidate_index"]),
                               ("case_sha256", entry["sha256"])):
                require(job.get(key) == value, "job_identity_disagrees", location+"/"+key)
            require(job["status"] in STATUSES, "unknown_job_status", location)
            if "group" in job:
                require(job["group"] == group, "job_group_disagrees", location)
            original, rawcast = case[field], fp32(case[field])
            actions_checked(original, n, location+"/original")
            actions_checked(rawcast, n, location+"/raw_cast", stored=True)
            before = geometry(case, original, location+"/original")
            raw_geometry = before
            enhanced_field = "enhanced_actions" if source == "pred" else "gt_enhanced_actions"
            require(enhanced_field in case, "frozen_enhanced_baseline_missing", location)
            enhanced = geometry(case, case[enhanced_field], location+"/enhanced")
            baseline = job["baseline_geometry"]
            check_geometry(baseline["raw"], raw_geometry, location+"/baseline_raw")
            check_geometry(baseline["enhanced"], enhanced, location+"/baseline_enhanced")
            check_geometry(job["before"], before, location+"/before")
            seed, seedcast = checked_seed(job, original, rawcast, case, entry, source,
                seedroot, seedmeta, extension_hashes, location)
            if job["status"] == "completed":
                checked_job, measured = check_completed(case, original, rawcast, seed,
                    seedcast, job, location, counts)
                counts["completed_selected_geometries"] += 1
            else:
                require(job.get("after") is None and job.get("comparison") is None,
                        "failed_job_claims_selected_geometry", location)
                require(job.get("portfolio") is None,
                        "failed_job_claims_completed_portfolio", location)
                require(isinstance(job.get("error"), (str, dict)),
                        "failed_job_missing_explicit_error", location)
                if isinstance(job.get("timing_seconds"), dict):
                    for key, value in job["timing_seconds"].items():
                        if value is not None:
                            finite_nonnegative(value, location+"/timing/"+key)
                checked_job, measured = dict(job), {"selected": None}
                counts["failed_inputs_retained"] += 1
            checked_job["before"] = before
            checked_job["baseline_geometry"] = {"raw": raw_geometry, "enhanced": enhanced}
            jobs.append(checked_job)
            counts["seed_sources_checked"] += 1
            counts["baseline_geometries_checked"] += 2
            samples.append({"sample_index": index, "source": source, "group": group,
                "candidate_index": entry["candidate_index"], "status": job["status"],
                "references": {"raw": independent.brief(raw_geometry),
                               "enhanced": independent.brief(enhanced)}, **measured})
            print(json.dumps({"verified": location, "status": job["status"],
                              "selected_start": measured.get("selected_start")}), flush=True)
    require(len(jobs) == PLANNED_INPUTS and
            Counter(job["source"] for job in jobs) == {"pred": 128, "gt": 128},
            "source_denominators_incomplete")
    aggregates = aggregate_checked_jobs(jobs, development)
    close(summary["aggregates"], aggregates, "summary/aggregates", ATOL)
    return {
        "status": "verified", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha(Path(__file__).resolve()),
        "scope": "128 frozen samples, one preselected saved candidate each, paired prediction and GT; 16 development and 112 engineering transfer samples",
        "independence": "No optimizer or native extension imported. Both distinct start ledgers replayed with affine geometry; every accepted move checked against all contours and strip bounds; all endpoints independently remeasured",
        "hashes": {"frozen_manifest_sha256": manifest_sha,
            "results_metadata_sha256": sha(root/"metadata.json"),
            "results_inputs_sha256": sha(root/"inputs.json"),
            "seed_metadata_sha256": sha(seedroot/"metadata.json"),
            "seed_inputs_sha256": sha(seedroot/"inputs.json"),
            "portable_extension_sha256": sorted(extension_hashes),
            "scripts": {key: digest for key, digest in inputs.items() if key.startswith("scripts/")}},
        "coverage": {"planned_inputs": PLANNED_INPUTS, "predictions": 128, "gt": 128,
            "development_samples": len(development), "transfer_samples": TOTAL_SAMPLES-len(development),
            "development_inputs": 2*len(development),
            "transfer_inputs": 2*(TOTAL_SAMPLES-len(development)),
            "result_files_hashed": len(metadata["output_sha256"]),
            "seed_files_hashed": len(seedmeta["output_sha256"]),
            "runner_inputs_hashed": len(inputs), "seed_inputs_hashed": len(seedinputs),
            "all_denominators_retained": True},
        "thresholds": {"height": HEIGHT, "horizontal_walls": None,
            "strict_area": EPS, "boundary": EPS, "reference_area50": 50.,
            "metric_comparison_absolute_tolerance": ATOL},
        "ledger_checks": dict(sorted(counts.items())), "aggregates": aggregates,
        "gt_regression": gt_regression([job for job in jobs if job["source"] == "gt"]),
        "samples": samples,
        "limitations": [
            "This verifies frozen evidence and actual measured outcomes; verified does not mean every input succeeded or GT had no regressions.",
            "128 x 1 engineering transfer is not a 512-candidate success rate, fresh inference, new data generalization, or an official native-evaluator result.",
            "Sixteen development samples are reported separately from the remaining 112 transfer samples; all share the same saved 128-sample baseline.",
            "Native seed generation is bound by portable source, binary, input and result hashes; its internal computation is not independently reexecuted.",
            "Stored endpoint feasibility does not establish collision-free continuous motion.",
            "Solver timing and native seed generation are reported separately; no native seed generation cost is hidden.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "results", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "verification_output_already_exists")
    try:
        report = verify(args)
    except independent.VerificationError as exc:
        print(json.dumps({"status": "verification_failed", "code": exc.code,
                          "location": exc.location}), flush=True)
        raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(json.dumps({"status": "verification_failed",
                          "code": "malformed_or_unavailable_evidence",
                          "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], **report["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
